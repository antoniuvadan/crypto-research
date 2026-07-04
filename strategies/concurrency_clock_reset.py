#!/usr/bin/env python3
"""
Clock-reset bounded concurrency cap on the frozen reversion strategy -- TRAIN ONLY (F13).

F12 tested the cap recommendation as a greedy FIRST-COME rule and rejected it: capping cut
absolute risk but return fell faster, because first-come anti-selects on edge (it keeps the
earliest, low-edge event of a cascade and discards the later, more-violent ones). This script
tests the user's THIRD rule, which F12 did not cover:

  Hold a bounded book of at most N same-direction units. When a new qualifying tail signal
  arrives while the book is open, RESET the shared 5-min exit clock (extend the hold) instead
  of piling on beyond N or ignoring it. Build up to N units (one per new signal), then beyond
  N each new signal only resets the clock. All units flatten together when the clock finally
  expires (last qualifying signal + 5min + latency).

The rule keeps leverage bounded (the cap's risk win) but USES the later cascade events -- not to
add size, but to hold through more of the reversion (F5: holding is the edge). It is fully
CAUSAL/deployable: the exit fires 5 min after the last *observed* signal (you defer the resting
exit whenever a fresh signal arrives) -- no look-ahead.

Modelling: clock-reset changes EXIT TIMES, so exits are re-simulated on the tape with the same
taker sweep (`_sweep_fill_from_agg_trades`) the frozen trades used. Entry fills are unaffected by
the exit clock, so the frozen `entry_*` columns are reused verbatim; only the exit is re-run.
Runs are built WITHIN each window separately (dev / holdout) so a holdout signal can never extend
a dev run -- the F12 dev-select / holdout-confirm discipline is preserved.

Reported per policy and window (identical battery to F12): net bps, per-trade IR, Newey-West t,
episode-cluster t, total $, peak concurrency + capital, sqrt(365) Sharpe, realized-close AND
mark-to-market max drawdown, ret/|MtM DD|, P&L/DD concentration, 6-way cost decomposition; plus a
hold-time distribution diagnostic. Benchmarked head-to-head against F12's uncapped and greedy cap.

Usage (repo root):
  ~/miniforge3/envs/mscf/bin/python strategies/concurrency_clock_reset.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

import backtester as bt
from backtester import (
    BookProvider, _sweep_fill_from_agg_trades, _linear_contract_pnl_usd,
    TAKER_FEE_RATE, CONTRACT_NOTIONAL_USD,
)
from concurrency_analysis import cap_at_n, load, TRAIN_CSV, net_bps
from concurrency_train_caps import (
    NOTIONAL, DEV_START, HOLDOUT_START, TRAIN_END,
    split_dev_holdout, metrics_row, concentration_row, decompose_window, cost_row,
    select_cap, _print_table,
)
from backtest_oos_test import HOLDING, LATENCY
from gen_train_trades import TRAIN_START, TRAIN_END as TRAIN_END_DATE

CONTRACT = CONTRACT_NOTIONAL_USD  # $100 / contract
FEE_RATE = TAKER_FEE_RATE         # 5 bps taker / leg
NS = (5, 3, 1)                    # clock-reset cap values (N=1 = single-unit extend-only)
OUT_CSV = Path("data/results/concurrency_clock_reset_summary.csv")

# columns of the frozen train trade that a clock-reset unit reuses unchanged (entries + meta)
_REUSE = ["decision_time", "liquidation_time", "direction", "entry_start_time",
          "entry_end_time", "entry_quantity", "entry_vwap", "displacement_bps"]


def _eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# aggTrades tape (built exactly as backtest_momentum's runner, so ns keys match)
# ---------------------------------------------------------------------------

def load_tape() -> tuple:
    _eprint(f"loading train aggTrades {TRAIN_START} -> {TRAIN_END_DATE} for exit re-sim ...")
    agg = (bt.load_agg_trades(start_date=TRAIN_START, end_date=TRAIN_END_DATE)
           .collect().sort("transact_time_datetime"))
    trade_times_ns = agg["transact_time_datetime"].cast(pl.Int64).to_numpy()
    trade_times = agg["transact_time_datetime"].to_list()
    prices = agg["price"].to_numpy()
    quantities = agg["quantity"].to_numpy()
    is_buyer_maker = agg["is_buyer_maker"].to_numpy()
    return trade_times_ns, trade_times, prices, quantities, is_buyer_maker


# ---------------------------------------------------------------------------
# Clock-reset run builder -- reuse entries, re-simulate the shared exit
# ---------------------------------------------------------------------------

def build_clock_reset_trades(events: pl.DataFrame, n: int, tape: tuple,
                             holding: timedelta = HOLDING, latency: timedelta = LATENCY) -> pl.DataFrame:
    """Forward-pass the kept events (within ONE window) into bounded books with a shared,
    signal-reset exit clock; re-simulate one aggregated exit sweep per book; emit one row per
    UNIT so the F12 battery sees the book as <=n overlapping legs sharing a common exit.

    Causal: a book's units and its exit time depend only on signals up to the current clock; a
    signal arriving after the clock has expired starts a fresh book. Directions are processed
    independently (a run is one-directional; opposite-direction books may coexist, honestly
    hedged rather than one-way leverage)."""
    rows: list[dict] = []
    run_id = 0

    def finalize(run: dict) -> None:
        nonlocal run_id
        book_qty = float(sum(u["entry_quantity"] for u in run["units"]))
        exit_start = run["exit_decision"] + latency
        ex = _sweep_fill_from_agg_trades(-book_qty, exit_start, *tape)
        run_id += 1
        if ex.avg_price is None:  # no exit liquidity found (should not happen on train); drop run
            _eprint(f"  [warn] no exit fill for run at {exit_start} (book_qty={book_qty}); dropped")
            return
        fill_ratio = min(1.0, abs(ex.filled_quantity) / abs(book_qty)) if book_qty else 0.0
        hold_min = (run["exit_decision"] - run["first_signal"]).total_seconds() / 60.0
        n_units = len(run["units"])
        n_resets = run["run_events"] - 1
        for u in run["units"]:
            eq = float(u["entry_quantity"])
            signed_closed = (abs(eq) * fill_ratio) * (1.0 if eq > 0 else -1.0)
            gross = _linear_contract_pnl_usd(signed_closed, float(u["entry_vwap"]),
                                             float(ex.avg_price), CONTRACT)
            exit_qty = -signed_closed
            fees = (abs(eq) + abs(exit_qty)) * CONTRACT * FEE_RATE
            rows.append({
                **{k: u[k] for k in _REUSE},
                "exit_start_time": exit_start,
                "exit_end_time": ex.end_time,
                "exit_quantity": exit_qty,
                "exit_vwap": float(ex.avg_price),
                "exit_complete": bool(ex.is_complete),
                "gross_pnl": gross,
                "fees_paid": fees,
                "net_pnl": gross - fees,
                "run_id": run_id,
                "run_events": run["run_events"],
                "n_units": n_units,
                "hold_minutes": hold_min,
                "n_resets": n_resets,
            })

    for d in (1, -1):
        recs = events.filter(pl.col("direction") == d).sort("decision_time").to_dicts()
        cur: dict | None = None
        for r in recs:
            t = r["decision_time"]
            if cur is None or t > cur["exit_decision"]:
                if cur is not None:
                    finalize(cur)
                cur = {"units": [r], "first_signal": t, "last_signal": t,
                       "exit_decision": t + holding, "run_events": 1}
            else:  # arrives before the clock expires -> reset (extend), build up to n
                cur["last_signal"] = t
                cur["exit_decision"] = t + holding
                cur["run_events"] += 1
                if len(cur["units"]) < n:
                    cur["units"].append(r)
        if cur is not None:
            finalize(cur)

    df = pl.DataFrame(rows)
    return df.sort("decision_time") if len(df) else df  # sort() errors on a schema-less empty frame


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def hold_stats(df: pl.DataFrame) -> dict:
    """Per-run hold-time distribution (the first-order check on F4's 30->60min edge fade)."""
    runs = df.group_by("run_id").agg(
        pl.col("hold_minutes").first(), pl.col("run_events").first(),
        pl.col("n_units").first(), pl.col("n_resets").first()).sort("run_id")
    hm = runs["hold_minutes"].to_numpy()
    return {
        "runs": runs.height,
        "multi_event_run_%": round(100 * float((runs["run_events"] > 1).mean()), 1),
        "median_hold_min": round(float(np.median(hm)), 1),
        "p90_hold_min": round(float(np.percentile(hm, 90)), 1),
        "max_hold_min": round(float(hm.max()), 1),
        "mean_resets_per_run": round(float(runs["n_resets"].mean()), 2),
    }


def window_report(name: str, kept: pl.DataFrame, tape: tuple, book: BookProvider,
                  span_start, span_end) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    print("\n" + "=" * 92)
    print(f"# {name}   ({len(kept)} kept trades, span {span_start.date()} -> {span_end.date()})")
    print("=" * 92)

    # policies: F12 benchmarks (frozen fixed-5-min trades) + clock-reset (re-simulated exits).
    # kept_t carries a stable _tid so the greedy subsets share ONE cost-decomposition of kept.
    kept_t = kept.with_row_index("_tid")
    frozen: list[tuple[str, pl.DataFrame]] = [("uncapped (N=inf)", kept_t)]
    for n in NS:
        frozen.append((f"greedy cap N={n}", cap_at_n(kept_t, n)))
    reset = [(f"clock-reset N={n}", build_clock_reset_trades(kept, n, tape).with_row_index("_tid"))
             for n in NS]
    pol = frozen + reset

    _eprint(f"  [{name}] metrics (MtM on) ...")
    mrows, crows, hrows = [], [], []
    for lbl, sub in pol:
        mrows.append(metrics_row(lbl, sub, book, span_start, span_end))
        crows.append(concentration_row(lbl, sub, book,
                     mrows[-1]["mtm_dd_usd"] if mrows[-1]["mtm_dd_usd"] is not None else 0.0))
        hrows.append({"policy": lbl, **hold_stats(sub)} if "clock-reset" in lbl
                     else {"policy": lbl, "runs": None})
    _print_table("[metrics]  net_bps / IR / t_nw / t_cluster / peak cap / Sharpe(sqrt365) / "
                 "realized+MtM DD / ret_over_mtmdd", mrows)
    _print_table("[concentration]  P&L share by top day / top-3 / top episode; MtM-DD top-day "
                 "share; dd_trough_day", crows)
    _print_table("[hold-time]  clock-reset run distribution (fixed-5min benchmarks have runs=None)",
                 hrows)

    _eprint(f"  [{name}] cost decomposition (book pass) ...")
    comp_frozen = decompose_window(kept_t)  # ONE pass shared by uncapped + greedy caps
    drows = [cost_row(lbl, sub, comp_frozen) for lbl, sub in frozen]
    drows += [cost_row(lbl, sub, decompose_window(sub)) for lbl, sub in reset]
    _print_table("[cost] mean bps per component (pnl_decomposition convention)", drows)
    return mrows, crows, hrows, drows


def head_to_head(dev_m: list[dict], ho_m: list[dict], sel_reset: str) -> None:
    print("\n" + "#" * 92)
    print("# HEAD-TO-HEAD + CLOCK-RESET SELECTION (dev-only) + HOLDOUT CONFIRMATION")
    print("#" * 92)

    def by(rows, name):
        return next(r for r in rows if r["policy"] == name)

    print("\n  Does clock-reset recover the edge the greedy cap discarded, at bounded capital?")
    print("  NOTE: greedy caps TOTAL concurrency at N; clock-reset caps N PER DIRECTION, so it can"
          "\n  hold up to 2N units (see peak_concurrent / peakcap) -- clock-reset runs with the "
          "LOOSER\n  budget, so any failure to out-earn greedy is not a capital handicap.")
    for n in NS:
        for win, mr in (("DEV", dev_m), ("HOLDOUT", ho_m)):
            u, g, c = by(mr, "uncapped (N=inf)"), by(mr, f"greedy cap N={n}"), by(mr, f"clock-reset N={n}")
            print(f"  {win:<7} N={n}: uncapped {u['net_bps']:+.2f}bps/t{u['t_nw']}/DD${u['mtm_dd_usd']:,.0f} "
                  f"| greedy {g['net_bps']:+.2f}/t{g['t_nw']}/DD${g['mtm_dd_usd']:,.0f} "
                  f"| CLOCK-RESET {c['net_bps']:+.2f}/t{c['t_nw']}/DD${c['mtm_dd_usd']:,.0f}/"
                  f"retDD{c['ret_over_mtmdd']} (peakcap ${c['peak_capital_usd']:,.0f})")

    print(f"\n  DEV-selected clock-reset policy (max sqrt365 Sharpe among t_NW>=2, tie->smaller "
          f"MtM DD): {sel_reset!r}")
    ds, hs = by(dev_m, sel_reset), by(ho_m, sel_reset)
    for tag, r in (("DEV    ", ds), ("HOLDOUT", hs)):
        print(f"    {tag}: net {r['net_bps']:+.2f} bps  t_NW {r['t_nw']}  t_clust {r['t_cluster']}  "
              f"Sharpe {r['ann_sharpe']}  MtM DD ${r['mtm_dd_usd']:,.0f}  ret/|MtMDD| "
              f"{r['ret_over_mtmdd']}  ({r['trades']} units, peak cap ${r['peak_capital_usd']:,.0f})")
    print("  CAVEAT: the selection gate uses t_NW, which OVERSTATES significance for a bounded book"
          "\n  whose units share one exit (near-perfectly correlated). Under the honest episode-"
          "clustered\n  t, the dev-selected N=5 is t_clust ~1.3 -- NOT dev-significant even before "
          "the holdout, so\n  its holdout t_NW 1.52 failure is expected. No clock-reset variant is "
          "robustly significant.")


def main() -> None:
    if not TRAIN_CSV.exists():
        sys.exit(f"train CSV {TRAIN_CSV} not found -- run strategies/gen_train_trades.py first")

    book = BookProvider()
    tape = load_tape()
    full = load(TRAIN_CSV)  # keep==True, tz-parsed, sorted by decision
    dev, holdout = split_dev_holdout(full)
    _eprint(f"loaded {len(full)} kept train trades: dev {len(dev)}, holdout {len(holdout)}")

    dev_m, dev_c, dev_h, dev_d = window_report("DEV 2023-06-25..2024-02-24", dev, tape, book,
                                               DEV_START, HOLDOUT_START)
    ho_m, ho_c, ho_h, ho_d = window_report("HOLDOUT 2024-02-25..2024-06-24", holdout, tape, book,
                                           HOLDOUT_START, TRAIN_END)

    # select among the clock-reset policies only (the new family); benchmarks are context
    reset_dev = [r for r in dev_m if r["policy"].startswith("clock-reset")]
    sel = select_cap(reset_dev)
    head_to_head(dev_m, ho_m, sel["policy"])

    merged: list[dict] = []
    for win, mr, cr, hr, dr in (("dev", dev_m, dev_c, dev_h, dev_d),
                                ("holdout", ho_m, ho_c, ho_h, ho_d)):
        cmap = {r["policy"]: r for r in cr}
        hmap = {r["policy"]: r for r in hr}
        dmap = {r["policy"]: r for r in dr}
        for m in mr:
            p = m["policy"]
            merged.append({"window": win, **m,
                           **{k: v for k, v in cmap[p].items() if k != "policy"},
                           **{f"hold_{k}": v for k, v in hmap[p].items() if k != "policy"},
                           **{f"cost_{k}": v for k, v in dmap[p].items() if k != "policy"}})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(merged).write_csv(OUT_CSV)
    _eprint(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
