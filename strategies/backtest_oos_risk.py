#!/usr/bin/env python3
"""
Post-hoc RISK audit of the frozen OOS reversion result (research.md Finding 10).

Two deliverables, both over the reserved test window 2024-06-25 -> 2024-10-14:

  1. MARK-TO-MARKET drawdown. The frozen driver's `max_drawdown` is a realized-close
     cumsum ordered by decision_time: it books P&L only when a round-trip closes and
     never marks the up-to-18 *concurrently open* positions to market. During a cascade
     that runs against the reversion book, all open positions are underwater at once, so
     the true peak-to-trough is worse. Here we reconstruct a mark-to-mid equity curve
     (realized cash on closed legs + unrealized MtM on open legs, marked to the L1 mid at
     every book tick inside a busy span) and report the true max drawdown.

  2. HONEST SHARPE. The frozen annualized Sharpe (+4.46) is flattered by idle-day
     zero-padding, sqrt(365) on a 112-day/22-active-day sample, and treating overlapping
     positions as independent. We report the robust per-trade IR with HAC uncertainty and
     a block-bootstrap CI on the annualized number to show how wide it really is.

  3. PARTICIPATION-CAP SLIPPAGE STRESS (agreed re-touch of the spent holdout, labelled a
     stress). `participation < 1` caps how much of each aggressor print our order may take,
     so fills spread over the tape, VWAPs degrade, and thin windows leave partial/missed
     trades. We re-run the frozen config at alpha in {1.0 (control), 0.20, 0.10} and report
     how completion, VWAP, edge, t_NW and MtM drawdown move.

participation=1.0 is an exact no-op and MUST reproduce the frozen headline
(+48.85 bps, per-trade IR 0.467, realized DD -$6,571, total +$25,647) -- asserted below.

Usage:
  python strategies/backtest_oos_risk.py        # run from the repo root
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
import backtest_momentum as bm
from backtester import BookProvider
from significance import hac_stats

# Reuse the frozen OOS pipeline verbatim: same event construction, same causal trim,
# same book-feature pass, same windows. We only vary `participation`.
from backtest_oos_test import (
    HOLDING,
    LATENCY,
    TEST_START,
    TEST_END,
    LOAD_START,
    LOAD_END,
    add_book_features,
    causal_keep,
    peak_capital,
)

FEE_RATE = bt.TAKER_FEE_RATE
CONTRACT = bt.CONTRACT_NOTIONAL_USD  # $100 / contract
PARTICIPATIONS = (1.0, 0.20, 0.10)
STRESS_OUT = Path("data/results/oos_test_slippage_stress_trades.csv")


def _eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def run_size(liq: pl.DataFrame, agg: pl.DataFrame, notional: float,
             participation: float) -> pl.DataFrame:
    """Frozen Model-C 5-min tail reversion at one size and one participation cap."""
    res = bm.run_liquidation_momentum_model_c_backtests(
        liq_snap=liq, agg_trades=agg,
        holding_periods=(HOLDING,),
        trade_notional_usd=notional,
        signal_direction_sign=-1, percentile=0.98, upper_percentile=None,
        participation=participation,
        summary_csv_path=None, trades_csv_path=None, show_progress=False,
    )
    return res["trades"].with_columns(
        pl.col("decision_time").str.to_datetime(time_zone="UTC"),
        pl.col("liquidation_time").str.to_datetime(time_zone="UTC"),
        pl.col("entry_start_time").str.to_datetime(time_zone="UTC"),
        pl.col("entry_end_time").str.to_datetime(time_zone="UTC"),
        pl.col("exit_start_time").str.to_datetime(time_zone="UTC"),
        pl.col("exit_end_time").str.to_datetime(time_zone="UTC"),
    ).sort("decision_time")


# ---------------------------------------------------------------------------
# 1. Mark-to-market drawdown
# ---------------------------------------------------------------------------

def mtm_max_drawdown(trades: pl.DataFrame, book: BookProvider) -> dict[str, float]:
    """True max drawdown of a mark-to-mid equity curve for a book of round-trips.

    Accounting per trade: at entry_end_time realized cash drops by the entry fee and the
    position joins the open book; while open it is marked to the L1 mid; at exit_end_time
    realized cash gains (gross_pnl - exit_fee) and the position leaves the book. Summed over
    a trade's life this equals its net_pnl, so once every trade has closed the curve equals
    the realized total (asserted by the caller).

    Between consecutive trade-boundary timestamps the open set is constant, so unrealized
    equity is *linear in the mid*: equity(mid) = realized + A*mid - B, with
    A = sum(signed_qty * $100 / entry_vwap) and B = sum(signed_qty * $100) over open legs.
    We evaluate that at every book tick inside each busy span (exact to L1 resolution) and
    stream a running peak / trough to get the max drawdown without holding the whole curve.
    """
    df = trades.sort("entry_end_time")
    n = len(df)
    if n == 0:
        return {"max_dd_usd": 0.0, "final_equity_usd": 0.0, "n_ticks": 0.0}

    ee = df["entry_end_time"].to_list()
    xe = df["exit_end_time"].to_list()
    evwap = df["entry_vwap"].to_numpy()
    eqty = df["entry_quantity"].to_numpy()      # signed contracts
    xqty = df["exit_quantity"].to_numpy()       # signed, opposite sign
    gross = df["gross_pnl"].to_numpy()
    entry_fee = np.abs(eqty) * CONTRACT * FEE_RATE
    exit_fee = np.abs(xqty) * CONTRACT * FEE_RATE

    events: list[tuple] = []
    for i in range(n):
        events.append((ee[i], 0, i))  # open
        events.append((xe[i], 1, i))  # close
    # All events at an equal timestamp are applied as a batch before the boundary snapshot,
    # and segments use the pre-batch state, so the intra-tie open/close order does not affect
    # the result. (Contrast peak_capital, where the tie order genuinely matters.)
    events.sort(key=lambda e: (e[0], e[1]))

    realized = 0.0
    A = 0.0  # sum signed_qty * mult / entry_vwap over open legs
    B = 0.0  # sum signed_qty * mult over open legs
    n_open = 0
    mult = CONTRACT

    running_peak = 0.0   # equity starts flat at 0 before the first entry
    max_dd = 0.0
    n_ticks = 0

    def stream(arr: np.ndarray) -> None:
        nonlocal running_peak, max_dd
        if arr.size == 0:
            return
        cummax = np.maximum(np.maximum.accumulate(arr), running_peak)
        max_dd = min(max_dd, float((arr - cummax).min()))
        running_peak = float(cummax[-1])

    i = 0
    prev_t = events[0][0]
    while i < len(events):
        t = events[i][0]
        # Segment (prev_t, t] under the CURRENT (pre-event) open set.
        if t > prev_t and n_open > 0:
            w = book.window(prev_t, t)
            mids = w.mid_price
            if mids.size:
                stream(realized + A * mids - B)
                n_ticks += int(mids.size)
        # Apply every event stamped at t.
        while i < len(events) and events[i][0] == t:
            _, typ, k = events[i]
            if typ == 0:
                realized -= float(entry_fee[k])
                A += float(eqty[k]) * mult / float(evwap[k])
                B += float(eqty[k]) * mult
                n_open += 1
            else:
                realized += float(gross[k]) - float(exit_fee[k])
                A -= float(eqty[k]) * mult / float(evwap[k])
                B -= float(eqty[k]) * mult
                n_open -= 1
            i += 1
        # Boundary equity right after the state change at t.
        snap = book.as_of(t)
        eq_t = realized + (A * snap.mid_price - B if snap is not None and n_open > 0 else 0.0)
        stream(np.array([eq_t], dtype=float))
        prev_t = t

    return {"max_dd_usd": max_dd, "final_equity_usd": realized, "n_ticks": float(n_ticks)}


def realized_close_dd(trades: pl.DataFrame) -> float:
    """The frozen definition: cumsum of round-trip net_pnl ordered by decision_time."""
    eq = np.cumsum(trades.sort("decision_time")["net_pnl"].to_numpy())
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min()) if len(eq) else 0.0


# ---------------------------------------------------------------------------
# 2. Honest Sharpe
# ---------------------------------------------------------------------------

def daily_pnl_series(trades: pl.DataFrame) -> np.ndarray:
    """Net P&L per calendar day over the whole test span, idle days padded with 0
    (exactly the frozen `sharpe_block` construction)."""
    closes = trades["decision_time"].dt.offset_by("5m").dt.date()
    daily = (trades.with_columns(close_date=closes)
             .group_by("close_date").agg(pl.col("net_pnl").sum()).sort("close_date"))
    all_days = pl.date_range(TEST_START.date(), (TEST_END - timedelta(days=1)).date(),
                             interval="1d", eager=True)
    cal = (pl.DataFrame({"close_date": all_days})
           .join(daily, on="close_date", how="left").fill_null(0.0))
    return cal["net_pnl"].to_numpy()


def block_bootstrap_sharpe_ci(daily: np.ndarray, block: int = 7, n_boot: int = 5000,
                              seed: int = 0) -> tuple[float, float, float]:
    """Circular block-bootstrap CI on the sqrt(365)-annualized daily Sharpe. Blocks
    preserve short-horizon autocorrelation that the IID sqrt(365) scaling ignores."""
    rng = np.random.default_rng(seed)
    m = len(daily)
    n_blocks = int(np.ceil(m / block))
    ext = np.concatenate([daily, daily[:block]])  # wrap for circular blocks
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, m, size=n_blocks)
        samp = np.concatenate([ext[s:s + block] for s in starts])[:m]
        sd = samp.std(ddof=1)
        out[b] = samp.mean() / sd * np.sqrt(365) if sd > 0 else np.nan
    out = out[~np.isnan(out)]
    return (float(np.percentile(out, 5)), float(np.percentile(out, 50)),
            float(np.percentile(out, 95)))


def sharpe_report(kept: pl.DataFrame, notional: float) -> None:
    net_bps = (1e4 * kept["net_pnl"] / notional).to_numpy()
    s = hac_stats(kept.sort("decision_time").pipe(
        lambda d: (1e4 * d["net_pnl"] / notional).to_numpy()))
    daily = daily_pnl_series(kept)
    active = int((daily != 0).sum())
    sd = daily.std(ddof=1)
    ann_as_coded = float(daily.mean() / sd * np.sqrt(365)) if sd > 0 else float("nan")
    # Active-days-only variant: drop the zero days entirely, annualize by the active
    # cadence (active days per calendar year) rather than sqrt(365).
    act = daily[daily != 0]
    sd_a = act.std(ddof=1)
    per_active = float(act.mean() / sd_a) if sd_a > 0 else float("nan")
    ann_active = per_active * np.sqrt(active / (len(daily) / 365.0)) if sd_a > 0 else float("nan")
    lo, mid, hi = block_bootstrap_sharpe_ci(daily)

    print(f"\n=== HONEST SHARPE (filtered, ${notional/1000:.0f}k) ===")
    print(f"  per-trade IR (robust)   {net_bps.mean()/net_bps.std(ddof=1):+.3f}   "
          f"mean {s['mean_bps']:+.2f} bps  t_IID {s['t_iid']:+.2f}  t_NW {s['t_nw']:+.2f} "
          f"(L={int(s['nw_lag'])})")
    print(f"  ann Sharpe (as-coded)   {ann_as_coded:+.2f}   "
          f"[idle days=0, sqrt365, {active}/{len(daily)} active] -- the flattered number")
    print(f"  ann Sharpe (active-only){ann_active:+.2f}   [zero days dropped, cadence-annualized]")
    print(f"  ann Sharpe 90% CI       [{lo:+.2f}, {hi:+.2f}]  median {mid:+.2f}   "
          f"(block bootstrap, 7d blocks) -- width of the *as-coded* (flattered) estimator; "
          f"the interval inherits its upward bias, so read it as 'even generously framed, "
          f"the point estimate is this uncertain'")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(liq: pl.DataFrame, agg: pl.DataFrame, notional: float,
          participation: float, keep_ref: pl.DataFrame | None):
    """Run one (size, participation) cell; attach book features + the causal trim.

    The trim `keep` is fill-independent (it reads only the book), so at participation<1 we
    reuse the control's keep set by decision_time -- the SAME events, only the fills differ.
    """
    t = run_size(liq, agg, notional, participation)
    if keep_ref is None:
        t = add_book_features(t)
        keep = causal_keep(t)
        t = t.with_columns(keep=pl.Series(keep))
    else:
        t = t.join(keep_ref.select("decision_time", "keep"), on="decision_time", how="inner")
    return t.filter((pl.col("decision_time") >= TEST_START) & (pl.col("decision_time") < TEST_END))


def stress_row(kept: pl.DataFrame, notional: float, book: BookProvider,
               participation: float) -> dict:
    net_bps = (1e4 * kept["net_pnl"] / notional).to_numpy()
    # Newey-West is order-sensitive; the capped cells arrive via a polars join whose row
    # order is not guaranteed, so sort by decision_time before the HAC t-stat.
    s = hac_stats((1e4 * kept.sort("decision_time")["net_pnl"] / notional).to_numpy())
    ent_dur = (kept["entry_end_time"] - kept["entry_start_time"]).dt.total_milliseconds()
    dd = mtm_max_drawdown(kept, book)
    return {
        "participation": participation,
        "trades": len(kept),
        "entry_complete_%": round(100 * kept["entry_complete"].mean(), 1),
        "exit_complete_%": round(100 * kept["exit_complete"].mean(), 1),
        "entry_fill_ms_median": round(ent_dur.median(), 0),
        "entry_fill_ms_p99": round(ent_dur.quantile(0.99), 0),
        "net_bps": round(float(net_bps.mean()), 2),
        "t_nw": round(float(s["t_nw"]), 2),
        "total_net_usd": round(float(kept["net_pnl"].sum()), 0),
        "realized_dd_usd": round(realized_close_dd(kept), 0),
        "mtm_dd_usd": round(dd["max_dd_usd"], 0),
    }


def main() -> None:
    _eprint("loading liq + agg (warmup 2024-04-15 -> test end 2024-10-14) ...")
    liq = bt.load_liq_snap(start_date=LOAD_START, end_date=LOAD_END).collect().sort("time_datetime")
    agg = bt.load_agg_trades(start_date=LOAD_START, end_date=LOAD_END).collect().sort("transact_time_datetime")
    agg_qty = bm._same_direction_aggregate_quantities(liq_df=liq, trades_df=agg, seconds_before=5, seconds_after=5)
    liq = liq.with_columns(pl.Series("agg_qty_5s_before_5s_after", agg_qty))

    book = BookProvider()
    stress_records: list[dict] = []
    kept_50_by_part: dict[float, pl.DataFrame] = {}
    keep_ref = None

    for p in PARTICIPATIONS:
        _eprint(f"\n=== participation = {p} ===")
        all50 = build(liq, agg, 50_000.0, p, keep_ref)
        if keep_ref is None:
            keep_ref = all50  # control defines the kept event set (book-based trim)
        all100 = build(liq, agg, 100_000.0, p, keep_ref)
        # The frozen strategy is the TRIM-FILTERED book; select keep==True for all metrics.
        kept50 = all50.filter(pl.col("keep"))
        kept100 = all100.filter(pl.col("keep"))
        kept_50_by_part[p] = kept50
        stress_records.append(stress_row(kept50, 50_000.0, book, p))

        if p == 1.0:
            # ----- regression guard: control must reproduce the frozen headline -----
            net_bps = (1e4 * kept50["net_pnl"] / 50_000.0).to_numpy()
            assert abs(net_bps.mean() - 48.85) < 0.5, f"control net bps {net_bps.mean():.2f} != 48.85"
            assert abs(realized_close_dd(kept50) - (-6571)) < 50, "control realized DD drift"
            _eprint(f"  [guard OK] control net {net_bps.mean():+.2f} bps, "
                    f"realized DD ${realized_close_dd(kept50):,.0f}, total ${kept50['net_pnl'].sum():,.0f}")

            # ----- deliverable 1: mark-to-market drawdown at both sizes -----
            print("\n" + "#" * 78)
            print("# DELIVERABLE: MARK-TO-MARKET DRAWDOWN (frozen filtered strategy, control)")
            print("#" * 78)
            for notional, kdf in ((50_000.0, kept50), (100_000.0, kept100)):
                dd = mtm_max_drawdown(kdf, book)
                total = float(kdf["net_pnl"].sum())
                pk, pk_cap = peak_capital(kdf, notional)
                rc = realized_close_dd(kdf)
                # Invariant note: final_equity == sum(net_pnl) holds by construction (the A/B
                # marks cancel once a leg closes); it guards the fee/realized bookkeeping, not
                # the intra-segment mark formula -- the tests validate that.
                assert abs(dd["final_equity_usd"] - total) < 1.0, (
                    f"MtM curve final {dd['final_equity_usd']:.2f} != realized total {total:.2f}")
                # MtM DD <= realized-close DD is guaranteed only because the holding period is a
                # constant 5 min (decision-time order == exit-time order, so the MtM curve passes
                # through every realized-close level). It would not hold for a variable horizon.
                assert dd["max_dd_usd"] <= rc + 1e-6, "MtM DD must be <= realized-close DD"
                print(f"\n  -- ${notional/1000:.0f}k / trade  ({len(kdf)} trades, "
                      f"peak {pk} concurrent, peak capital ${pk_cap:,.0f}) --")
                print(f"  total net P&L            ${total:,.0f}")
                print(f"  realized-close max DD    ${rc:,.0f}   (frozen definition)")
                print(f"  MARK-TO-MARKET max DD    ${dd['max_dd_usd']:,.0f}   "
                      f"({dd['max_dd_usd']/pk_cap*100:+.2f}% of peak capital, "
                      f"{int(dd['n_ticks']):,} book ticks marked)")

            # ----- deliverable 2: honest Sharpe -----
            sharpe_report(kept50, 50_000.0)

    # ----- deliverable 3: participation-cap stress table -----
    print("\n" + "#" * 78)
    print("# SLIPPAGE STRESS -- participation cap on tape consumption ($50k, filtered)")
    print("#   (re-touches the spent holdout; reported as a stress, not a fresh OOS)")
    print("#" * 78 + "\n")
    stress = pl.DataFrame(stress_records)
    pl.Config.set_tbl_rows(20); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
    print(stress)

    STRESS_OUT.parent.mkdir(parents=True, exist_ok=True)
    # The control frame carries the book-feature columns (displacement/mid path) that the
    # capped frames lack, so align on the shared columns before concatenating.
    frames = [kept_50_by_part[p].with_columns(participation=pl.lit(p)) for p in PARTICIPATIONS]
    shared = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
    pl.concat([f.select(shared) for f in frames]).write_csv(STRESS_OUT)
    _eprint(f"\nwrote {STRESS_OUT}")


if __name__ == "__main__":
    main()
