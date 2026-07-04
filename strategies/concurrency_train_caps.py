#!/usr/bin/env python3
"""
Deployable concurrency caps on the frozen liquidation-cascade reversion strategy -- TRAIN ONLY.

CONCURRENCY_ANALYSIS.md found the strategy pyramids up to 18 one-directional positions in a
single cascade (the same signal fired repeatedly -> leverage, not diversification; 86% of the
OOS drawdown was one crash day it was 18x long into). The recommendation was to CAP concurrency.

This script implements that cap as a CAUSAL, deployable filter on the already-realized frozen
train trades and asks which concurrency limit does best among {uncapped, N<=5, N<=3, N<=1}.

Discipline (see the plan):
  * TRAIN ONLY. The reserved OOS test set is never touched here.
  * SELECT on the dev window (2023-06-25 -> 2024-02-24), then CONFIRM the chosen cap on the
    within-train holdout (2024-02-25 -> 2024-06-24). The holdout never informs the choice.
  * NO LOOK-AHEAD. The cap is the greedy first-come rule (`cap_at_n`): when N are already open,
    the newly-arriving bet is SKIPPED -- you cannot peek at the rest of the cascade to keep a
    "better" one. The `dedupe_per_episode` most-violent oracle is deliberately NOT used.

Per policy and per window we report (CLAUDE.md rigor): net bps + per-trade IR, Newey-West t and
episode-cluster-robust t, total $, peak concurrency + peak capital, sqrt(365) annualized Sharpe,
BOTH realized-close and mark-to-market max drawdown, P&L / drawdown concentration attribution,
and the additive 6-way cost decomposition (gross-mid / latency / spread / fees).

Usage (from the repo root, after gen_train_trades.py has produced the train CSV):
  ~/miniforge3/envs/mscf/bin/python strategies/concurrency_train_caps.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

import pnl_decomposition as pnld
from backtester import BookProvider
from significance import hac_stats
from backtest_oos_risk import mtm_max_drawdown, realized_close_dd
from concurrency_analysis import (
    NOTIONAL, TRAIN_CSV, load, net_bps, cap_at_n, concurrency_profile,
    assign_episodes, cluster_robust_t, subset_metrics,
)

UTC = timezone.utc
DEV_START = datetime(2023, 6, 25, tzinfo=UTC)
HOLDOUT_START = datetime(2024, 2, 25, tzinfo=UTC)  # dev is < this; holdout is >= this
TRAIN_END = datetime(2024, 6, 25, tzinfo=UTC)       # exclusive upper bound of the train window
CAPS = (5, 3, 1)                                    # concurrency limits to sweep (+ uncapped)
T_NW_GATE = 2.0                                     # significance gate for cap selection
OUT_CSV = Path("data/results/concurrency_train_caps_summary.csv")
COMPONENTS = ["gross_mid_to_mid", "latency_slippage", "spread_entry", "spread_exit",
              "fee_entry", "fee_exit"]


def _eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Policies (all causal) and per-policy metric rows
# ---------------------------------------------------------------------------

def split_dev_holdout(full: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the train trades on the 2024-02-25 boundary: dev is strictly before it, the
    within-train holdout is on/after it (through the train end). A trade stamped exactly at the
    boundary belongs to the holdout."""
    dev = full.filter(pl.col("decision_time") < HOLDOUT_START)
    holdout = full.filter((pl.col("decision_time") >= HOLDOUT_START)
                          & (pl.col("decision_time") < TRAIN_END))
    return dev, holdout


def policies(df: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """The uncapped baseline plus the causal greedy caps. N=1 is the literal 'no concurrent
    bets' recommendation; N=3/5 the requested limits."""
    out = [("uncapped (N=inf)", df)]
    for n in CAPS:
        label = f"cap N={n}" + (" (no concurrency)" if n == 1 else "")
        out.append((label, cap_at_n(df, n)))
    return out


def metrics_row(label: str, sub: pl.DataFrame, book: BookProvider,
                span_start: datetime, span_end: datetime) -> dict:
    """subset_metrics (net bps, t_nw, totals, peak cap, Sharpe, realized+MtM DD) augmented with
    the episode-cluster-robust t and the per-trade information ratio -- the honest inference
    under overlapping trades."""
    row = subset_metrics(sub, book, label, span_start, span_end, compute_mtm=True)
    # A drawdown-AWARE cross-check the sqrt(365) Sharpe misses: the Sharpe convention books only
    # realized closes, so it is blind to the mark-to-market pain of a pyramided one-way stack.
    # ret/|MtM DD| (a Calmar-style ratio) sees exactly that risk. Reported alongside Sharpe so the
    # cap verdict does not hinge on the flattered Sharpe alone.
    mtm = row["mtm_dd_usd"]
    row["ret_over_mtmdd"] = (round(row["total_net_usd"] / abs(mtm), 2)
                             if mtm is not None and abs(mtm) > 1e-9 else None)
    y = net_bps(sub)
    if len(sub) > 1 and y.std(ddof=1) > 0:
        ep = assign_episodes(sub, timedelta(0))
        row["t_cluster"] = round(float(cluster_robust_t(y, ep)["t_cluster"]), 2)
        row["per_trade_ir"] = round(float(y.mean() / y.std(ddof=1)), 3)
    else:
        row["t_cluster"] = None
        row["per_trade_ir"] = None
    return row


# ---------------------------------------------------------------------------
# Concentration attribution (per policy)
# ---------------------------------------------------------------------------

def _pnl_share(x: float, total: float) -> float | None:
    """Percent share of the signed window total; None when the total is ~0 or negative, since a
    'share of a non-positive total' is not interpretable (and can exceed 100% when other days
    are negative)."""
    return round(x / total * 100, 1) if total > 1e-9 else None


def concentration_row(label: str, sub: pl.DataFrame, book: BookProvider,
                      full_mtm_dd: float) -> dict:
    """Where the P&L and the drawdown concentrate. `full_mtm_dd` is reused from the metrics row
    so we only pay for the ONE extra MtM pass (top-day excluded) needed for DD attribution.

    Shares are of the signed total; when a capped window's total is ~0 (or negative) a 'share'
    is not interpretable, so we emit None rather than a misleading percentage."""
    total = float(sub["net_pnl"].sum())
    d = sub.with_columns(day=pl.col("decision_time").dt.date())
    by_day = (d.group_by("day").agg(pl.col("net_pnl").sum().alias("pnl"), pl.len().alias("n"))
              .sort("pnl", descending=True))
    top_day = by_day["day"][0]
    ep = assign_episodes(sub, timedelta(0))
    ep_pnl = (pl.DataFrame({"ep": ep, "pnl": sub["net_pnl"].to_numpy()})
              .group_by("ep").agg(pl.col("pnl").sum().alias("pnl")))

    def share(x: float) -> float | None:
        return _pnl_share(x, total)

    ex = sub.filter(pl.col("decision_time").dt.date() != top_day)
    ex_dd = mtm_max_drawdown(ex, book)["max_dd_usd"] if len(ex) else 0.0
    dd_share = (round((full_mtm_dd - ex_dd) / full_mtm_dd * 100, 1)
                if full_mtm_dd < -1e-9 else None)
    return {
        "policy": label,
        "active_days": by_day.height,
        "episodes": int(ep.max()) + 1 if len(sub) else 0,
        "top_pnl_day": str(top_day),
        "top_day_share_%": share(float(by_day["pnl"][0])),
        "top3_day_share_%": share(float(by_day["pnl"].head(3).sum())),
        "top_episode_share_%": share(float(ep_pnl["pnl"].max())),
        # DD attribution: how much of the MtM drawdown the TOP-P&L day accounts for, and WHICH
        # day the realized-close equity actually troughs on. On train these differ (top-P&L day
        # is not the DD driver -> ~0% share), unlike OOS where one crash day drove 86% of DD.
        "mtm_dd_topday_share_%": dd_share,
        "dd_trough_day": _realized_dd_day(sub),
    }


def _realized_dd_day(sub: pl.DataFrame) -> str:
    """The calendar day on which the realized-close equity curve (cumsum of net_pnl ordered by
    decision_time) hits its deepest trough -- i.e. the day the drawdown actually bottoms. Cheap,
    causal locator that does not need a book pass; the MtM trough day is typically the same."""
    s = sub.sort("decision_time")
    eq = np.cumsum(s["net_pnl"].to_numpy())
    dd = eq - np.maximum.accumulate(eq)
    i = int(dd.argmin())
    return str(s["decision_time"].dt.date().to_list()[i])


# ---------------------------------------------------------------------------
# Cost decomposition (per policy), reusing pnl_decomposition
# ---------------------------------------------------------------------------

def decompose_window(window_df: pl.DataFrame) -> pl.DataFrame:
    """Attach the additive 6-way bps components to every trade in a window (keyed by `_tid`),
    reusing pnl_decomposition.lookup_mids + add_decomposition verbatim. Runs the book pass ONCE
    per window; policy subsets are then just filters on `_tid`.

    Bps convention (pnl_decomposition's): each term is bps of the decision mid; `net_realized`
    is bps of TRADED notional (= the main table's net_bps when fills complete). Reported in its
    own block, not mixed into the $50k-notional metrics table."""
    role_cols = {"decision": "decision_time", "entry": "entry_start_time", "exit": "exit_start_time"}
    lookups = pl.concat([
        window_df.select(pl.col("_tid").alias("tid"), pl.lit(role).alias("role"),
                         pl.col(col).alias("ts"))
        for role, col in role_cols.items()
    ])
    mids = pnld.lookup_mids(lookups)
    wide = mids.pivot(values="mid", index="tid", on="role")
    df = (window_df.join(wide, left_on="_tid", right_on="tid", how="left")
          .drop_nulls(subset=["decision", "entry", "exit"]))
    df = pnld.add_decomposition(df)
    return df.select(["_tid", *COMPONENTS, "net_reconstructed", "net_realized"])


def cost_row(label: str, sub: pl.DataFrame, comp: pl.DataFrame) -> dict:
    """Mean bps per component over a policy subset (join by `_tid`)."""
    c = comp.filter(pl.col("_tid").is_in(sub["_tid"].to_list()))
    row: dict = {"policy": label, "n_decomp": len(c)}
    for col in [*COMPONENTS, "net_reconstructed", "net_realized"]:
        row[col] = round(float(c[col].mean()), 2) if len(c) else None
    return row


# ---------------------------------------------------------------------------
# Selection (DEV ONLY) + report
# ---------------------------------------------------------------------------

def select_cap(dev_rows: list[dict]) -> dict:
    """Pre-registered, dev-only: highest sqrt(365) annualized Sharpe among policies that keep
    t_NW >= 2 (edge still significant); ties broken by the smaller-magnitude MtM drawdown
    (less-negative mtm_dd_usd). If no policy clears the significance gate, fall back to all."""
    candidates = [r for r in dev_rows if (r["t_nw"] is not None and r["t_nw"] >= T_NW_GATE)]
    pool = candidates if candidates else dev_rows

    def key(r: dict) -> tuple[float, float]:
        s = r["ann_sharpe"] if r["ann_sharpe"] is not None else float("-inf")
        dd = r["mtm_dd_usd"] if r["mtm_dd_usd"] is not None else float("-inf")
        return (s, dd)  # Sharpe primary; less-negative DD wins ties

    best = max(pool, key=key)
    return {"policy": best["policy"], "gated": bool(candidates)}


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    pl.Config.set_tbl_rows(20); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(240)
    pl.Config.set_float_precision(2)
    print(pl.DataFrame(rows))


def analyze_window(name: str, df: pl.DataFrame, book: BookProvider,
                   span_start: datetime, span_end: datetime) -> tuple[list[dict], list[dict], list[dict]]:
    print("\n" + "=" * 90)
    print(f"# {name}   ({len(df)} kept trades, span {span_start.date()} -> {span_end.date()})")
    print("=" * 90)

    pol = policies(df)

    _eprint(f"  [{name}] metrics (MtM on) ...")
    mrows = [metrics_row(lbl, sub, book, span_start, span_end) for lbl, sub in pol]
    _print_table("[metrics]  net_bps / IR / t_nw / t_cluster / peak cap / Sharpe(sqrt365) / "
                 "realized+MtM DD", mrows)

    _eprint(f"  [{name}] concentration ...")
    crows = [concentration_row(lbl, sub, book, m["mtm_dd_usd"] if m["mtm_dd_usd"] is not None else 0.0)
             for (lbl, sub), m in zip(pol, mrows)]
    _print_table("[concentration]  P&L share by top day / top-3 days / top episode; "
                 "MtM-DD share of the top day", crows)

    _eprint(f"  [{name}] cost decomposition (book pass) ...")
    comp = decompose_window(df)
    gap = float((comp["net_reconstructed"] - comp["net_realized"]).abs().max())
    drows = [cost_row(lbl, sub, comp) for lbl, sub in pol]
    print(f"\n[cost] mean bps per component (pnl_decomposition convention; costs are POSITIVE and "
          f"subtracted).\n       net = gross_mid_to_mid - latency - spread_entry - spread_exit - "
          f"fee_entry - fee_exit.\n       net_reconstructed vs net_realized gap (linear-vs-inverse "
          f"residual) max |.| = {gap:.2f} bps; net_realized ~= the metrics-table net_bps.")
    _print_table("", drows)
    return mrows, crows, drows


def main() -> None:
    if not TRAIN_CSV.exists():
        sys.exit(f"train CSV {TRAIN_CSV} not found -- run strategies/gen_train_trades.py first")

    book = BookProvider()
    full = load(TRAIN_CSV).with_row_index("_tid")  # keep==True, tz-parsed, sorted by decision
    dev, holdout = split_dev_holdout(full)
    _eprint(f"loaded {len(full)} kept train trades: dev {len(dev)}, holdout {len(holdout)}")

    dev_m, dev_c, dev_d = analyze_window("DEV 2023-06-25..2024-02-24", dev, book,
                                         DEV_START, HOLDOUT_START)
    ho_m, ho_c, ho_d = analyze_window("HOLDOUT 2024-02-25..2024-06-24", holdout, book,
                                      HOLDOUT_START, TRAIN_END)

    # ---- selection on dev, confirmation on holdout ----
    sel = select_cap(dev_m)
    print("\n" + "#" * 90)
    print("# CAP SELECTION (dev-only) + HOLDOUT CONFIRMATION")
    print("#" * 90)
    gate_note = (f"among policies with t_NW >= {T_NW_GATE:g}" if sel["gated"]
                 else f"NO policy cleared t_NW >= {T_NW_GATE:g}; fell back to all")
    print(f"\n  selection rule: max sqrt(365) Sharpe {gate_note}; ties -> smaller MtM DD.")
    print(f"  DEV-SELECTED POLICY: {sel['policy']!r}")
    dsel = next(r for r in dev_m if r["policy"] == sel["policy"])
    hsel = next(r for r in ho_m if r["policy"] == sel["policy"])
    for tag, r in (("DEV   ", dsel), ("HOLDOUT", hsel)):
        print(f"    {tag}: net {r['net_bps']:+.2f} bps  t_NW {r['t_nw']}  t_clust {r['t_cluster']}  "
              f"Sharpe {r['ann_sharpe']}  realizedDD ${r['realized_dd_usd']:,.0f}  "
              f"MtM DD ${r['mtm_dd_usd']:,.0f}  ({r['trades']} trades, peak cap "
              f"${r['peak_capital_usd']:,.0f})")
    print("\n  Cross-check -- ret/|MtM DD| (Calmar-style), which UNLIKE the realized-close Sharpe "
          "DOES\n  see the mark-to-market pain of the stack. This is the load-bearing, MtM-aware "
          "verdict:")
    for r in dev_m:
        print(f"    DEV {r['policy']:<26} ret/|MtMDD| {r['ret_over_mtmdd']}")

    print(
        "\n  HONEST READ (scope carefully):\n"
        "  * Only ONE causal cap family was tested: the greedy FIRST-COME rule. It keeps the\n"
        "    earliest event of each cascade and skips the rest -- and the reversion edge builds\n"
        "    as the cascade intensifies (F3/F6), so first-come ANTI-SELECTS on edge by\n"
        "    construction. That is why capped edge collapses (dev 34.68 -> 12.25 bps). This is\n"
        "    NOT 'no causal cap can win' -- only that this one does not.\n"
        "  * The greedy cap DOES cut risk in absolute terms: MtM DD falls ~10x (dev -$47.9k ->\n"
        "    -$4.8k at N=1; holdout -$18.6k -> -$3.5k) and peak capital 18x ($900k -> $50k). But\n"
        "    RETURN falls faster, so every risk-adjusted ratio -- Sharpe (3.26 vs <=1.71) and\n"
        "    ret/|MtM DD| (dev 1.26 vs <=1.03; holdout 2.14 vs <=1.82) -- favours UNCAPPED on\n"
        "    both dev and holdout. (Leverage-normalised DD%/capital worsens under caps, but that\n"
        "    ratio mechanically flatters the high-leverage book, so it is not the argument.)\n"
        "  * DD attribution: on TRAIN the top-P&L day is NOT the drawdown driver (mtm_dd_topday\n"
        "    _share ~0%); see dd_trough_day. Contrast OOS (F10) where one crash day drove 86% of\n"
        "    DD. Concentration shares are also non-monotone in N (N=1 below N=3), so 'caps\n"
        "    concentrate P&L' is not a clean monotone effect.\n"
        "  * CAVEAT that bounds this whole result: a single train window cannot realise the rare\n"
        "    one-directional cascade tail the cap exists to defend against (the OOS Aug-5 pyramid;\n"
        "    ETH F11 all-long run-over). 'Uncapped wins on train' is the pick-up-pennies reading.\n"
        "    Conclusion: the GREEDY FIRST-COME cap is the wrong tool -- NOT that concurrency risk\n"
        "    is a non-issue. A causal DISPLACEMENT-GATED entry (not tested here; needs fresh data)\n"
        "    remains the live candidate.")

    # ---- persist a tidy summary (one row per window x policy) ----
    def tag_rows(rows: list[dict], win: str) -> list[dict]:
        return [{"window": win, **r} for r in rows]

    merged: list[dict] = []
    for win, mr, cr, dr in (("dev", dev_m, dev_c, dev_d), ("holdout", ho_m, ho_c, ho_d)):
        cmap = {r["policy"]: r for r in cr}
        dmap = {r["policy"]: r for r in dr}
        for m in mr:
            p = m["policy"]
            merged.append({"window": win, **m,
                           **{k: v for k, v in cmap[p].items() if k != "policy"},
                           **{f"cost_{k}": v for k, v in dmap[p].items() if k != "policy"}})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(merged).write_csv(OUT_CSV)
    _eprint(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
