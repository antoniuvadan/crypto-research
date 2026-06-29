#!/usr/bin/env python3
"""
Pre-OOS de-risking of the frozen candidate (trimmed 5-min tail reversion).

Before spending the one-shot test period, confirm the +51 bps within-train holdout
(Finding 7) is not knife-edge on parameters we asserted but never tested, and that
it survives a conservative (pessimistic) fill model. All checks are DEV/HOLDOUT
(within-train) only -- the test period (2024-06-25 ->) is NOT touched.

Two checks (the neutralization-window sweep is run separately via market_neutralize.py):

  1. TRIM-FILTER ROBUSTNESS -- sweep the causal trailing-median displacement filter's
     lookback W in {15,30,45,60}d and min_history in {10,15,25}. F7 fixed W=30d,
     min_hist=15 from Finding 6 with no holdout tuning; this shows the holdout edge is
     stable across the neighbourhood, not a single lucky setting.

  2. PESSIMISTIC FILL FLOOR -- the headline net rests on the optimistic aggTrades
     sweep, whose spread terms came out slightly *favorable* (F4: -2.0 bps combined).
     Recompute net charging a conservative flat 2 bps/side taker spread instead, on
     the trim-kept dev/holdout subsets. This is the honest floor to carry to the OOS.

Reuses the decomposition trades (per-trade bps already split into gross_mid_to_mid /
latency / spread / fees); displacement for the trim filter is recomputed causally
from L1 mids, exactly as apply_trim_filter.py.

Usage:
  python robustness_checks.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from significance import hac_stats

# Displacement is precomputed (causally, from L1 mids) by apply_trim_filter.py;
# the decomposition file carries the per-trade bps split for the fill-floor check.
# Both are the same 969 full-train 5-min tail trades, keyed by decision_time.
TRIM = Path("data/results/trim_filter_trades.csv")
DECOMP = Path("data/results/reversion_long_horizons_decomp_trades.csv")
DEV_END = datetime(2024, 2, 25, tzinfo=timezone.utc)

# Conservative taker spread to replace the optimistic sweep's favorable fills.
CONSERVATIVE_SPREAD_BPS_PER_SIDE = 2.0
ROUND_TRIP_FEE_BPS = 10.0


def load_5min_with_displacement() -> pl.DataFrame:
    """The 969 full-train 5-min tail trades: realized net (optimistic), the
    decomposition bps split, and the precomputed causal displacement, joined on
    decision_time and time-ordered."""
    trim = pl.read_csv(TRIM).with_columns(
        pl.col("decision_time").str.to_datetime(time_zone="UTC")
    ).select("decision_time", "displacement_bps", "net_bps")
    decomp = (
        pl.read_csv(DECOMP)
        .filter(pl.col("holding_period") == "5min")
        .with_columns(pl.col("decision_time").str.to_datetime(time_zone="UTC"))
        .select("decision_time", "direction", "gross_mid_to_mid",
                "latency_slippage", "net_realized")
    )
    return trim.join(decomp, on="decision_time", how="inner").sort("decision_time")


def causal_keep(df: pl.DataFrame, w_days: int, min_hist: int) -> np.ndarray:
    """Keep event i iff displacement_i > median of past tail-event displacements in
    [t_i - W, t_i) (>= min_hist prior, else keep -- the causal cold-start)."""
    times_ns = df["decision_time"].cast(pl.Int64).to_numpy()
    d = df["displacement_bps"].to_numpy()
    w_ns = int(timedelta(days=w_days).total_seconds() * 1e9)
    keep = np.zeros(len(d), dtype=bool)
    for i in range(len(d)):
        lo = int(np.searchsorted(times_ns, times_ns[i] - w_ns, side="left"))
        past = d[lo:i]
        keep[i] = True if len(past) < min_hist else (d[i] > float(np.median(past)))
    return keep


def _stat(y: np.ndarray) -> str:
    if len(y) == 0:
        return "  n=   0"
    s = hac_stats(y)
    return f"n={len(y):4d}  mean {y.mean():+7.2f}  t_NW {s['t_nw']:+5.2f} (L={int(s['nw_lag'])})"


def trim_robustness(df: pl.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("CHECK 1 -- trim-filter robustness (causal displacement trailing-median)")
    print("F7 baseline = W=30d, min_hist=15. net_realized (optimistic fills).")
    print("=" * 78)
    hold = df.filter(pl.col("decision_time") >= DEV_END)
    print(f"\nHOLDOUT all (no filter): {_stat(hold['net_realized'].to_numpy())}")
    print("\nHOLDOUT filtered, by (lookback W, min_history):")
    print(f"  {'W (days)':>9s} {'min_hist':>9s} {'keep%':>6s}   holdout-filtered net")
    for w in (15, 30, 45, 60):
        for mh in (10, 15, 25):
            keep = causal_keep(df, w, mh)
            d2 = df.with_columns(keep=pl.Series(keep)).filter(pl.col("decision_time") >= DEV_END)
            kept = d2.filter(pl.col("keep"))
            mark = "  <- F7" if (w == 30 and mh == 15) else ""
            print(f"  {w:9d} {mh:9d} {d2['keep'].mean()*100:5.0f}%   "
                  f"{_stat(kept['net_realized'].to_numpy())}{mark}")


def pessimistic_floor(df: pl.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("CHECK 2 -- pessimistic fill floor (replace favorable sweep with flat "
          f"{CONSERVATIVE_SPREAD_BPS_PER_SIDE:.0f} bps/side taker)")
    print("=" * 78)
    keep = causal_keep(df, 30, 15)  # F7 filter
    floor = (
        pl.col("gross_mid_to_mid")
        - pl.col("latency_slippage")
        - 2 * CONSERVATIVE_SPREAD_BPS_PER_SIDE
        - ROUND_TRIP_FEE_BPS
    )
    df = df.with_columns(keep=pl.Series(keep), net_floor=floor)
    cells = [
        ("DEV", df.filter(pl.col("decision_time") < DEV_END)),
        ("HOLDOUT", df.filter(pl.col("decision_time") >= DEV_END)),
    ]
    print(f"\n  {'cell':18s} {'optimistic (net_realized)':>30s}   {'floor (2bps/side taker)':>30s}")
    for name, sub in cells:
        for tag, s in ((f"{name} all", sub), (f"{name} filtered", sub.filter(pl.col("keep")))):
            opt = _stat(s["net_realized"].to_numpy())
            flr = _stat(s["net_floor"].to_numpy())
            print(f"  {tag:18s} {opt:>30s}   {flr:>30s}")


def capacity_on_violent(df: pl.DataFrame) -> None:
    """F1 found book-walk negligible ($50k ~ $100k) on the *full* population, but
    the trimmed book trades the most violent cascades -- when the L1 book is
    thinnest. Re-check the entry+exit book-walk (the `spread_*` decomposition terms,
    bps; negative = filled inside mid) at $50k vs $100k restricted to the kept
    (violent) events. Entry is at decision+latency, identical across holding periods,
    so the 2-min decomposition's spread terms carry to the 5-min tail; this is the
    one horizon with both sizes simulated on the same 969 events."""
    print("\n" + "=" * 78)
    print("CHECK 3 -- capacity on violent (kept) events: $50k vs $100k book-walk")
    print("spread_entry + spread_exit (bps, negative = price improvement vs mid), 2min decomp")
    print("=" * 78)
    keep_map = df.select("decision_time",
                         pl.Series("keep", causal_keep(df, 30, 15)))
    dec = (
        pl.read_csv("data/results/reversion_decomposition_trades.csv")
        .filter(pl.col("holding_period") == "2min")
        .with_columns(pl.col("decision_time").str.to_datetime(time_zone="UTC"))
        .select("decision_time", "trade_notional_usd", "spread_entry", "spread_exit")
        .join(keep_map, on="decision_time", how="inner")
    )
    print(f"\n  {'subset':16s} {'size':>8s} {'spread_entry':>13s} {'spread_exit':>12s} {'total book-walk':>16s}")
    for tag, sub in (("all events", dec), ("kept (violent)", dec.filter(pl.col("keep")))):
        for size in (50000.0, 100000.0):
            s = sub.filter(pl.col("trade_notional_usd") == size)
            se, sx = s["spread_entry"].mean(), s["spread_exit"].mean()
            print(f"  {tag:16s} {size/1000:6.0f}k {se:13.3f} {sx:12.3f} {se + sx:16.3f}")
    # The capacity-relevant number: how much does doubling size move the book-walk on
    # the violent subset?
    v = dec.filter(pl.col("keep"))
    tot = lambda sz: float(v.filter(pl.col("trade_notional_usd") == sz)
                           .select((pl.col("spread_entry") + pl.col("spread_exit")).mean()).item())
    print(f"\n  violent-subset book-walk increment $50k -> $100k: "
          f"{tot(100000.0) - tot(50000.0):+.3f} bps (impact of doubling size)")


def main() -> None:
    df = load_5min_with_displacement()
    print(f"loaded {len(df)} 5-min tail trades with valid displacement (< holdout end)")
    trim_robustness(df)
    pessimistic_floor(df)
    capacity_on_violent(df)
    print("\nTEST PERIOD (2024-06-25 -> 2024-10-14) untouched.")


if __name__ == "__main__":
    main()
