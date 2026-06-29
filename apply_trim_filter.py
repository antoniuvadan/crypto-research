#!/usr/bin/env python3
"""
Bake a causal trailing-median displacement filter into the tail-reversion signal
and measure net + Newey-West on the 8-month dev window, then validate OOS on the
within-train holdout (2024-02-25 -> 2024-06-24). The test period is NOT touched.

Finding 6: the edge is concentrated in violent/overshot cascades; the calm
small-displacement ones barely revert. Filter: trade a >=98th cascade only when its
decision-time displacement exceeds a *trailing median* of past tail-event
displacements (causal, adaptive; parameters fixed from dev). Trimming only drops
events, so kept-event fills are unchanged -- masking the existing 5min trades is
exactly equivalent to baking the filter into the signal.

displacement = direction * (mid_pre - mid_decision) / mid_decision  (room to revert)
  mid_pre = mid 5s before the liquidation; mid_decision = mid at decision.
threshold_i = median displacement of tail events in [t_i - W, t_i) (>= MIN_HISTORY,
  else keep -- causal cold-start). Keep event i iff displacement_i > threshold_i.

Usage:
  python apply_trim_filter.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from backtester import BookProvider
from significance import hac_stats

TRADES = Path("data/results/reversion_long_horizons_trades.csv")
RESULTS_DIR = Path("data/results")
DEV_END = datetime(2024, 2, 25, tzinfo=timezone.utc)        # dev: [start, DEV_END)
HOLDOUT_END = datetime(2024, 6, 25, tzinfo=timezone.utc)    # holdout: [DEV_END, HOLDOUT_END) = rest of train
PRE_LOOKBACK = timedelta(seconds=5)
TRAIL_WINDOW = timedelta(days=30)
MIN_HISTORY = 15


def _cell(df: pl.DataFrame, label: str) -> None:
    y = df["net_bps"].to_numpy()  # already time-ordered
    s = hac_stats(y)
    print(f"  {label:26s} n={len(y):4d}  mean {y.mean():+7.2f}  t_IID {s['t_iid']:+5.2f}  "
          f"t_NW {s['t_nw']:+5.2f}  (L={int(s['nw_lag'])})")


def main() -> None:
    trades = (
        pl.read_csv(TRADES)
        .with_columns(
            pl.col("decision_time").str.to_datetime(time_zone="UTC"),
            pl.col("liquidation_time").str.to_datetime(time_zone="UTC"),
        )
        .filter((pl.col("holding_period") == "5min") & (pl.col("decision_time") < HOLDOUT_END))
        .sort("decision_time")
    )

    book = BookProvider()
    disp = np.full(len(trades), np.nan)
    for i, t in enumerate(trades.iter_rows(named=True)):
        m = book.as_of(t["decision_time"])
        pre = book.as_of(t["liquidation_time"] - PRE_LOOKBACK)
        if m is not None and pre is not None:
            disp[i] = float(t["direction"]) * 1e4 * (pre.mid_price - m.mid_price) / m.mid_price

    trades = trades.with_columns(
        displacement_bps=pl.Series(disp),
        net_bps=1e4 * pl.col("net_pnl") / (pl.col("entry_quantity").abs() * 100.0),
    ).drop_nulls(subset=["displacement_bps"])

    # Causal trailing-median threshold over PAST tail events in [t-W, t).
    times_ns = trades["decision_time"].cast(pl.Int64).to_numpy()
    d = trades["displacement_bps"].to_numpy()
    w_ns = int(TRAIL_WINDOW.total_seconds() * 1e9)
    keep = np.zeros(len(d), dtype=bool)
    for i in range(len(d)):
        lo = int(np.searchsorted(times_ns, times_ns[i] - w_ns, side="left"))
        past = d[lo:i]
        keep[i] = True if len(past) < MIN_HISTORY else (d[i] > float(np.median(past)))
    trades = trades.with_columns(keep=pl.Series(keep))

    dev = trades.filter(pl.col("decision_time") < DEV_END)
    hold = trades.filter(pl.col("decision_time") >= DEV_END)

    print(f"\n=== Causal trailing-median displacement filter (W={TRAIL_WINDOW.days}d, "
          f"min_hist={MIN_HISTORY}) ===")
    print(f"filter is fixed from dev (Finding 6); holdout = within-train OOS, test period untouched\n")

    print("DEV (2023-06-25 -> 2024-02-24):")
    _cell(dev, "all")
    _cell(dev.filter(pl.col("keep")), f"filtered ({dev['keep'].mean():.0%} kept)")
    print("\nHOLDOUT / within-train OOS (2024-02-25 -> 2024-06-24):")
    _cell(hold, "all")
    _cell(hold.filter(pl.col("keep")), f"filtered ({hold['keep'].mean():.0%} kept)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "trim_filter_trades.csv"
    trades.select("decision_time", "direction", "displacement_bps", "net_bps", "keep").write_csv(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
