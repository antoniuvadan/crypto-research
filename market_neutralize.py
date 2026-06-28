#!/usr/bin/env python3
"""
Market-neutralize the reversion trades: separate cascade-reversion alpha from the
prevailing market drift (the regime-concentration worry from Finding 4).

A single-BTC-asset directional trade cannot be hedged against BTC over the same
window (that is identically the trade's own return). So this uses an event-study
*abnormal return*: subtract the trend the price was already on, estimated causally
from a pre-cascade window. For each trade, all from L1 mids:

    signal_ret  = direction * (mid[decision + hold] / mid[decision] - 1)
    normal_ret  = direction * drift_per_sec * hold_seconds
                  where drift_per_sec is the mid drift over the clean pre-event
                  window [decision - (pre+lag)min, decision - lag_min], per second
    abnormal    = signal_ret - normal_ret          (the market-neutral edge, bps)

signal_ret is the mid-to-mid signal (single-feed, the robust quantity from F2);
fees/execution are separate constant offsets, so neutralizing the signal answers
"is the move reversion alpha or just trend?" cleanly. A net-tradeable view is
abnormal minus the ~10 bps round-trip taker fee.

Usage:
  python market_neutralize.py            # 5min, long-horizons tail trades
  python market_neutralize.py --holding 30min --pre-window-min 60
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from backtester import BookProvider
from significance import hac_stats

DEFAULT_TRADES = Path("data/results/reversion_long_horizons_trades.csv")
RESULTS_DIR = Path("data/results")
ROUND_TRIP_FEE_BPS = 10.0

HOLDING_MAP = {
    "5s": timedelta(seconds=5),
    "10s": timedelta(seconds=10),
    "30s": timedelta(seconds=30),
    "1min": timedelta(minutes=1),
    "2min": timedelta(minutes=2),
    "5min": timedelta(minutes=5),
    "30min": timedelta(minutes=30),
    "60min": timedelta(minutes=60),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--holding", default="5min", help="holding_period label to neutralize")
    parser.add_argument("--pre-window-min", type=int, default=60, help="drift estimation window (min)")
    parser.add_argument("--pre-lag-min", type=int, default=5, help="gap before decision to avoid the cascade (min)")
    parser.add_argument("--label", default="reversion_long_horizons")
    args = parser.parse_args()

    hold = HOLDING_MAP[args.holding]
    pre_lag = timedelta(minutes=args.pre_lag_min)
    pre_window = timedelta(minutes=args.pre_window_min)
    drift_seconds = pre_window.total_seconds()
    hold_seconds = hold.total_seconds()

    trades = (
        pl.read_csv(args.trades)
        .with_columns(pl.col("decision_time").str.to_datetime(time_zone="UTC"))
        .filter(pl.col("holding_period") == args.holding)
        .sort("decision_time")
    )
    if trades.is_empty():
        raise SystemExit(f"no trades with holding_period == {args.holding!r} in {args.trades}")

    book = BookProvider()
    rows: list[dict] = []
    for t in trades.iter_rows(named=True):
        dec = t["decision_time"]
        direction = float(t["direction"])
        m_dec = book.as_of(dec)
        m_exit = book.as_of(dec + hold)
        m_pre0 = book.as_of(dec - pre_lag - pre_window)
        m_pre1 = book.as_of(dec - pre_lag)
        if None in (m_dec, m_exit, m_pre0, m_pre1):
            continue

        signal_ret = m_exit.mid_price / m_dec.mid_price - 1.0
        drift_per_sec = (m_pre1.mid_price / m_pre0.mid_price - 1.0) / drift_seconds
        normal_ret = drift_per_sec * hold_seconds

        raw_bps = 1e4 * direction * signal_ret
        normal_bps = 1e4 * direction * normal_ret
        rows.append(
            {
                "decision_time": dec,
                "direction": int(direction),
                "month": dec.strftime("%Y-%m"),
                "raw_bps": raw_bps,
                "normal_bps": normal_bps,
                "abnormal_bps": raw_bps - normal_bps,
            }
        )

    df = pl.DataFrame(rows)
    n_drop = len(trades) - len(df)
    abn = df["abnormal_bps"].to_numpy()
    raw = df["raw_bps"].to_numpy()
    nrm = df["normal_bps"].to_numpy()
    stats = hac_stats(abn)

    print(f"\n=== Market-neutral (abnormal) reversion edge: {args.holding} tail ===")
    print(f"drift window {args.pre_window_min}min ending {args.pre_lag_min}min pre-decision; "
          f"n={len(df)} ({n_drop} dropped for missing mids)\n")
    print(f"  raw mid signal   : {raw.mean():+7.2f} bps")
    print(f"  normal (trend)   : {nrm.mean():+7.2f} bps   <- the drift component removed")
    print(f"  ABNORMAL (alpha) : {abn.mean():+7.2f} bps   t_IID={stats['t_iid']:+.2f}  "
          f"t_NW(L={int(stats['nw_lag'])})={stats['t_nw']:+.2f}")
    print(f"  abnormal - fee   : {abn.mean() - ROUND_TRIP_FEE_BPS:+7.2f} bps   (net-tradeable view)\n")

    for d, name in ((1, "LONG "), (-1, "SHORT")):
        g = df.filter(pl.col("direction") == d)
        if len(g):
            print(f"  {name} n={len(g):4d}: raw {g['raw_bps'].mean():+7.2f}  "
                  f"normal {g['normal_bps'].mean():+7.2f}  abnormal {g['abnormal_bps'].mean():+7.2f}")

    print("\n  by month (abnormal bps):")
    by_month = (
        df.group_by("month")
        .agg(pl.len().alias("n"), pl.col("raw_bps").mean(), pl.col("abnormal_bps").mean())
        .sort("month")
    )
    for r in by_month.iter_rows(named=True):
        print(f"    {r['month']}  n={r['n']:4d}  raw {r['raw_bps']:+7.2f}  abnormal {r['abnormal_bps']:+7.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.label}_market_neutral.csv"
    df.write_csv(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
