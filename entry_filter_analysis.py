#!/usr/bin/env python3
"""
Entry-filter (trimming) analysis for the >=98th-tail reversion trades.

Finding 5 ruled out path-based early exits (TP/stop hurt). The remaining lever is
TRIMMING: drop trades that are predictably bad using only information available AT
decision time. This computes a set of causal decision-time features per trade and
tests which predict the realized net return, via rank correlation and quintile
conditional means -- so reliably-negative buckets become trim candidates.

Features (all causal, read at/before decision via BookProvider):
  direction        long (+1, after SELL cascade) vs short (-1)
  spread_bps       L1 spread / mid at decision
  imbalance        (bid_qty - ask_qty)/(bid_qty + ask_qty) at decision
  micro_lean_bps   direction * (micro_price - mid)/mid at decision (book lean)
  displacement_bps direction * (mid_pre - mid_decision)/mid_decision  (room to revert)
  predrift_bps     direction * mid drift over [liq-65m, liq-5m]  (the adverse trend)
  size_ratio       aggregate_quantity / trailing_threshold  (how extreme within the tail)
  vol_30m_bps      std of 1-min log-mid returns over the 30 min before decision
  hour             decision UTC hour

Target: realized net bps per trade (after the 10 bps round-trip taker fee).

8-month dev window only (2023-06-25 -> 2024-02-24). Book reads only ever look back
from decision, so nothing past the dev boundary is touched.

Usage:
  python entry_filter_analysis.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from backtester import BookProvider

TRADES = Path("data/results/reversion_long_horizons_trades.csv")
RESULTS_DIR = Path("data/results")
DEV_END = datetime(2024, 2, 25, tzinfo=timezone.utc)  # exclusive
PRE_LOOKBACK = timedelta(seconds=5)
DRIFT_WINDOW = timedelta(minutes=60)
VOL_WINDOW = timedelta(minutes=30)
FEATURES = ["spread_bps", "imbalance", "micro_lean_bps", "displacement_bps",
            "predrift_bps", "size_ratio", "vol_30m_bps"]


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r)) if n > 2 else float("nan")
    return r, t


def _quintiles(x: np.ndarray, y: np.ndarray) -> list[tuple[float, float, int]]:
    order = np.argsort(x)
    return [(float(x[g].mean()), float(y[g].mean()), len(g)) for g in np.array_split(order, 5)]


def main() -> None:
    trades = (
        pl.read_csv(TRADES)
        .with_columns(
            pl.col("decision_time").str.to_datetime(time_zone="UTC"),
            pl.col("liquidation_time").str.to_datetime(time_zone="UTC"),
        )
        .filter((pl.col("holding_period") == "5min") & (pl.col("decision_time") < DEV_END))
        .sort("decision_time")
    )
    book = BookProvider()
    rows: list[dict] = []
    for t in trades.iter_rows(named=True):
        dec, liq, d = t["decision_time"], t["liquidation_time"], float(t["direction"])
        m = book.as_of(dec)
        pre = book.as_of(liq - PRE_LOOKBACK)
        p0 = book.as_of(liq - PRE_LOOKBACK - DRIFT_WINDOW)
        if m is None or pre is None or p0 is None or m.micro_price is None:
            continue
        # 1-min grid realized vol over the 30 min before decision
        grid = [book.as_of(dec - timedelta(seconds=s)) for s in range(int(VOL_WINDOW.total_seconds()), -1, -60)]
        mids = np.array([g.mid_price for g in grid if g is not None])
        vol_bps = float(np.std(np.diff(np.log(mids))) * 1e4) if len(mids) > 2 else np.nan

        net_bps = 1e4 * t["net_pnl"] / (abs(t["entry_quantity"]) * 100.0)
        rows.append(
            {
                "direction": int(d),
                "net_bps": net_bps,
                "spread_bps": 1e4 * (m.ask_price - m.bid_price) / m.mid_price,
                "imbalance": m.book_imbalance,
                "micro_lean_bps": d * 1e4 * (m.micro_price - m.mid_price) / m.mid_price,
                "displacement_bps": d * 1e4 * (pre.mid_price - m.mid_price) / m.mid_price,
                "predrift_bps": d * 1e4 * (pre.mid_price / p0.mid_price - 1.0),
                "size_ratio": float(t["aggregate_quantity"]) / float(t["trailing_threshold"]),
                "vol_30m_bps": vol_bps,
                "hour": dec.hour,
            }
        )

    df = pl.DataFrame(rows).drop_nulls()
    y = df["net_bps"].to_numpy()
    n = len(df)
    print(f"\n=== Entry-filter analysis: 5min tail reversion, 8-month dev (n={n}) ===")
    print(f"baseline mean net {y.mean():+.2f} bps\n")

    longs = df.filter(pl.col("direction") == 1)["net_bps"].to_numpy()
    shorts = df.filter(pl.col("direction") == -1)["net_bps"].to_numpy()
    print(f"direction:  LONG  n={len(longs):3d} net {longs.mean():+.2f}    "
          f"SHORT n={len(shorts):3d} net {shorts.mean():+.2f}\n")

    print("feature -> net_bps (Spearman r, t; quintile means low->high):")
    for f in FEATURES:
        x = df[f].to_numpy()
        r, t = _spearman(x, y)
        q = _quintiles(x, y)
        qs = "  ".join(f"[{fv:+.2f}] {yv:+6.2f}(n{nn})" for fv, yv, nn in q)
        print(f"  {f:>16}: r={r:+.3f} t={t:+5.2f}   {qs}")

    print("\nby UTC session:")
    for lo, hi, name in ((0, 8, "Asia 00-08"), (8, 16, "EU 08-16"), (16, 24, "US 16-24")):
        g = df.filter((pl.col("hour") >= lo) & (pl.col("hour") < hi))["net_bps"].to_numpy()
        if len(g):
            print(f"  {name}: n={len(g):3d} net {g.mean():+.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "entry_filter_8mo_features.csv"
    df.write_csv(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
