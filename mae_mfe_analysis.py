#!/usr/bin/env python3
"""
MAE/MFE path analysis for the >=98th-tail reversion trades (exit-timing study).

For each tail event, replay the L1 mid path forward from decision and measure the
running signed return r(t) = direction * (mid_t / mid_decision - 1) in bps. From
that path: Maximum Favorable Excursion (best the trade ever reached) and Maximum
Adverse Excursion (worst drawdown) and when the peak occurs, plus take-profit /
stop exit simulations -- to answer whether closing early beats holding to the
fixed 5-min cap, and at what level.

Mid-to-mid (pre-fee, pre-execution), consistent with gross_mid_to_mid; the
actionable net subtracts the ~10 bps round-trip taker fee. Latency is ignored in
the path (300ms is immaterial over minutes).

8-month dev window only (2023-06-25 -> 2024-02-24); book reads are capped at the
dev boundary so the holdout (2024-02-25+) is never touched.

Usage:
  python mae_mfe_analysis.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from backtester import BookProvider, _datetime_to_ns

TRADES = Path("data/results/reversion_long_horizons_trades.csv")
RESULTS_DIR = Path("data/results")
DEV_END = datetime(2024, 2, 25, tzinfo=timezone.utc)  # exclusive; holdout starts here
MAX_WINDOW = timedelta(minutes=30)
CAP_5MIN_S = 300.0
FEE_BPS = 10.0
CHECKPOINTS_S = [0, 5, 15, 30, 60, 120, 180, 300, 600, 900, 1200, 1800]
TP_GRID = [10, 15, 20, 30, 40, 50, 75, 100]
STOP_GRID = [-10, -20, -30, -50]


def _fmt_t(s: float) -> str:
    return f"{int(s)}s" if s < 60 else f"{s/60:g}m"


def main() -> None:
    trades = (
        pl.read_csv(TRADES)
        .with_columns(pl.col("decision_time").str.to_datetime(time_zone="UTC"))
        .filter((pl.col("holding_period") == "5min") & (pl.col("decision_time") < DEV_END))
        .sort("decision_time")
    )
    book = BookProvider()

    dirs: list[int] = []
    mfe5: list[float] = []
    mae5: list[float] = []
    tmfe5: list[float] = []
    fin5: list[float] = []
    mfe30: list[float] = []
    fin30: list[float] = []
    path_rows: list[list[float]] = []
    tp_exit = {tp: [] for tp in TP_GRID}
    tp_hit = {tp: 0 for tp in TP_GRID}
    tp_hold = {tp: [] for tp in TP_GRID}
    stop_exit = {s: [] for s in STOP_GRID}

    for t in trades.iter_rows(named=True):
        dec = t["decision_time"]
        d = float(t["direction"])
        m0 = book.as_of(dec)
        if m0 is None:
            continue
        w = book.window(dec, min(dec + MAX_WINDOW, DEV_END))
        if len(w) == 0:
            continue
        rel = (w.times_ns - np.int64(_datetime_to_ns(dec))) / 1e9
        r = d * (w.mid_price / m0.mid_price - 1.0) * 1e4

        m5 = rel <= CAP_5MIN_S
        if not m5.any():
            continue
        r5, rel5 = r[m5], rel[m5]

        dirs.append(int(d))
        mfe5.append(float(r5.max()))
        mae5.append(float(r5.min()))
        tmfe5.append(float(rel5[int(np.argmax(r5))]))
        fin5.append(float(r5[-1]))
        mfe30.append(float(r.max()))
        fin30.append(float(r[-1]))
        path_rows.append([float(r[max(np.searchsorted(rel, c, side="right") - 1, 0)]) for c in CHECKPOINTS_S])

        for tp in TP_GRID:
            hit = np.where(r5 >= tp)[0]
            if len(hit):
                tp_exit[tp].append(float(r5[hit[0]]))
                tp_hold[tp].append(float(rel5[hit[0]]))
                tp_hit[tp] += 1
            else:
                tp_exit[tp].append(float(r5[-1]))
                tp_hold[tp].append(CAP_5MIN_S)
        for s in STOP_GRID:
            hit = np.where(r5 <= s)[0]
            stop_exit[s].append(float(r5[hit[0]]) if len(hit) else float(r5[-1]))

    n = len(dirs)
    fin5a, mfe5a, mae5a, tmfe5a = map(np.array, (fin5, mfe5, mae5, tmfe5))
    mfe30a, fin30a, dira = np.array(mfe30), np.array(fin30), np.array(dirs)
    base_g = fin5a.mean()

    print(f"\n=== MAE/MFE path study: 5min tail reversion, 8-month dev (n={n}) ===")
    print(f"baseline hold-to-5min: gross {base_g:+.2f}  net {base_g - FEE_BPS:+.2f} bps "
          f"(long {fin5a[dira == 1].mean():+.2f} / short {fin5a[dira == -1].mean():+.2f})\n")

    print("avg signed-return path r(t), mean / median bps:")
    P = np.array(path_rows)
    for j, c in enumerate(CHECKPOINTS_S):
        print(f"  {_fmt_t(c):>4}: mean {P[:, j].mean():+7.2f}   median {np.median(P[:, j]):+7.2f}")

    print(f"\nwithin 5min:  MFE mean {mfe5a.mean():+.2f} (med {np.median(mfe5a):+.2f})   "
          f"MAE mean {mae5a.mean():+.2f} (med {np.median(mae5a):+.2f})")
    print(f"  give-back (MFE - final): mean {(mfe5a - fin5a).mean():+.2f} bps   "
          f"median time-to-MFE-peak {np.median(tmfe5a):.0f}s")
    print(f"  reach +X within 5min:  " + "  ".join(f"+{x}:{(mfe5a >= x).mean():.0%}" for x in (10, 20, 30, 50)))
    print(f"  within 30min: MFE mean {mfe30a.mean():+.2f}   final-30min mean {fin30a.mean():+.2f}")

    print("\ntake-profit sim (cap 5min): exit first time r>=TP, else 5min")
    print(f"  {'TP':>5} {'%hit':>6} {'avg_hold':>9} {'gross':>8} {'net':>8}   vs base net {base_g - FEE_BPS:+.2f}")
    for tp in TP_GRID:
        g = float(np.mean(tp_exit[tp]))
        print(f"  {tp:>5} {tp_hit[tp]/n:>6.0%} {np.mean(tp_hold[tp]):>8.0f}s {g:>8.2f} {g - FEE_BPS:>8.2f}")

    print("\nstop sim (cap 5min): exit first time r<=stop, else 5min")
    for s in STOP_GRID:
        g = float(np.mean(stop_exit[s]))
        print(f"  stop {s:>4}: gross {g:+7.2f}  net {g - FEE_BPS:+7.2f}")

    win, los = fin5a > 0, fin5a <= 0
    print(f"\nwinners vs losers (by 5min sign):  win {win.mean():.0%} MAE mean {mae5a[win].mean():+.2f}   "
          f"los {los.mean():.0%} MAE mean {mae5a[los].mean():+.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "mae_mfe_8mo_trades.csv"
    pl.DataFrame(
        {"direction": dirs, "mfe_5min": mfe5, "mae_5min": mae5, "t_mfe_5min_s": tmfe5,
         "final_5min": fin5, "mfe_30min": mfe30, "final_30min": fin30}
    ).write_csv(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
