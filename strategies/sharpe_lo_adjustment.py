#!/usr/bin/env python3
"""
Lo (2002) autocorrelation-adjusted annualised Sharpe for the frozen reversion strategy.

The repo's headline Sharpe uses the naive rule SR_ann = sqrt(365) * SR_daily, which assumes
IID daily returns. The strategy's trades overlap (up to 18 concurrent) and cluster, which can
serially-correlate the daily P&L and make sqrt(365) the wrong annualiser. This driver replaces
sqrt(365) with the Lo factor eta(q) (significance.lo_annualized_sharpe) and reports both, so the
annualised number is honest about autocorrelation.

Scope: the UNCAPPED / frozen strategy (the resume candidate) on the reserved OOS test window and,
for context, on the train dev + within-train-holdout windows. It reads only per-trade `net_pnl`
(no book pass), so it is fast; the method applies verbatim to any trades CSV with `decision_time`
+ `net_pnl` (see `series_specs`).

Usage (repo root):
  ~/miniforge3/envs/mscf/bin/python strategies/sharpe_lo_adjustment.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl

from significance import lo_annualized_sharpe
from concurrency_analysis import load, OOS_CSV, TRAIN_CSV

UTC = timezone.utc
Q = 365                       # crypto trades 24/7 -> 365 periods/yr, matching the repo's sqrt(365)
MAX_LAGS = (1, 5, 10, 20)     # truncation sensitivity for the autocorrelation sum
DEV_START = datetime(2023, 6, 25, tzinfo=UTC)
HOLDOUT_START = datetime(2024, 2, 25, tzinfo=UTC)
TRAIN_END = datetime(2024, 6, 25, tzinfo=UTC)
TEST_START = datetime(2024, 6, 25, tzinfo=UTC)
TEST_END = datetime(2024, 10, 15, tzinfo=UTC)


def _eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def daily_pnl(df: pl.DataFrame, start: datetime, end: datetime) -> np.ndarray:
    """Net P&L per CALENDAR day over [start, end), idle days padded with 0 -- exactly the repo's
    Sharpe construction: a trade is booked on its close date (decision_time + 5min)."""
    closes = df["decision_time"].dt.offset_by("5m").dt.date()
    daily = (df.with_columns(close_date=closes).group_by("close_date")
             .agg(pl.col("net_pnl").sum()).sort("close_date"))
    all_days = pl.date_range(start.date(), (end - timedelta(days=1)).date(), interval="1d", eager=True)
    cal = pl.DataFrame({"close_date": all_days}).join(daily, on="close_date", how="left").fill_null(0.0)
    return cal["net_pnl"].to_numpy()


def series_specs() -> list[tuple[str, np.ndarray]]:
    """(label, daily-pnl series) for each window of the frozen (uncapped) strategy."""
    specs: list[tuple[str, np.ndarray]] = []
    if OOS_CSV.exists():
        oos = load(OOS_CSV)
        specs.append(("OOS uncapped (frozen, F10)", daily_pnl(oos, TEST_START, TEST_END)))
    if TRAIN_CSV.exists():
        train = load(TRAIN_CSV)
        dev = train.filter(pl.col("decision_time") < HOLDOUT_START)
        hold = train.filter((pl.col("decision_time") >= HOLDOUT_START)
                            & (pl.col("decision_time") < TRAIN_END))
        specs.append(("TRAIN dev uncapped", daily_pnl(dev, DEV_START, HOLDOUT_START)))
        specs.append(("TRAIN holdout uncapped", daily_pnl(hold, HOLDOUT_START, TRAIN_END)))
    return specs


def main() -> None:
    rows: list[dict] = []
    for label, series in series_specs():
        for L in MAX_LAGS:
            r = lo_annualized_sharpe(series, q=Q, max_lag=L)
            rows.append({
                "strategy": label,
                "n_days": int(r["n_days"]),
                "active": int(r["active_days"]),
                "sr_daily": round(r["sr_period"], 4),
                "sr_ann_naive_sqrt365": round(r["sr_ann_iid"], 2),
                "max_lag": L,
                "rho1": round(r["rho1"], 3),
                "sum_rho_1..L": round(r["sum_rho"], 3),
                "eta": round(r["eta"], 2),
                "sqrt_q": round(r["sqrt_q"], 2),
                "sr_ann_LO": round(r["sr_ann_lo"], 2),
                "deflation_eta/sqrtq": round(r["deflation"], 3),
            })
    tbl = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(50); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220)
    print("\n=== Lo (2002) autocorrelation-adjusted annualised Sharpe ===")
    print("naive: SR_ann = sqrt(365)*SR_daily (assumes IID days).")
    print("Lo:    SR_ann = eta(q)*SR_daily,  eta = q/sqrt(q + 2*sum_{k=1}^{L}(q-k)*rho_k).")
    print("deflation = eta/sqrt(q) = SR_ann_LO / SR_ann_naive. rho1<0 => LO number is HIGHER.\n")
    print(tbl)

    out = _Path("data/results/sharpe_lo_adjustment.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.write_csv(out)
    _eprint(f"\nwrote {out}")


if __name__ == "__main__":
    main()
