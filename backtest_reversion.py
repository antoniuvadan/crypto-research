#!/usr/bin/env python3
"""
Backtest the REVERSION variant of the liquidation signal — trade *against* the
liquidating flow — over the training window, mirroring the momentum Model C grid.

Reuses `backtester.run_liquidation_momentum_model_c_backtests` with
`signal_direction_sign=-1`. The event set (percentile, trailing window, +/-5s
window) is identical to the momentum run, so the same signal events fire; only
the trade direction is flipped. To keep events identical across both trade sizes
and avoid recomputing the expensive +/-5s aggregate-quantity column per cell, the
data is loaded once and the column is precomputed a single time.

Outputs (same schema as the momentum CSVs):
  data/results/reversion_model_c_summary.csv
  data/results/reversion_model_c_trades.csv

Usage:
  python backtest_reversion.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

import backtester as bt

START_DATE = date(2023, 6, 25)
END_DATE = date(2024, 6, 24)
HOLDING_PERIODS = (
    timedelta(seconds=5),
    timedelta(seconds=10),
    timedelta(seconds=30),
    timedelta(minutes=1),
    timedelta(minutes=2),
)
SIZES = bt.DEFAULT_TRADE_NOTIONAL_USD_GRID  # (50_000, 100_000)
SUMMARY_OUT = Path("data/results/reversion_model_c_summary.csv")
TRADES_OUT = Path("data/results/reversion_model_c_trades.csv")


def main() -> None:
    liq = (
        bt.load_liq_snap(start_date=START_DATE, end_date=END_DATE)
        .collect()
        .sort("time_datetime")
    )
    agg = (
        bt.load_agg_trades(start_date=START_DATE, end_date=END_DATE)
        .collect()
        .sort("transact_time_datetime")
    )

    # Precompute the +/-5s same-direction aggregate-quantity column once so both
    # size runs build an identical event set without recomputing it.
    agg_qty = bt._same_direction_aggregate_quantities(
        liq_df=liq, trades_df=agg, seconds_before=5, seconds_after=5
    )
    liq = liq.with_columns(pl.Series("agg_qty_5s_before_5s_after", agg_qty))

    summaries: list[pl.DataFrame] = []
    trades: list[pl.DataFrame] = []
    for size in SIZES:
        result = bt.run_liquidation_momentum_model_c_backtests(
            liq_snap=liq,
            agg_trades=agg,
            holding_periods=HOLDING_PERIODS,
            trade_notional_usd=size,
            signal_direction_sign=-1,  # reversion: trade against the flow
            summary_csv_path=None,
            trades_csv_path=None,
            show_progress=True,
        )
        summaries.append(result["summary"])
        trades.append(result["trades"])

    summary = pl.concat(summaries).sort(["trade_notional_usd", "holding_period"])
    trade_rows = pl.concat(trades).sort(
        ["trade_notional_usd", "holding_period", "decision_time"]
    )

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(SUMMARY_OUT)
    trade_rows.write_csv(TRADES_OUT)

    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(200)
    print(summary)
    print(f"\nWrote {SUMMARY_OUT} and {TRADES_OUT}")


if __name__ == "__main__":
    main()
