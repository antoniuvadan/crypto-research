#!/usr/bin/env python3
"""
Reversion with a STATE-DEPENDENT exit (Improvement 1) over the training window.

Same liquidation signal and reversion direction as backtest_reversion.py
(`signal_direction_sign=-1`), but the fixed holding-period grid is replaced by a
single `RetracementExit` policy that closes each position when the L1 mid path
says the reversion is done: a take-profit when the mid retraces a fraction of the
cascade displacement, a stop when the cascade keeps running, or a max-hold time
cap. The exit reads the book through the lazy `BookProvider` seam.

The event set is identical to the fixed-exit baseline (the +/-5s aggregate-
quantity column is precomputed once), so the dynamic-exit trades line up one-to-
one with `reversion_model_c_trades.csv` for a clean A/B on the same events.

Outputs (same schema as the other Model C CSVs):
  data/results/reversion_dynamic_exit_summary.csv
  data/results/reversion_dynamic_exit_trades.csv

Usage:
  python backtest_reversion_dynamic.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

import backtester as bt
import backtest_momentum as bm

START_DATE = date(2023, 6, 25)
END_DATE = date(2024, 6, 24)

# Exit policy: cap holds at 2 min (the horizon where the gross reversion edge was
# largest in Finding 2), take profit at half the cascade displacement, stop if the
# cascade extends a full displacement further, pre-cascade mid sampled 5s before
# the liquidation (the start of the +/-5s signal window).
MAX_HOLD = timedelta(minutes=2)
EXIT_POLICY = bt.RetracementExit(
    max_hold=MAX_HOLD,
    take_profit_frac=0.5,
    stop_frac=1.0,
    pre_lookback=timedelta(seconds=5),
)

SIZES = bt.DEFAULT_TRADE_NOTIONAL_USD_GRID  # (50_000, 100_000)
SUMMARY_OUT = Path("data/results/reversion_dynamic_exit_summary.csv")
TRADES_OUT = Path("data/results/reversion_dynamic_exit_trades.csv")


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
    # size runs build the same event set as the fixed-exit baseline.
    agg_qty = bm._same_direction_aggregate_quantities(
        liq_df=liq, trades_df=agg, seconds_before=5, seconds_after=5
    )
    liq = liq.with_columns(pl.Series("agg_qty_5s_before_5s_after", agg_qty))

    summaries: list[pl.DataFrame] = []
    trades: list[pl.DataFrame] = []
    for size in SIZES:
        result = bm.run_liquidation_momentum_model_c_backtests(
            liq_snap=liq,
            agg_trades=agg,
            holding_periods=(MAX_HOLD,),  # single cell; the cap matches the policy
            trade_notional_usd=size,
            signal_direction_sign=-1,  # reversion: trade against the flow
            exit_policy=EXIT_POLICY,
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
