#!/usr/bin/env python3
"""
Longer holding horizons on the >=98th-pct tail (reversion).

Finding 3 showed reversion strength grows with cascade size (the >=98th tail is
best) and that the gross edge keeps rising with horizon in every band (tail:
+5.2 -> +17.9 bps over 5s -> 2min). That monotone-in-horizon gross suggests the
2-min cap was truncating the retracement. This extends the holding grid to
[1m, 5m, 30m, 60m] on the tail to see whether gross keeps growing and whether net
clears the ~10 bps round-trip fee with room to spare.

Same signal/event set as Findings 1-3 (>=98th pct, trailing 7d), reversion
direction, fixed-horizon exit, $50k. Writes a trades CSV so the per-trade edge can
be run through Newey-West (`significance.py`) -- essential here, since 30-60 min
holds overlap massively and inflate the HAC standard errors.

Outputs:
  data/results/reversion_long_horizons_summary.csv
  data/results/reversion_long_horizons_trades.csv

Usage:
  python backtest_reversion_long_horizons.py
  python significance.py --trades data/results/reversion_long_horizons_trades.csv \\
                         --label reversion_long_horizons
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

import backtester as bt
import backtest_momentum as bm

START_DATE = date(2023, 6, 25)
END_DATE = date(2024, 6, 24)
CONTRACT_NOTIONAL_USD = 100.0

HOLDING_PERIODS = (
    timedelta(minutes=1),
    timedelta(minutes=5),
    timedelta(minutes=30),
    timedelta(minutes=60),
)
TRADE_NOTIONAL_USD = 50_000.0

SUMMARY_OUT = Path("data/results/reversion_long_horizons_summary.csv")
TRADES_OUT = Path("data/results/reversion_long_horizons_trades.csv")


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
    agg_qty = bm._same_direction_aggregate_quantities(
        liq_df=liq, trades_df=agg, seconds_before=5, seconds_after=5
    )
    liq = liq.with_columns(pl.Series("agg_qty_5s_before_5s_after", agg_qty))

    result = bm.run_liquidation_momentum_model_c_backtests(
        liq_snap=liq,
        agg_trades=agg,
        holding_periods=HOLDING_PERIODS,
        trade_notional_usd=TRADE_NOTIONAL_USD,
        signal_direction_sign=-1,  # reversion
        percentile=0.98,  # the >=98th tail (Finding 3: where reversion is strongest)
        upper_percentile=None,
        summary_csv_path=None,
        trades_csv_path=None,
        show_progress=True,
    )
    summary, trades = result["summary"], result["trades"]

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(SUMMARY_OUT)
    trades.write_csv(TRADES_OUT)

    # Per-trade gross/net bps + a rough IID t (Newey-West via significance.py).
    rows: list[dict[str, float | str]] = []
    for hp in HOLDING_PERIODS:
        h = bt._format_timedelta(hp)
        tr = trades.filter(pl.col("holding_period") == h)
        n = len(tr)
        notional = tr["entry_quantity"].abs() * CONTRACT_NOTIONAL_USD
        gross = (1e4 * tr["gross_pnl"] / notional).to_numpy()
        net = (1e4 * tr["net_pnl"] / notional).to_numpy()
        std = net.std(ddof=1) if n > 1 else float("nan")
        rows.append(
            {
                "holding_period": h,
                "n": n,
                "gross_bps": round(float(gross.mean()), 2),
                "net_bps": round(float(net.mean()), 2),
                "t_iid": round(float(net.mean() / (std / np.sqrt(n))), 2) if std > 0 else float("nan"),
            }
        )

    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(200)
    print(pl.DataFrame(rows))
    print(f"\nWrote {SUMMARY_OUT} and {TRADES_OUT}")
    print("Now run: python significance.py --trades", TRADES_OUT, "--label reversion_long_horizons")


if __name__ == "__main__":
    main()
