#!/usr/bin/env python3
"""
Which slice of the liquidation-cascade size distribution is most profitable to
trade as a reversion? (Event-selection sweep — Improvement 2, the entry side.)

The original 98th-percentile threshold was designed for a *momentum* signal: it
isolates the open-ended right tail, where the largest cascades live. For
reversion that tail is the wrong place to look — the heuristic is that very large
cascades (>=98th pct) are regime-shifting, non-reverting moves, while the far left
tail produces dislocations too small to clear fees. The tradeable reversion should
sit in the inner region.

This sweeps three inner-region percentile *bands* of the trailing-7d cascade-size
distribution, trading reversion (`signal_direction_sign=-1`) across the holding
grid, and reports gross/net bps per trade so the most profitable slice is visible.
Exit is the fixed-horizon clock (no bookTicker loads) to keep the comparison about
event selection only; the +/-5s aggregate-quantity column is computed once and
reused across all bands.

Output:
  data/results/reversion_event_slices_summary.csv

Usage:
  python backtest_reversion_event_slices.py
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
    timedelta(seconds=5),
    timedelta(seconds=10),
    timedelta(seconds=30),
    timedelta(minutes=1),
    timedelta(minutes=2),
)

# Three inner-region slices of the cascade-size distribution, walking down from
# just-below the old 98th-pct tail toward the median.
BANDS: tuple[tuple[float, float], ...] = (
    (0.50, 0.80),
    (0.80, 0.95),
    (0.95, 0.98),
)

TRADE_NOTIONAL_USD = 50_000.0  # size is ~irrelevant at this depth (Finding 1)
SUMMARY_OUT = Path("data/results/reversion_event_slices_summary.csv")


def _net_bps(trades: pl.DataFrame) -> pl.Series:
    notional = trades["entry_quantity"].abs() * CONTRACT_NOTIONAL_USD
    return 1e4 * trades["net_pnl"] / notional


def _gross_bps(trades: pl.DataFrame) -> pl.Series:
    notional = trades["entry_quantity"].abs() * CONTRACT_NOTIONAL_USD
    return 1e4 * trades["gross_pnl"] / notional


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

    # Compute the +/-5s same-direction aggregate-quantity column once; every band
    # reuses it (only the percentile thresholds differ).
    agg_qty = bm._same_direction_aggregate_quantities(
        liq_df=liq, trades_df=agg, seconds_before=5, seconds_after=5
    )
    liq = liq.with_columns(pl.Series("agg_qty_5s_before_5s_after", agg_qty))

    rows: list[dict[str, float | str]] = []
    for lower, upper in BANDS:
        result = bm.run_liquidation_momentum_model_c_backtests(
            liq_snap=liq,
            agg_trades=agg,
            holding_periods=HOLDING_PERIODS,
            trade_notional_usd=TRADE_NOTIONAL_USD,
            signal_direction_sign=-1,  # reversion
            percentile=lower,
            upper_percentile=upper,
            summary_csv_path=None,
            trades_csv_path=None,
            show_progress=True,
        )
        trades = result["trades"]
        for hp in HOLDING_PERIODS:
            hp_label = bt._format_timedelta(hp)
            tr = trades.filter(pl.col("holding_period") == hp_label)
            n = len(tr)
            if n == 0:
                rows.append(
                    {"band": f"[{lower:.2f},{upper:.2f}]", "holding_period": hp_label, "n": 0}
                )
                continue
            net = _net_bps(tr).to_numpy()
            gross = _gross_bps(tr).to_numpy()
            mean_net = float(net.mean())
            std_net = float(net.std(ddof=1)) if n > 1 else float("nan")
            t_iid = mean_net / (std_net / np.sqrt(n)) if std_net > 0 else float("nan")
            rows.append(
                {
                    "band": f"[{lower:.2f},{upper:.2f}]",
                    "holding_period": hp_label,
                    "n": n,
                    "gross_bps": round(float(gross.mean()), 2),
                    "net_bps": round(mean_net, 2),
                    "t_iid": round(t_iid, 2),
                }
            )

    summary = pl.DataFrame(rows)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(SUMMARY_OUT)

    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_width_chars(200)
    print(summary)
    print(f"\nWrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
