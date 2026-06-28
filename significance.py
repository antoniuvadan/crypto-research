#!/usr/bin/env python3
"""
Significance of the per-trade net edge, with Newey-West (HAC) standard errors.

The naive t-stat (mean / [std/sqrt(n)]) assumes trades are IID. They are not:
liquidation events cluster, and holding periods longer than the gap between
events produce overlapping positions — both induce positive serial correlation
in the per-trade return series, which deflates the naive SE and inflates the
t-stat. Newey-West corrects for it.

For each (trade_notional, holding_period) group, regress the time-ordered
per-trade net return (bps) on a constant and report the HAC SE of the mean.
Bandwidth uses the standard automatic rule L = floor(4 (n/100)^(2/9)).

Usage:
  python significance.py                                                   # reversion
  python significance.py --trades data/results/liquidation_momentum_model_c_trades.csv --label momentum
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

DEFAULT_TRADES_PATH = Path("data/results/reversion_model_c_trades.csv")
RESULTS_DIR = Path("data/results")
CONTRACT_NOTIONAL_USD = 100.0
HOLDING_ORDER = ["5s", "10s", "30s", "1min", "2min"]


def nw_lag(n: int) -> int:
    """Automatic Bartlett-kernel bandwidth (Newey-West 1994)."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_stats(returns_bps: np.ndarray) -> dict[str, float]:
    """Mean per-trade return with IID and Newey-West HAC standard errors."""
    n = len(returns_bps)
    mean = float(returns_bps.mean())
    se_iid = float(returns_bps.std(ddof=1) / np.sqrt(n))

    lag = nw_lag(n)
    model = sm.OLS(returns_bps, np.ones(n)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True}
    )
    se_nw = float(model.bse[0])
    return {
        "n": float(n),
        "mean_bps": mean,
        "se_iid": se_iid,
        "t_iid": mean / se_iid if se_iid else float("nan"),
        "nw_lag": float(lag),
        "se_nw": se_nw,
        "t_nw": float(model.tvalues[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--label", default="reversion")
    args = parser.parse_args()

    trades = pl.read_csv(args.trades).with_columns(
        pl.col("decision_time").str.to_datetime(time_zone="UTC")
    )
    # Per-trade net return in bps of traded notional.
    trades = trades.with_columns(
        net_bps=1e4
        * pl.col("net_pnl")
        / (pl.col("entry_quantity").abs() * CONTRACT_NOTIONAL_USD)
    )

    rows: list[dict[str, float | str]] = []
    for (size, holding), grp in trades.group_by(
        ["trade_notional_usd", "holding_period"], maintain_order=True
    ):
        grp = grp.sort("decision_time")
        stats = hac_stats(grp["net_bps"].to_numpy())
        rows.append({"trade_notional_usd": size, "holding_period": holding, **stats})

    table = (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("holding_period")
            .replace_strict({h: i for i, h in enumerate(HOLDING_ORDER)}, default=99)
            .alias("_ord")
        )
        .sort("trade_notional_usd", "_ord")
        .drop("_ord")
    )

    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_float_precision(3)
    pl.Config.set_tbl_width_chars(200)

    print(f"\n=== Per-trade net edge significance: {args.label} (bps) ===")
    print("IID assumes independent trades; NW = Newey-West HAC (clustering + overlap).\n")
    print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.label}_significance.csv"
    table.write_csv(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
