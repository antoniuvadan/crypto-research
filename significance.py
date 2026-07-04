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
HOLDING_ORDER = ["5s", "10s", "30s", "1min", "2min", "5min", "30min", "60min"]


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


def lo_eta(rho: np.ndarray, q: int) -> float:
    """Lo (2002) autocorrelation-corrected Sharpe ANNUALISATION factor.

    The naive rule annualises a per-period Sharpe by multiplying by sqrt(q) (q = periods per
    year). That is only correct if the per-period returns are IID. When they are serially
    correlated the variance of the q-period cumulative return is NOT q*sigma^2 but

        Var(sum of q returns) = sigma^2 * ( q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k )

    so the correct annualisation factor is

        eta(q) = q / sqrt( q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k )      and   SR_annual = eta(q) * SR_period.

    `rho[k-1]` is the lag-k autocorrelation of the returns. With rho == 0 this reduces to
    eta = q/sqrt(q) = sqrt(q) (the naive rule). POSITIVE autocorrelation makes the denominator
    bigger, so eta < sqrt(q) and the annualised Sharpe is DEFLATED; negative autocorrelation
    inflates it. Reference: Andrew W. Lo, "The Statistics of Sharpe Ratios", FAJ 2002, eq. (7)-(9).

    `rho` may be truncated (only the first L lags supplied); lags beyond its length are treated
    as 0. Returns nan if the corrected variance term is non-positive (extreme negative
    autocorrelation), which signals the estimate is unusable at that truncation.
    """
    lags = np.arange(1, len(rho) + 1)
    corrected = q + 2.0 * float(np.sum((q - lags) * rho))
    return q / np.sqrt(corrected) if corrected > 0 else float("nan")


def autocorrelations(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelations rho_1..rho_max_lag (standard biased ACF: divide every lag by the
    lag-0 sum, so |rho_k| <= 1)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([float(np.sum(x[k:] * x[:-k]) / denom) for k in range(1, max_lag + 1)])


def lo_annualized_sharpe(daily: np.ndarray, q: int = 365, max_lag: int = 10) -> dict[str, float]:
    """Lo-adjusted annualised Sharpe of a per-period (here daily) P&L / return series.

    Steps: (1) per-period Sharpe SR = mean/std(ddof=1); (2) estimate rho_1..rho_max_lag; (3) the
    Lo factor eta(q) with the sum truncated at max_lag (you cannot estimate q-1=364 lags from a
    short sample, so truncate where rho is still estimable and report a couple of max_lag values);
    (4) SR_ann_lo = eta * SR vs the naive SR_ann_iid = sqrt(q) * SR. The ratio SR_ann_lo /
    SR_ann_iid = eta/sqrt(q) is the deflation factor. Scale (dollars vs return-on-capital) does
    not matter -- the Sharpe is scale-invariant; only the autocorrelation structure does.
    """
    daily = np.asarray(daily, float)
    n = len(daily)
    sd = float(daily.std(ddof=1)) if n > 1 else 0.0
    sr = float(daily.mean()) / sd if sd > 0 else float("nan")
    max_lag = int(min(max_lag, n - 2)) if n > 2 else 0
    rho = autocorrelations(daily, max_lag) if max_lag >= 1 else np.array([])
    eta = lo_eta(rho, q)
    return {
        "n_days": float(n),
        "active_days": float(int(np.count_nonzero(daily))),
        "sr_period": sr,
        "sr_ann_iid": np.sqrt(q) * sr,   # naive sqrt(q) annualisation
        "sr_ann_lo": eta * sr,           # Lo autocorrelation-adjusted annualisation
        "eta": eta,
        "sqrt_q": float(np.sqrt(q)),
        "deflation": eta / np.sqrt(q),   # SR_ann_lo / SR_ann_iid
        "max_lag": float(max_lag),
        "rho1": float(rho[0]) if len(rho) else float("nan"),
        "sum_rho": float(np.sum(rho)),
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
