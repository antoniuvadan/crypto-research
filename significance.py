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
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329

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


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
# Bailey & Lopez de Prado (2012, 2014).
# ---------------------------------------------------------------------------

def psr(returns: np.ndarray, sr_star: float = 0.0, n_eff: float | None = None) -> dict[str, float]:
    """Probabilistic Sharpe Ratio: P(true Sharpe > sr_star), correcting for sample size and the
    non-normality (skew, kurtosis) of returns.

        PSR(SR*) = Phi[ (SR_hat - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR_hat + ((g4 - 1)/4)*SR_hat^2) ]

    SR_hat = per-observation Sharpe (mean/std, ddof=1); g3 = skewness; g4 = kurtosis (NON-excess,
    normal = 3); Phi = standard normal CDF. The denominator is the standard error of the Sharpe
    estimate: it grows with |SR_hat|, with negative skew, and with fat tails -- so those lower PSR.

    `n_eff` overrides n in the sqrt(n-1) term: pass the number of independent CLUSTERS (episodes)
    when the observations overlap, since the raw trade count overstates independence. PSR(0) is a
    finite-sample, non-normality-robust one-sided significance statement on the Sharpe
    (PSR(0) > 0.975 ~ significant at 5% one-sided; under normality PSR(0) ~ Phi(t_IID))."""
    x = np.asarray(returns, float)
    n = len(x)
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    sr = mu / sd if sd > 0 else float("nan")
    z = (x - mu) / sd  # standardised (sample sd); moments below are the standard estimators
    g3 = float(np.mean(z ** 3))
    g4 = float(np.mean(z ** 4))
    nn = float(n if n_eff is None else n_eff)
    var_sr = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) / (nn - 1.0)
    se = float(np.sqrt(var_sr)) if var_sr > 0 else float("nan")
    return {
        "sr_hat": sr, "skew": g3, "kurtosis": g4, "n": float(n), "n_used": nn,
        "se_sr": se, "psr": float(norm.cdf((sr - sr_star) / se)) if se and se == se else float("nan"),
        "sr_star": sr_star,
    }


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected MAXIMUM per-observation Sharpe across `n_trials` independent strategies whose true
    Sharpe is 0 (the benchmark SR* the DSR must beat):

        SR* = sqrt(var_sr) * [ (1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - 1/(N*e)) ]

    gamma = Euler-Mascheroni, Z^-1 = inverse normal CDF, var_sr = variance of the SR estimates
    ACROSS the N trials. More trials => higher SR* (you expect a luckier best-of-N by chance)."""
    if n_trials < 2:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def dsr(returns: np.ndarray, n_trials: int, var_sr: float, n_eff: float | None = None) -> dict[str, float]:
    """Deflated Sharpe Ratio = PSR evaluated at SR* = expected max Sharpe of N trials. It is the
    probability the strategy's true Sharpe beats what the LUCKIEST of N random trials would show,
    i.e. PSR corrected for selection bias / backtest overfitting. Needs the trial count `n_trials`
    and the cross-trial SR variance `var_sr` -- both judgement inputs; report over a range."""
    sr_star = expected_max_sharpe(n_trials, var_sr)
    out = psr(returns, sr_star=sr_star, n_eff=n_eff)
    out["n_trials"] = float(n_trials)
    out["var_sr_trials"] = var_sr
    return out


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
