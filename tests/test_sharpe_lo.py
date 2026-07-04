"""
Tests for the Lo (2002) autocorrelation-adjusted Sharpe (significance.lo_eta /
lo_annualized_sharpe / autocorrelations). Validates the annualisation factor against
hand-computed values and the key qualitative properties (deflation under positive
autocorrelation, inflation under negative, scale-invariance).

Run from the repo root:
    ~/miniforge3/envs/mscf/bin/python -m pytest tests/test_sharpe_lo.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from significance import lo_eta, autocorrelations, lo_annualized_sharpe


# ---------------------------------------------------------------------------
# lo_eta -- the annualisation factor
# ---------------------------------------------------------------------------

def test_lo_eta_zero_autocorr_recovers_sqrt_q():
    """rho == 0 must reproduce the naive rule eta = sqrt(q)."""
    for q in (12, 252, 365):
        assert lo_eta(np.zeros(5), q) == pytest.approx(np.sqrt(q))
    assert lo_eta(np.array([]), 365) == pytest.approx(np.sqrt(365))


def test_lo_eta_hand_value():
    # q=4, rho=[0.5]: corr = 4 + 2*(4-1)*0.5 = 7 -> eta = 4/sqrt(7)
    assert lo_eta(np.array([0.5]), 4) == pytest.approx(4 / np.sqrt(7))
    # q=12, rho=[0.2, 0.1]: corr = 12 + 2*[(12-1)*0.2 + (12-2)*0.1] = 12 + 2*(2.2+1.0) = 18.4
    assert lo_eta(np.array([0.2, 0.1]), 12) == pytest.approx(12 / np.sqrt(18.4))


def test_lo_eta_positive_autocorr_deflates():
    q = 252
    assert lo_eta(np.array([0.3, 0.2, 0.1]), q) < np.sqrt(q)  # eta below sqrt(q) => deflation


def test_lo_eta_negative_autocorr_inflates():
    # q=12, rho=[-0.3]: corr = 12 + 2*11*(-0.3) = 5.4 -> eta = 12/sqrt(5.4) > sqrt(12)
    assert lo_eta(np.array([-0.3]), 12) > np.sqrt(12)


def test_lo_eta_nonpositive_variance_is_nan():
    # q=2, rho=[-1]: corr = 2 + 2*(2-1)*(-1) = 0 -> not > 0 -> nan
    assert np.isnan(lo_eta(np.array([-1.0]), 2))


# ---------------------------------------------------------------------------
# autocorrelations
# ---------------------------------------------------------------------------

def test_autocorrelations_alternating_series_is_minus_one_at_lag1():
    x = np.array([1.0, -1.0] * 50)
    rho = autocorrelations(x, max_lag=3)
    assert rho[0] == pytest.approx(-(len(x) - 1) / len(x), abs=1e-9)  # ~ -1
    assert rho[1] == pytest.approx((len(x) - 2) / len(x), abs=1e-9)   # lag 2 ~ +1


def test_autocorrelations_constant_series_is_zero():
    assert np.allclose(autocorrelations(np.full(20, 3.0), max_lag=4), 0.0)


# ---------------------------------------------------------------------------
# lo_annualized_sharpe
# ---------------------------------------------------------------------------

def test_iid_series_lo_approx_naive():
    """A ~IID series (tiny autocorrelation) should give eta ~ sqrt(q) and Lo ~ naive. Uses q=12
    with a long sample and few lags so autocorrelation-estimation noise stays small (at q=365 the
    (q-k) weights amplify even tiny noise -- a real caveat, exercised in the driver's sensitivity)."""
    rng = np.random.default_rng(0)
    daily = 1.0 + rng.standard_normal(4000)  # positive mean so the Sharpe sign is stable
    r = lo_annualized_sharpe(daily, q=12, max_lag=3)
    assert r["deflation"] == pytest.approx(1.0, abs=0.1)
    assert r["sr_ann_lo"] == pytest.approx(r["sr_ann_iid"], rel=0.1)


def test_positive_ar1_deflates_annualized_sharpe():
    """AR(1) with phi>0 induces positive autocorrelation -> Lo annualised Sharpe below naive."""
    rng = np.random.default_rng(1)
    n, phi, mu = 4000, 0.6, 0.05
    e = rng.standard_normal(n)
    r = np.empty(n)
    r[0] = mu
    for t in range(1, n):
        r[t] = mu + phi * (r[t - 1] - mu) + e[t]
    out = lo_annualized_sharpe(r, q=12, max_lag=5)
    assert out["rho1"] > 0.4                       # recovers ~phi
    assert out["eta"] < out["sqrt_q"]
    assert out["sr_ann_lo"] < out["sr_ann_iid"]    # deflation (both positive-mean)


def test_scale_invariance():
    """Sharpe is scale-invariant: multiplying the P&L series by a constant (e.g. re-basing to
    return-on-capital) leaves SR_daily, eta and the annualised Sharpe unchanged."""
    rng = np.random.default_rng(2)
    daily = 0.5 + rng.standard_normal(500)
    base = lo_annualized_sharpe(daily, q=365, max_lag=10)
    scaled = lo_annualized_sharpe(1234.5 * daily, q=365, max_lag=10)
    assert scaled["sr_period"] == pytest.approx(base["sr_period"])
    assert scaled["eta"] == pytest.approx(base["eta"])
    assert scaled["sr_ann_lo"] == pytest.approx(base["sr_ann_lo"])


def test_max_lag_truncation_is_respected():
    rng = np.random.default_rng(3)
    daily = rng.standard_normal(200) + 0.2
    assert lo_annualized_sharpe(daily, q=365, max_lag=5)["max_lag"] == 5.0
    # max_lag is capped at n-2 for short series
    assert lo_annualized_sharpe(daily[:6], q=365, max_lag=50)["max_lag"] == 4.0
