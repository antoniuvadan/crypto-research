"""
Tests for the Probabilistic Sharpe Ratio and Deflated Sharpe Ratio
(significance.psr / expected_max_sharpe / dsr).

Run from the repo root:
    ~/miniforge3/envs/mscf/bin/python -m pytest tests/test_psr_dsr.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from significance import psr, expected_max_sharpe, dsr, EULER_MASCHERONI


# ---------------------------------------------------------------------------
# PSR
# ---------------------------------------------------------------------------

def test_psr_equals_half_at_own_sharpe():
    """PSR(SR*) = 0.5 exactly when the benchmark equals the estimate."""
    rng = np.random.default_rng(0)
    x = 0.3 + rng.standard_normal(500)
    sr = psr(x)["sr_hat"]
    assert psr(x, sr_star=sr)["psr"] == pytest.approx(0.5, abs=1e-9)


def test_psr_increases_with_mean_and_decreases_with_benchmark():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(400)
    low = psr(0.1 + base)["psr"]
    high = psr(0.4 + base)["psr"]
    assert high > low                                  # higher mean -> higher PSR
    x = 0.3 + base
    assert psr(x, sr_star=0.0)["psr"] > psr(x, sr_star=0.1)["psr"]  # higher bar -> lower PSR


def test_psr_effective_n_lowers_confidence():
    """Using a smaller effective sample (overlap) widens the SE and lowers PSR."""
    rng = np.random.default_rng(2)
    x = 0.2 + rng.standard_normal(300)
    full = psr(x, n_eff=None)["psr"]
    clustered = psr(x, n_eff=30)["psr"]  # 300 trades but only 30 independent episodes
    assert clustered < full


def test_psr_normality_moments_and_matches_closed_form():
    """On a large normal sample skew~0, kurtosis~3, and PSR(0) matches the closed-form
    Phi(SR*sqrt(n-1)/sqrt(1 + 0.5 SR^2))."""
    rng = np.random.default_rng(3)
    x = 0.15 + rng.standard_normal(5000)
    r = psr(x)
    assert abs(r["skew"]) < 0.12
    assert abs(r["kurtosis"] - 3.0) < 0.25
    sr = r["sr_hat"]
    closed = norm.cdf(sr * np.sqrt(len(x) - 1) / np.sqrt(1 + 0.5 * sr ** 2))
    assert r["psr"] == pytest.approx(closed, abs=1e-6)


def test_fat_tails_lower_psr_at_matched_sharpe():
    """Two series scaled to the SAME estimated Sharpe: the fat-tailed one has lower PSR."""
    rng = np.random.default_rng(4)
    normal = rng.standard_normal(2000)
    heavy = rng.standard_t(df=3, size=2000)  # fat tails, high kurtosis

    def at_sharpe(sample, target=0.1):
        z = (sample - sample.mean()) / sample.std(ddof=1)
        return z + target  # mean=target, std=1 -> SR_hat = target
    pn = psr(at_sharpe(normal))
    ph = psr(at_sharpe(heavy))
    assert ph["kurtosis"] > pn["kurtosis"]
    assert ph["psr"] < pn["psr"]


# ---------------------------------------------------------------------------
# expected_max_sharpe / DSR
# ---------------------------------------------------------------------------

def test_expected_max_sharpe_hand_value():
    # N=10, var_sr=1: z1=Phi^-1(0.9), z2=Phi^-1(1 - 1/(10e)); SR* = 0.4228*z1 + 0.5772*z2
    z1 = norm.ppf(0.9)
    z2 = norm.ppf(1 - 1 / (10 * np.e))
    expected = (1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    assert expected_max_sharpe(10, 1.0) == pytest.approx(expected)
    assert expected_max_sharpe(10, 1.0) == pytest.approx(1.5746, abs=1e-3)


def test_expected_max_sharpe_monotone_and_edge():
    assert expected_max_sharpe(1, 1.0) == 0.0          # <2 trials -> no deflation
    assert expected_max_sharpe(100, 1.0) > expected_max_sharpe(10, 1.0) > 0
    assert expected_max_sharpe(30, 4.0) == pytest.approx(2.0 * expected_max_sharpe(30, 1.0))  # sqrt(var)


def test_dsr_below_psr0_and_reduces_to_psr0_at_one_trial():
    rng = np.random.default_rng(5)
    x = 0.2 + rng.standard_normal(400)
    psr0 = psr(x, sr_star=0.0)["psr"]
    assert dsr(x, n_trials=1, var_sr=0.01)["psr"] == pytest.approx(psr0)  # N=1 -> SR*=0
    d30 = dsr(x, n_trials=30, var_sr=0.01)["psr"]
    assert d30 < psr0                                   # deflation from selection bias
    d100 = dsr(x, n_trials=100, var_sr=0.01)["psr"]
    assert d100 < d30                                   # more trials -> more deflation
