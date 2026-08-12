"""
Tests for the consolidated OOS Sharpe report (strategies/oos_sharpe_report.py).

Two kinds of test here:
  * PROPERTY tests on synthetic data, where the right answer is known in closed form
    (scale invariance, the sqrt(365) identity, the Gaussian SE formula, drawdown edge cases).
  * REGRESSION LOCKS against the real OOS trades CSV, pinning this driver to the frozen,
    already-published F10 / F14 / audit figures. These are the tests that prove the driver
    RESTATES the frozen result rather than quietly re-deriving a different one.

Run from the repo root:
    ~/miniforge3/envs/mscf/bin/python -m pytest tests/test_oos_sharpe_report.py -q
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "strategies"))

from significance import psr
from sharpe_lo_adjustment import daily_pnl
from oos_sharpe_report import (
    FLOOR_S_BPS,
    HEADLINE_S_BPS,
    NOTIONAL,
    Q,
    F14_KISH_N_EFF,
    ROUND_TRIP_FEE_BPS,
    basis_row,
    concentration,
    cost_bases,
    daily_max_dd,
    floor_frame,
    gaussian_sharpe_se,
    n_episodes,
    net_bps_of,
    kish_n_eff,
    psr_ladder,
    sharpe_stats,
    window_sensitivity,
)

UTC = timezone.utc
OOS_CSV = Path("data/results/oos_test_trades.csv")
TEST_START = datetime(2024, 6, 25, tzinfo=UTC)
TEST_END = datetime(2024, 10, 15, tzinfo=UTC)

# Frozen published figures (research.md F10/F14, OOS_RISK_AUDIT).
F10_NET_BPS = 48.85
F10_IR_TRADE = 0.467
F10_ANN_SHARPE = 4.46
F10_FLOOR_BPS = {2.0: 43.13, 3.0: 41.13, 4.0: 39.13}
F14_PSR0_EPISODES = 0.968
F14_N_EPISODES_OOS = 29
F10_N_TRADES = 105
F10_ACTIVE_DAYS = 22
F10_CAL_DAYS = 112


def _needs_data() -> pl.DataFrame:
    if not OOS_CSV.exists():
        pytest.skip(f"{OOS_CSV} not present (run strategies/backtest_oos_test.py)")
    from concurrency_analysis import load

    return load(OOS_CSV)


def _trades(decision_times: list[datetime], pnl: list[float]) -> pl.DataFrame:
    """Minimal frame with the columns the daily/episode machinery reads."""
    dt = pl.Series("decision_time", decision_times).dt.replace_time_zone("UTC") \
        if decision_times and decision_times[0].tzinfo is None else pl.Series("decision_time", decision_times)
    return pl.DataFrame({
        "decision_time": dt,
        "net_pnl": pnl,
        "entry_start_time": dt,
        "exit_end_time": [t + timedelta(minutes=5) for t in decision_times],
    })


# ---------------------------------------------------------------------------
# Annualization arithmetic
# ---------------------------------------------------------------------------

def test_sharpe_is_scale_invariant():
    """The capital base cannot move the Sharpe. This is the claim the report makes in print,
    so it is tested rather than asserted: scaling every daily P&L by k leaves sr_ann fixed."""
    rng = np.random.default_rng(7)
    daily = rng.normal(120.0, 900.0, 112)
    base = sharpe_stats(daily, n_boot=200, seed=1)
    for k in (0.01, 3.0, 1e6):
        scaled = sharpe_stats(daily * k, n_boot=200, seed=1)
        assert scaled["sr_ann"] == pytest.approx(base["sr_ann"], rel=1e-12)
        assert scaled["ci_lo_gauss"] == pytest.approx(base["ci_lo_gauss"], rel=1e-12)
        # the bootstrap resamples the same INDICES under the same seed, so it is exact too
        assert scaled["ci_lo_boot_basic"] == pytest.approx(base["ci_lo_boot_basic"], rel=1e-9)


def test_sqrt_q_identity_on_known_series():
    """SR_ann must equal sqrt(365) * mean/std(ddof=1), computed independently here."""
    rng = np.random.default_rng(3)
    daily = rng.normal(5.0, 50.0, 200)
    s = sharpe_stats(daily, n_boot=100, seed=0)
    expected = daily.mean() / daily.std(ddof=1)
    assert s["sr_daily"] == pytest.approx(expected)
    assert s["sr_ann"] == pytest.approx(np.sqrt(Q) * expected)
    assert Q == 365


def test_negative_mean_gives_negative_sharpe():
    s = sharpe_stats(np.array([-10.0, -5.0, -20.0, 1.0, -3.0]), n_boot=100, seed=0)
    assert s["sr_ann"] < 0


# ---------------------------------------------------------------------------
# Daily series construction (reused from sharpe_lo_adjustment)
# ---------------------------------------------------------------------------

def test_daily_series_pads_idle_days_and_preserves_total():
    """One trade in a 5-day window -> 5 observations, 4 of them exactly zero, and the
    series must sum to the realized P&L (no leakage from the padding)."""
    df = _trades([datetime(2024, 6, 26, 12, 0, tzinfo=UTC)], [1234.5])
    daily = daily_pnl(df, datetime(2024, 6, 25, tzinfo=UTC), datetime(2024, 6, 30, tzinfo=UTC))
    assert len(daily) == 5
    assert np.count_nonzero(daily) == 1
    assert daily.sum() == pytest.approx(1234.5)
    assert daily[1] == pytest.approx(1234.5)  # booked on 6-26


def test_trade_is_booked_on_its_close_date_across_midnight():
    """A trade decided at 23:58 closes at 00:03 the NEXT day and must book there -- otherwise
    the daily series would misattribute P&L across the day boundary."""
    df = _trades([datetime(2024, 6, 25, 23, 58, tzinfo=UTC)], [500.0])
    daily = daily_pnl(df, datetime(2024, 6, 25, tzinfo=UTC), datetime(2024, 6, 28, tzinfo=UTC))
    assert daily[0] == pytest.approx(0.0)
    assert daily[1] == pytest.approx(500.0)


def test_same_day_trades_are_summed():
    df = _trades([datetime(2024, 6, 26, 1, 0, tzinfo=UTC),
                  datetime(2024, 6, 26, 15, 0, tzinfo=UTC)], [100.0, -30.0])
    daily = daily_pnl(df, datetime(2024, 6, 25, tzinfo=UTC), datetime(2024, 6, 28, tzinfo=UTC))
    assert daily[1] == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# Conservative fill floor
# ---------------------------------------------------------------------------

def test_floor_reduces_to_gross_mid_minus_fees_at_zero_spread():
    """s=0 with zero latency slip must leave exactly gross_mid - round_trip_fee."""
    df = pl.DataFrame({"gross_mid_bps": [100.0, -40.0], "latency_slip_bps": [0.0, 0.0],
                       "net_pnl": [0.0, 0.0]})
    out = floor_frame(df, 0.0)["net_pnl"].to_numpy() * 1e4 / NOTIONAL
    assert out == pytest.approx([100.0 - ROUND_TRIP_FEE_BPS, -40.0 - ROUND_TRIP_FEE_BPS])


def test_floor_charges_two_sides_and_subtracts_latency():
    df = pl.DataFrame({"gross_mid_bps": [100.0], "latency_slip_bps": [7.0], "net_pnl": [0.0]})
    out = float(floor_frame(df, 3.0)["net_pnl"][0]) * 1e4 / NOTIONAL
    assert out == pytest.approx(100.0 - 7.0 - 6.0 - ROUND_TRIP_FEE_BPS)


def test_floor_is_monotone_decreasing_in_spread():
    df = pl.DataFrame({"gross_mid_bps": [100.0, 20.0], "latency_slip_bps": [1.0, 2.0],
                       "net_pnl": [0.0, 0.0]})
    means = [float(floor_frame(df, s)["net_pnl"].mean()) for s in (0.0, 2.0, 4.0, 8.0)]
    assert means == sorted(means, reverse=True)
    # each extra bp/side costs exactly 2 bps of notional round-trip
    assert means[1] - means[2] == pytest.approx(2 * 2.0 / 1e4 * NOTIONAL)


def test_floor_preserves_columns_needed_downstream():
    """floor_frame only restates net_pnl; the episode/concurrency columns must survive."""
    oos = _needs_data()
    out = floor_frame(oos, 3.0)
    for col in ("decision_time", "entry_start_time", "exit_end_time"):
        assert col in out.columns
    assert len(out) == len(oos)


# ---------------------------------------------------------------------------
# Gaussian SE, bootstrap SD, and the documented degeneracy under zero-inflation
# ---------------------------------------------------------------------------

def test_gaussian_se_recovers_the_true_sampling_sd_by_monte_carlo():
    """Validate the SE against the thing it is supposed to estimate, not against its own formula.
    Draw 4000 independent normal samples, compute the Sharpe of each, and compare the SD of those
    Sharpes to the closed form. This is what makes the report's claim meaningful: the Gaussian SE
    is CORRECT on normal data, so its 1.7x miss on the real series is a property of the data
    (zero-inflation), not a broken formula."""
    rng = np.random.default_rng(17)
    n, mu, sigma = 120, 0.3, 1.0
    srs = np.array([(x := rng.normal(mu, sigma, n)).mean() / x.std(ddof=1) for _ in range(4000)])
    assert srs.std(ddof=1) == pytest.approx(gaussian_sharpe_se(mu / sigma, n), rel=0.06)


def test_gaussian_se_shrinks_with_sample_size():
    assert gaussian_sharpe_se(0.3, 1000) < gaussian_sharpe_se(0.3, 100) < gaussian_sharpe_se(0.3, 10)


def test_psr_se_reduces_to_gaussian_under_normality():
    """Sanity on the two SEs agreeing where they should: for a normal series (skew 0, kurt 3)
    the PSR variance 1 - g3*SR + (g4-1)/4*SR^2 collapses to 1 + SR^2/2."""
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 1.0, 40_000)
    p = psr(x, sr_star=0.0)
    assert p["se_sr"] == pytest.approx(gaussian_sharpe_se(p["sr_hat"], len(x)), rel=0.02)


def test_zero_inflated_series_makes_moment_se_artificially_small():
    """The degeneracy the report warns about, pinned as a test: on a series that is mostly exact
    zeros the sample skew is large and positive, which SHRINKS the PSR standard error below the
    Gaussian one and drives PSR(0) to ~1. This is why the report's primary CI is the block bootstrap, not either closed form."""
    daily = np.zeros(112)
    daily[:22] = np.array([900.0] * 21 + [12_000.0])
    s = sharpe_stats(daily, n_boot=200, seed=0)
    assert s["skew_daily"] > 3.0
    assert s["se_moment"] < s["se_gauss"]
    assert psr(daily, sr_star=0.0)["psr"] > 0.999
    # ... and therefore the moment interval is strictly narrower than the Gaussian one
    assert (s["ci_hi_moment"] - s["ci_lo_moment"]) < (s["ci_hi_gauss"] - s["ci_lo_gauss"])


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_is_deterministic_under_seed():
    rng = np.random.default_rng(5)
    daily = rng.normal(50.0, 400.0, 112)
    a = sharpe_stats(daily, n_boot=400, seed=42)
    b = sharpe_stats(daily, n_boot=400, seed=42)
    assert a["ci_lo_boot_pct"] == pytest.approx(b["ci_lo_boot_pct"])
    assert a["ci_hi_boot_basic"] == pytest.approx(b["ci_hi_boot_basic"])


def test_basic_bootstrap_ci_is_the_reflection_of_the_percentile_ci():
    """basic = [2*theta_hat - q_hi, 2*theta_hat - q_lo] -- the identity that removes the
    centring shift. Checked against the stored percentile bounds."""
    rng = np.random.default_rng(9)
    daily = rng.normal(50.0, 400.0, 112)
    s = sharpe_stats(daily, n_boot=500, seed=3)
    assert s["ci_lo_boot_basic"] == pytest.approx(2 * s["sr_ann"] - s["ci_hi_boot_pct"])
    assert s["ci_hi_boot_basic"] == pytest.approx(2 * s["sr_ann"] - s["ci_lo_boot_pct"])


def test_bootstrap_sd_agrees_with_the_gaussian_se_on_well_behaved_data():
    """The two estimators must AGREE where both are valid (IID normal). This is the control for
    the report's central statistical claim: when they disagree by 1.7x on the real series, that is
    evidence about the series, not about one estimator being broken."""
    rng = np.random.default_rng(23)
    daily = rng.normal(40.0, 300.0, 400)
    s = sharpe_stats(daily, n_boot=4000, seed=1)
    assert s["boot_sd"] == pytest.approx(s["se_gauss_ann"], rel=0.15)


def test_zero_inflation_breaks_the_closed_form_ses_in_either_direction():
    """The general claim is that zero-inflation makes the closed-form SEs UNRELIABLE -- not that
    they always err wide. Direction depends on the specific series: the real OOS series has the
    Gaussian SE ~1.7x too wide, while this synthetic one (a large adverse day among the active
    days) puts it ~1.5x too narrow. Hence the report pins the direction only on the real data
    (test_real_daily_series_moments_are_degenerate_as_documented) and claims only mis-calibration
    in general. Written as a test so the weaker claim is the one that is enforced."""
    daily = np.zeros(112)
    daily[:22] = np.array([700.0] * 20 + [-2_000.0, 9_000.0])
    s = sharpe_stats(daily, n_boot=4000, seed=1)
    assert s["boot_sd"] > 1.3 * s["se_gauss_ann"], "this series errs the OTHER way"
    ratio = max(s["se_gauss_ann"], s["boot_sd"]) / min(s["se_gauss_ann"], s["boot_sd"])
    assert ratio > 1.3, "closed form and bootstrap must disagree materially under zero-inflation"


def test_bootstrap_sd_shrinks_with_sample_size():
    rng = np.random.default_rng(13)
    short = sharpe_stats(rng.normal(50.0, 100.0, 40), 2000, 0)
    long_ = sharpe_stats(rng.normal(50.0, 100.0, 400), 2000, 0)
    assert long_["boot_sd"] < short["boot_sd"]


def test_block_length_actually_affects_the_bootstrap():
    """Guards BOOT_BLOCK: a mutation to block=1 would silently turn the block bootstrap into an
    iid one. On an autocorrelated series the two must give different spreads."""
    from backtest_oos_risk import block_bootstrap_sharpe_dist

    rng = np.random.default_rng(31)
    x = np.zeros(200)
    for i in range(1, 200):  # AR(1), rho = 0.7 -> block length must matter
        x[i] = 0.7 * x[i - 1] + rng.normal(1.0, 1.0)
    sd1 = block_bootstrap_sharpe_dist(x, block=1, n_boot=2000, seed=0).std(ddof=1)
    sd14 = block_bootstrap_sharpe_dist(x, block=14, n_boot=2000, seed=0).std(ddof=1)
    assert abs(sd14 - sd1) / sd1 > 0.10


def test_ci_percentile_reduction_matches_the_shared_helper():
    """sharpe_stats inlines the percentile reduction over the distribution helper; it must stay
    identical to the shared block_bootstrap_sharpe_ci used by the other drivers."""
    from backtest_oos_risk import block_bootstrap_sharpe_ci

    rng = np.random.default_rng(41)
    daily = rng.normal(30.0, 200.0, 112)
    s = sharpe_stats(daily, n_boot=1000, seed=7)
    lo, mid, hi = block_bootstrap_sharpe_ci(daily, block=7, n_boot=1000, seed=7)
    assert (s["ci_lo_boot_pct"], s["boot_median"], s["ci_hi_boot_pct"]) == \
           pytest.approx((lo, mid, hi))


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def test_daily_dd_hand_computed():
    # equity path 0 -> 10 -> 4 -> 14 -> 9 ; worst trough is 10 -> 4
    assert daily_max_dd(np.array([10.0, -6.0, 10.0, -5.0])) == pytest.approx(-6.0)


def test_daily_dd_counts_an_opening_loss():
    """The zero-prepend: a series that loses on day 1 is in drawdown from the start. Plain
    maximum.accumulate on the cumsum would anchor the peak at equity[0] and report 0."""
    assert daily_max_dd(np.array([-500.0, 100.0])) == pytest.approx(-500.0)
    assert daily_max_dd(np.array([-500.0, -100.0])) == pytest.approx(-600.0)


def test_daily_dd_is_never_positive_and_zero_for_monotone_gains():
    assert daily_max_dd(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)
    assert daily_max_dd(np.array([])) == 0.0
    rng = np.random.default_rng(2)
    assert daily_max_dd(rng.normal(0, 100, 200)) <= 0.0


def test_worse_cost_basis_gives_worse_drawdown_on_real_data():
    """Monotonicity across the floor grid: a wider assumed spread cannot improve the drawdown."""
    oos = _needs_data()
    dds = [daily_max_dd(daily_pnl(floor_frame(oos, s), TEST_START, TEST_END))
           for s in (0.0, 2.0, 4.0, 8.0)]
    assert dds == sorted(dds, reverse=True)


# ---------------------------------------------------------------------------
# Regression locks against the frozen published result
# ---------------------------------------------------------------------------

def test_realized_basis_reproduces_f10_headline():
    """The control. If this drifts, the driver is no longer restating the frozen OOS result."""
    oos = _needs_data()
    row = basis_row("realized", oos, 900_000.0, n_boot=500, seed=0)
    assert row["n_trades"] == F10_N_TRADES
    assert row["cal_days"] == F10_CAL_DAYS
    assert row["active_days"] == F10_ACTIVE_DAYS
    assert row["mean_net_bps"] == pytest.approx(F10_NET_BPS, abs=0.05)
    assert row["ir_per_trade"] == pytest.approx(F10_IR_TRADE, abs=0.005)
    assert row["sr_ann"] == pytest.approx(F10_ANN_SHARPE, abs=0.05)
    assert row["t_nw"] == pytest.approx(2.62, abs=0.05)


def test_floor_grid_reproduces_f10_published_means():
    """+43.13 / +41.13 / +39.13 bps at s = 2/3/4, exactly as published in F10."""
    oos = _needs_data()
    for s in FLOOR_S_BPS:
        mean = float(net_bps_of(floor_frame(oos, s)).mean())
        assert mean == pytest.approx(F10_FLOOR_BPS[s], abs=0.02), f"s={s}"


def test_episode_count_and_psr_match_f14():
    """F14 published 29 OOS episodes and PSR(0) = 0.968 on the realized-fill per-trade series;
    both are recomputed here from the CSV, so agreement is a genuine cross-check."""
    oos = _needs_data()
    assert n_episodes(oos) == F14_N_EPISODES_OOS
    row = basis_row("realized", oos, 900_000.0, n_boot=200, seed=0)
    assert row["psr0_episodes"] == pytest.approx(F14_PSR0_EPISODES, abs=0.002)


def test_headline_is_more_conservative_than_the_realized_basis():
    """The whole point of moving the headline onto the floor: it must not flatter the result."""
    oos = _needs_data()
    head = basis_row("floor", floor_frame(oos, HEADLINE_S_BPS), 900_000.0, n_boot=200, seed=0)
    ctl = basis_row("realized", oos, 900_000.0, n_boot=200, seed=0)
    assert head["mean_net_bps"] < ctl["mean_net_bps"]
    assert head["sr_ann"] < ctl["sr_ann"]
    assert head["t_nw"] < ctl["t_nw"]


def test_published_constants_are_locked():
    """Adversarial mutation testing showed the whole suite passed with Z90 silently changed to
    1.96 (turning every '90%' CI into a 95% one), the headline spread changed to 2bps, the
    bootstrap block to 1, and RF_ANNUAL to 0. Each is now pinned against a literal, because each
    changes a PUBLISHED number while leaving every behavioural test green."""
    from scipy.stats import norm

    import oos_sharpe_report as rep

    assert rep.Z90 == pytest.approx(float(norm.ppf(0.95)))   # 90% two-sided, not 95%
    assert rep.HEADLINE_S_BPS == 3.0
    assert rep.FLOOR_S_BPS == (2.0, 3.0, 4.0)
    assert rep.BOOT_BLOCK == 7
    assert rep.RF_ANNUAL == 0.0525
    assert rep.Q == 365
    assert rep.ROUND_TRIP_FEE_BPS == 10.0
    assert rep.LO_MAX_LAGS == (1, 5, 10, 20), "F14's full grid; L=20 must not be dropped"
    assert rep.NOTIONAL == 50_000.0
    # FROZEN_MTM_DD_USD started as a pure guard but is now a COMPUTATIONAL INPUT (it feeds
    # ret_over_mtm_dd and the headline block's printed drawdown), so it needs locking too.
    assert rep.FROZEN_MTM_DD_USD == -22_770.0
    assert rep.F14_KISH_N_EFF == 16


def test_headline_ci_bounds_are_locked():
    """No published CI bound was locked anywhere before this test; any of the mutations above
    would silently move them. Tolerances are wide enough for bootstrap jitter at n_boot=5000."""
    oos = _needs_data()
    row = basis_row("floor", floor_frame(oos, HEADLINE_S_BPS), 900_000.0, n_boot=5000, seed=0)
    assert row["sr_ann"] == pytest.approx(4.15, abs=0.02)
    assert row["ci90_boot_lo"] == pytest.approx(1.59, abs=0.10)
    assert row["ci90_boot_hi"] == pytest.approx(5.15, abs=0.10)
    assert row["ci90_gauss_lo"] == pytest.approx(1.14, abs=0.02)
    assert row["ci90_gauss_hi"] == pytest.approx(7.17, abs=0.02)
    assert row["psr0_episodes"] == pytest.approx(0.947, abs=0.002)


def test_real_daily_series_moments_are_degenerate_as_documented():
    """The degeneracy is asserted about the REAL series here, not only a synthetic stand-in:
    the printed skew/kurt and the resulting SE mis-calibration are pinned."""
    oos = _needs_data()
    daily = daily_pnl(floor_frame(oos, HEADLINE_S_BPS), TEST_START, TEST_END)
    s = sharpe_stats(daily, n_boot=5000, seed=0)
    assert np.count_nonzero(daily == 0) == 90
    assert s["skew_daily"] == pytest.approx(7.50, abs=0.05)
    assert s["kurt_daily"] == pytest.approx(68.4, abs=0.5)
    # the two closed forms bracket the empirical SD in the documented directions
    assert s["se_moment_ann"] < s["boot_sd"] < s["se_gauss_ann"]


def test_net_bps_is_time_ordered_regardless_of_input_order():
    """The .sort in net_bps_of is load-bearing: hac_stats computes a Newey-West SE, which is only
    meaningful on a time-ordered series. Removing the sort passed the entire original suite."""
    oos = _needs_data()
    ordered = net_bps_of(oos)
    shuffled = net_bps_of(oos.sample(fraction=1.0, shuffle=True, seed=5))
    assert shuffled == pytest.approx(ordered)


def test_window_sensitivity_is_monotone_and_flags_the_dead_tail():
    """Padding a fixed P&L series with idle days shrinks SR_ann ~ 1/sqrt(T). All three horizons
    AND all three Sharpes are locked: the third row is the one quoted in the summary text, and it
    silently desynced from a hard-coded literal once already."""
    oos = _needs_data()
    rows = window_sensitivity(floor_frame(oos, HEADLINE_S_BPS))
    ts = [t for _, t, _ in rows]
    srs = [sr for _, _, sr in rows]
    assert ts == [74, 112, 294], "window horizons are published numbers"
    assert srs == sorted(srs, reverse=True), "more idle padding must lower the Sharpe"
    assert srs[0] == pytest.approx(5.16, abs=0.02)   # to last trade, 2024-09-06
    assert srs[1] == pytest.approx(4.15, abs=0.02)   # as published
    assert srs[2] == pytest.approx(2.53, abs=0.02)   # + 6 idle months
    assert rows[0][0].endswith("(2024-09-06)"), "the last-trade date is a published claim"


def test_window_sensitivity_uses_close_time_not_decision_time():
    """A trade decided at 23:58 closes the next day. Keying the window off decision_time would put
    that close outside [start, last+1d) and silently drop it from its own last-trade window."""
    df = _trades([datetime(2024, 6, 26, 12, 0, tzinfo=UTC),
                  datetime(2024, 6, 30, 23, 58, tzinfo=UTC)], [100.0, 400.0])
    rows = window_sensitivity(df)
    assert rows[0][0].endswith("(2024-07-01)"), "labelled by CLOSE date, not decision date"
    # rows[..][1] is a DAY COUNT, not a datetime: the last-trade window must run 2024-06-25
    # through the 07-01 close inclusive = 7 days. Keying off decision_time would give 6 and
    # silently drop the $400 trade.
    assert rows[0][1] == 7
    assert daily_pnl(df, TEST_START, datetime(2024, 7, 2, tzinfo=UTC)).sum() == pytest.approx(500.0)


def test_mtm_ratio_is_only_defined_for_the_realized_basis():
    """The MtM drawdown marks a real mid path; a per-trade cost offset does not define one.
    Imputing the realized -$22,770 to the floor rows would be a flattering upper bound (deeper
    drawdown is provable: 20 trades were entered on the day carrying 86% of it)."""
    oos = _needs_data()
    real = basis_row("realized-sweep fills (F10)", oos, 900_000.0, 200, 0, is_realized=True)
    floor = basis_row("floor", floor_frame(oos, HEADLINE_S_BPS), 900_000.0, 200, 0)
    assert real["ret_over_mtm_dd"] == pytest.approx(25_646.9 / 22_770.0, rel=1e-3)
    assert np.isnan(floor["ret_over_mtm_dd"]), "floor bases must not borrow the realized MtM DD"
    # the daily-DD ratio IS defined for every basis -- and is ~30x flattering, which is why the
    # column is named for its drawdown definition
    against_mtm = abs(floor["total_net_usd"]) / 22_770.0
    assert floor["ret_over_dd_daily"] > 25 * against_mtm


def test_concentration_flags_the_single_dominant_day():
    oos = _needs_data()
    daily = daily_pnl(floor_frame(oos, HEADLINE_S_BPS), TEST_START, TEST_END)
    c = concentration(daily)
    assert c["top_day_share"] == pytest.approx(0.392, abs=0.005)
    assert c["top3_share"] == pytest.approx(0.582, abs=0.005)
    # zeroing it RAISES the Sharpe: the day carries more variance than mean
    assert c["sr_ann_ex_top"] > daily.mean() / daily.std(ddof=1) * np.sqrt(365)
    assert c["sr_ann_ex_top"] == pytest.approx(5.54, abs=0.02)


def test_concentration_zeroes_rather_than_deletes_the_top_day():
    """Deleting would shorten the window by a day, mixing the concentration effect with the
    length effect that window_sensitivity exists to isolate -- and would contradict the driver's
    'idle days = 0' convention. Pinned by comparing against both counterfactuals explicitly."""
    oos = _needs_data()
    daily = daily_pnl(floor_frame(oos, HEADLINE_S_BPS), TEST_START, TEST_END)
    top = int(np.argmax(daily))

    zeroed = daily.copy()
    zeroed[top] = 0.0
    sr_zeroed = zeroed.mean() / zeroed.std(ddof=1) * np.sqrt(365)
    deleted = np.delete(daily, top)
    sr_deleted = deleted.mean() / deleted.std(ddof=1) * np.sqrt(365)

    assert len(zeroed) == len(daily), "zeroing preserves the 112-day window"
    assert sr_zeroed != pytest.approx(sr_deleted, abs=1e-6), "the two differ; the choice matters"
    assert concentration(daily)["sr_ann_ex_top"] == pytest.approx(sr_zeroed)


def test_concentration_on_a_synthetic_series_is_exact():
    daily = np.array([0.0, 100.0, 0.0, 300.0, 100.0])
    c = concentration(daily)
    assert c["top_day_usd"] == pytest.approx(300.0)
    assert c["top_day_share"] == pytest.approx(0.6)
    assert c["top3_share"] == pytest.approx(1.0)


def test_psr_ladder_includes_the_most_conservative_kish_rung():
    """F14 §4 lists a Kish N_eff = 16 rung; omitting it would narrow the reported PSR range in
    the flattering direction. It is carried with attribution, not recomputed."""
    oos = _needs_data()
    nb = net_bps_of(floor_frame(oos, HEADLINE_S_BPS))
    ladder = psr_ladder(nb, n_ep=29, n_traded=22, n_kish=kish_n_eff(oos))
    assert [l for l, _, _ in ladder][-1] == "Kish"
    # COMPUTED from the repo's own effective_n (n^2 / sum cluster_size^2), not quoted -- and it
    # independently recovers F14 SS4's published 16, the same kind of cross-check as psr0_episodes
    # recovering F14's 0.968.
    assert ladder[-1][1] == pytest.approx(15.96, abs=0.02)
    assert round(ladder[-1][1]) == F14_KISH_N_EFF
    assert ladder[-1][2] == pytest.approx(0.882, abs=0.005)
    assert ladder[-1][2] < ladder[-2][2], "Kish is the most conservative rung"


def test_psr_ladder_is_monotone_decreasing_in_conservatism():
    """Coarser effective sample -> smaller sqrt(n_eff - 1) -> lower PSR. The ladder must be
    ordered, and the traded-day rung must be the lowest (it is what the report headlines)."""
    oos = _needs_data()
    nb = net_bps_of(floor_frame(oos, HEADLINE_S_BPS))
    ladder = psr_ladder(nb, n_ep=29, n_traded=22, n_kish=kish_n_eff(oos))
    labels = [l for l, _, _ in ladder]
    vals = [v for _, _, v in ladder]
    assert labels == ["raw trades", "episodes", "traded days", "Kish"]
    assert vals == sorted(vals, reverse=True)
    traded = dict((l, v) for l, _, v in ladder)["traded days"]
    assert traded == pytest.approx(0.919, abs=0.003)
    assert traded < 0.95, "the traded-day rung falls below the 0.95 line -- the report says so"


def test_cash_drag_counterfactual_arithmetic():
    """The +1.37 figure: subtracting a constant daily charge shifts the mean and leaves the SD
    untouched, so the result is a pure mean shift -- verified independently here."""
    oos = _needs_data()
    daily = daily_pnl(floor_frame(oos, HEADLINE_S_BPS), TEST_START, TEST_END)
    peak_cap, rf = 900_000.0, 0.0525
    excess = daily - peak_cap * rf / 365.0
    assert excess.std(ddof=1) == pytest.approx(daily.std(ddof=1))
    assert excess.mean() / excess.std(ddof=1) * np.sqrt(365) == pytest.approx(1.37, abs=0.02)


def test_cost_bases_without_the_stress_csv():
    """The driver must still work when only the frozen OOS CSV is present."""
    oos = _needs_data()
    bases = cost_bases(oos, None)
    # headline s=3, realized-sweep, floor s=2, floor s=4 -- the two participation rows drop out
    assert len(bases) == 4
    assert not any("participation" in lbl for lbl, _ in bases)
    assert all(len(df) == F10_N_TRADES for _, df in bases)


def test_floor_frame_propagates_nulls_rather_than_silently_zeroing():
    """If a mid were ever missing, the floor must produce a null (visible) not a 0 (invisible)."""
    df = pl.DataFrame({"gross_mid_bps": [100.0, None], "latency_slip_bps": [1.0, 1.0],
                       "net_pnl": [0.0, 0.0]})
    out = floor_frame(df, 3.0)["net_pnl"]
    assert out[0] is not None
    assert out[1] is None


def test_daily_pnl_would_drop_a_trade_closing_outside_the_calendar_range():
    """Documents a real property of the reused helper: it left-joins onto a fixed calendar, so a
    trade closing outside [start, end) is silently dropped. Harmless for the frozen OOS frame (all
    105 close inside), but the behaviour is pinned so a future window change cannot lose P&L
    unnoticed."""
    df = _trades([datetime(2024, 6, 26, 12, 0, tzinfo=UTC),
                  datetime(2024, 7, 30, 12, 0, tzinfo=UTC)], [100.0, 999.0])
    daily = daily_pnl(df, datetime(2024, 6, 25, tzinfo=UTC), datetime(2024, 6, 28, tzinfo=UTC))
    assert daily.sum() == pytest.approx(100.0), "the out-of-range trade is dropped, not clamped"
    # ... and the frozen frame does NOT trip this
    oos = _needs_data()
    assert daily_pnl(oos, TEST_START, TEST_END).sum() == pytest.approx(float(oos["net_pnl"].sum()))


def test_cost_bases_are_all_distinct_and_headline_is_first():
    oos = _needs_data()
    from concurrency_analysis import load

    stress_path = Path("data/results/oos_test_slippage_stress_trades.csv")
    stress = load(stress_path) if stress_path.exists() else None
    bases = cost_bases(oos, stress)
    labels = [b[0] for b in bases]
    # asserted against the LITERAL 3, not against HEADLINE_S_BPS -- otherwise the test moves
    # with the constant it is supposed to be pinning
    assert labels[0].startswith("floor s=3bps")
    # "[PRIMARY BASIS]", not "[HEADLINE]": the CSV label travels without the caveats
    assert "[PRIMARY BASIS]" in labels[0]
    assert "HEADLINE" not in labels[0]
    assert len(labels) == len(set(labels))
    totals = [float(df["net_pnl"].sum()) for _, df in bases]
    assert len(totals) == len(set(round(t, 6) for t in totals)), "cost bases must differ"


def test_built_table_flags_exactly_one_realized_basis():
    """INTEGRATION lock on the main() wiring, not just on basis_row's gate. basis_row honours
    is_realized, but the predicate that decides WHICH row gets it lives in main(); breaking it
    would silently re-impute the realized -$22,770 to all six rows (the B1 defect) with a green
    suite. Mirrors main()'s construction exactly."""
    oos = _needs_data()
    from concurrency_analysis import load

    stress_path = Path("data/results/oos_test_slippage_stress_trades.csv")
    stress = load(stress_path) if stress_path.exists() else None
    rows = [basis_row(lbl, df, 900_000.0, 200, 0, is_realized=lbl.startswith("realized-sweep"))
            for lbl, df in cost_bases(oos, stress)]

    defined = [r for r in rows if not np.isnan(r["ret_over_mtm_dd"])]
    assert len(defined) == 1, "exactly one basis may carry an MtM-based ratio"
    assert defined[0]["cost_basis"].startswith("realized-sweep")
    assert len(rows) == 6


def test_all_bases_share_the_same_trade_count():
    """Every basis re-prices the SAME 105 kept trades; a differing n would mean the trim or the
    stress CSV silently dropped events, which would make the Sharpes incomparable."""
    oos = _needs_data()
    from concurrency_analysis import load

    stress_path = Path("data/results/oos_test_slippage_stress_trades.csv")
    stress = load(stress_path) if stress_path.exists() else None
    for label, df in cost_bases(oos, stress):
        assert len(df) == F10_N_TRADES, label
