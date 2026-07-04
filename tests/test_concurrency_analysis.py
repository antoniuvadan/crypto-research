"""
Tests for the concurrency-analysis primitives (strategies/concurrency_analysis.py):
episode clustering, the greedy concurrency cap, effective sample size, the cluster-robust
t, and the concurrency/directionality profile.

Run from the repo root under the polars env:
    ~/miniforge3/envs/mscf/bin/python -m pytest tests/test_concurrency_analysis.py -q
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

from concurrency_analysis import (
    assign_episodes, effective_n, cluster_robust_t, cap_at_n, dedupe_per_episode,
    concurrency_profile, net_bps,
)
from significance import hac_stats

UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _trades(rows):
    """rows: list of (entry_s, exit_s, direction, net_pnl, disp). Builds a trades frame with
    the columns the concurrency primitives read."""
    recs = []
    for i, (es, xs, d, npnl, disp) in enumerate(rows):
        recs.append({
            "decision_time": BASE + timedelta(seconds=es),  # decision ~ entry for tests
            "entry_start_time": BASE + timedelta(seconds=es),
            "exit_end_time": BASE + timedelta(seconds=xs),
            "direction": d, "net_pnl": float(npnl), "displacement_bps": float(disp),
        })
    return pl.DataFrame(recs)


# ---------------------------------------------------------------------------
# episode clustering
# ---------------------------------------------------------------------------

def test_assign_episodes_overlap_chains_and_gaps_split():
    # A[0,10] B[5,15] overlap -> one episode; C[100,110] far -> new episode;
    # D[112,120] within 5s of C's end -> chains to C at tau=5s but splits at tau=0.
    df = _trades([(0, 10, 1, 1, 1), (5, 15, 1, 1, 1), (100, 110, 1, 1, 1), (112, 120, 1, 1, 1)])
    ep0 = assign_episodes(df, timedelta(0))
    assert list(ep0) == [0, 0, 1, 2]          # D splits from C (gap 2s > 0)
    ep5 = assign_episodes(df, timedelta(seconds=5))
    assert list(ep5) == [0, 0, 1, 1]          # D chains to C (gap 2s <= 5s)


def test_assign_episodes_transitive_chaining():
    # A[0,10] B[8,20] C[18,30]: A-C don't overlap but chain through B -> one episode.
    df = _trades([(0, 10, 1, 1, 1), (8, 20, 1, 1, 1), (18, 30, 1, 1, 1)])
    assert list(assign_episodes(df, timedelta(0))) == [0, 0, 0]


def test_assign_episodes_preserves_original_row_order():
    # rows given out of entry-start order; episode labels must map back to df's row order.
    df = _trades([(100, 110, 1, 1, 1), (0, 10, 1, 1, 1), (5, 15, 1, 1, 1)])
    ep = assign_episodes(df, timedelta(0))
    assert list(ep) == [1, 0, 0]  # row0 is the late one -> episode 1; rows1,2 overlap -> 0


# ---------------------------------------------------------------------------
# effective sample size
# ---------------------------------------------------------------------------

def test_effective_n_formula():
    assert effective_n(np.array([0, 1, 2]))["n_eff"] == pytest.approx(3.0)      # 3 singletons
    assert effective_n(np.array([0, 0, 0]))["n_eff"] == pytest.approx(1.0)      # one 3-cluster
    assert effective_n(np.array([0, 0, 1, 1]))["n_eff"] == pytest.approx(2.0)   # two 2-clusters


# ---------------------------------------------------------------------------
# cluster-robust t
# ---------------------------------------------------------------------------

def test_cluster_robust_t_equals_iid_for_singletons():
    """With every trade its own cluster, the finite-sample-corrected cluster SE equals the
    IID SE exactly -- a strong algebraic check of the formula."""
    y = np.array([10.0, -5.0, 30.0, 12.0, -8.0, 40.0])
    ep = np.arange(len(y))  # all singletons
    cr = cluster_robust_t(y, ep)
    iid = hac_stats(y)
    assert cr["t_cluster"] == pytest.approx(iid["t_iid"], rel=1e-9)


def test_cluster_robust_t_degenerate_single_cluster():
    """One cluster containing everything: residuals sum to zero -> zero SE, no inference."""
    y = np.array([10.0, -5.0, 30.0])
    cr = cluster_robust_t(y, np.zeros(3, dtype=int))
    assert cr["se_cluster"] == pytest.approx(0.0, abs=1e-9)


def test_cluster_robust_t_widens_se_under_positive_within_cluster_correlation():
    """Clusters whose members share the same residual sign inflate the cluster SE above IID
    (t falls). Uses a positive-mean series so the t-stats are non-degenerate."""
    y = np.array([100.0, 100.0, 0.0, 0.0])  # cluster A both high, B both low; mean 50
    ep = np.array([0, 0, 1, 1])
    cr = cluster_robust_t(y, ep)
    iid = hac_stats(y)
    assert cr["t_cluster"] > 0 and abs(cr["t_cluster"]) < abs(iid["t_iid"])


# ---------------------------------------------------------------------------
# greedy concurrency cap
# ---------------------------------------------------------------------------

def test_cap_at_n_identity_when_n_large():
    df = _trades([(0, 10, 1, 1, 1), (5, 15, 1, 1, 1), (6, 16, 1, 1, 1)])
    assert len(cap_at_n(df, 999)) == 3


def test_cap_at_n_one_keeps_first_skips_overlap_takes_next_after_release():
    # A[0,10] taken; B[5,15] skipped (A open); C[10,20] taken (A released at 10);
    # searchsorted/release uses x > s, so C at s=10 sees A's end 10 NOT > 10 -> released.
    df = _trades([(0, 10, 1, 1, 1), (5, 15, 1, 1, 1), (10, 20, 1, 1, 1)])
    taken = cap_at_n(df, 1)
    assert taken["entry_start_time"].to_list() == [BASE, BASE + timedelta(seconds=10)]


def test_cap_at_n_result_never_exceeds_n_concurrent():
    # dense overlap: 6 trades all open together; cap 2 must yield peak <= 2.
    df = _trades([(i, 100, 1, 1, 1) for i in range(6)])
    for n in (1, 2, 3):
        capped = cap_at_n(df, n)
        assert concurrency_profile(capped)["peak"] <= n


# ---------------------------------------------------------------------------
# dedupe per episode / directionality
# ---------------------------------------------------------------------------

def test_dedupe_per_episode_keeps_max_displacement():
    # one episode of three overlapping trades with displacements 5, 20, 8 -> keep the 20.
    df = _trades([(0, 10, 1, 1, 5.0), (2, 12, 1, 1, 20.0), (4, 14, 1, 1, 8.0)])
    kept = dedupe_per_episode(df, timedelta(0))
    assert len(kept) == 1
    assert kept["displacement_bps"][0] == pytest.approx(20.0)


def test_dedupe_nan_displacement_not_selected():
    """A NaN key must NOT be kept over a real max (regression for the descending-sort NaN bug)."""
    df = _trades([(0, 10, 1, 5.0, np.nan), (2, 12, 1, 99.0, 20.0), (4, 14, 1, 3.0, 8.0)])
    kept = dedupe_per_episode(df, timedelta(0))
    assert len(kept) == 1
    assert kept["displacement_bps"][0] == pytest.approx(20.0)
    assert kept["net_pnl"][0] == pytest.approx(99.0)


def test_dedupe_missing_key_raises():
    df = _trades([(0, 10, 1, 5.0, 1.0)]).drop("displacement_bps")
    with pytest.raises(ValueError):
        dedupe_per_episode(df, timedelta(0))


def test_dedupe_multi_episode_keeps_one_max_each():
    # episode A [0,10]/[2,12] (disp 5,20) -> keep 20; episode B [100,110]/[102,112] (disp 9,4) -> keep 9.
    df = _trades([(0, 10, 1, 1, 5.0), (2, 12, 1, 1, 20.0), (100, 110, 1, 1, 9.0), (102, 112, 1, 1, 4.0)])
    kept = dedupe_per_episode(df, timedelta(0)).sort("decision_time")
    assert kept["displacement_bps"].to_list() == [20.0, 9.0]


def test_assign_episodes_tied_entry_start():
    # three trades share entry_start=0 with different exits -> all one overlapping episode;
    # a far trade -> its own. Exercises the tie-heavy path of the label round-trip (R1).
    df = _trades([(0, 8, 1, 1, 1), (0, 20, 1, 1, 1), (0, 5, 1, 1, 1), (100, 110, 1, 1, 1)])
    assert list(assign_episodes(df, timedelta(0))) == [0, 0, 0, 1]


def test_cluster_robust_t_can_exceed_iid_when_within_cluster_correlation_negative():
    """When clusters mix high+low members (negative within-cluster correlation), the cluster
    SE is SMALLER than IID and t is LARGER -- proving Kish N_eff (rho=1) understates the true
    independence (the D1 point)."""
    y = np.array([100.0, 20.0, 60.0, 20.0])  # mean 50; each cluster has a high and a low
    ep = np.array([0, 0, 1, 1])
    cr = cluster_robust_t(y, ep)
    iid = hac_stats(y)
    assert cr["t_cluster"] > iid["t_iid"]


def test_effective_n_unequal_sizes():
    assert effective_n(np.array([0, 0, 0, 1]))["n_eff"] == pytest.approx(16 / 10)  # 4^2/(9+1)


def test_cluster_robust_t_single_cluster_is_nan():
    cr = cluster_robust_t(np.array([10.0, -5.0, 30.0]), np.zeros(3, dtype=int))
    assert np.isnan(cr["t_cluster"])


def test_oneway_fraction_is_duration_weighted():
    """A brief opposite-direction overlap inside a long same-direction hold should pull the
    duration-weighted one-way fraction below 1 but keep it high (not the 50% an event-point
    count might imply)."""
    # long open [0,100]; a short blips in only for [10,12].
    df = _trades([(0, 100, 1, 1, 1), (10, 12, -1, 1, 1)])
    p = concurrency_profile(df)
    assert 0.9 < p["frac_open_time_oneway"] < 1.0  # ~ (100-2)/100 open time is one-way


def test_concurrency_profile_direction_and_oneway():
    # two same-direction longs overlapping -> peak 2, fully one-way.
    same = _trades([(0, 10, 1, 1, 1), (5, 15, 1, 1, 1)])
    p = concurrency_profile(same)
    assert p["peak"] == 2 and p["frac_open_time_oneway"] == pytest.approx(1.0)
    # a long and a short overlapping -> at least one event-point is NOT one-way.
    opp = _trades([(0, 10, 1, 1, 1), (5, 15, -1, 1, 1)])
    p2 = concurrency_profile(opp)
    assert p2["peak"] == 2 and p2["frac_open_time_oneway"] < 1.0
