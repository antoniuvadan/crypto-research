"""
Tests for the train-only concurrency-cap driver (strategies/concurrency_train_caps.py):
the dev/holdout split, the causality (no look-ahead) of the greedy cap, cap monotonicity,
the pre-registered cap-selection rule, and the additive cost decomposition it reuses from
pnl_decomposition.

Run from the repo root under the polars env:
    ~/miniforge3/envs/mscf/bin/python -m pytest tests/test_concurrency_train_caps.py -q
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

import pnl_decomposition as pnld
import concurrency_train_caps as ctc
from concurrency_analysis import cap_at_n, concurrency_profile, load, TRAIN_CSV
from concurrency_train_caps import (
    HOLDOUT_START, TRAIN_END, CAPS, policies, split_dev_holdout, select_cap,
    _pnl_share, _realized_dd_day,
)

UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _trades(rows):
    """rows: (entry_s, exit_s, direction, net_pnl, disp) at second offsets from BASE."""
    recs = []
    for es, xs, d, npnl, disp in rows:
        recs.append({
            "decision_time": BASE + timedelta(seconds=es),
            "entry_start_time": BASE + timedelta(seconds=es),
            "exit_end_time": BASE + timedelta(seconds=xs),
            "direction": d, "net_pnl": float(npnl), "displacement_bps": float(disp),
        })
    return pl.DataFrame(recs)


# ---------------------------------------------------------------------------
# dev / holdout split
# ---------------------------------------------------------------------------

def test_split_boundary_and_end_are_causal():
    """A trade exactly on the 2024-02-25 boundary belongs to the HOLDOUT (dev is strictly
    before). Trades at/after the train end are excluded from both."""
    full = pl.DataFrame({"decision_time": [
        HOLDOUT_START - timedelta(seconds=1),  # dev
        HOLDOUT_START,                          # holdout (boundary is inclusive on the right)
        TRAIN_END - timedelta(seconds=1),       # holdout
        TRAIN_END,                              # excluded (>= train end)
    ]})
    dev, holdout = split_dev_holdout(full)
    assert len(dev) == 1
    assert len(holdout) == 2
    assert dev["decision_time"].to_list() == [HOLDOUT_START - timedelta(seconds=1)]


def test_split_partitions_all_in_window_trades():
    full = pl.DataFrame({"decision_time": [
        HOLDOUT_START - timedelta(days=30), HOLDOUT_START - timedelta(days=1),
        HOLDOUT_START + timedelta(days=1), HOLDOUT_START + timedelta(days=30),
    ]})
    dev, holdout = split_dev_holdout(full)
    assert len(dev) + len(holdout) == len(full)  # nothing lost, nothing double-counted


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train CSV not generated")
def test_real_train_split_counts_and_no_oos():
    """On the actual train trade set the dev/holdout split must be 347/156 (the 156 matches
    research.md F7's within-train holdout n), and NOTHING may fall on/after the train end
    (2024-06-25) -- i.e. the reserved OOS window is never included."""
    full = load(TRAIN_CSV)  # keep==True, tz-parsed
    dev, holdout = split_dev_holdout(full)
    assert len(dev) == 347
    assert len(holdout) == 156
    assert full.filter(pl.col("decision_time") >= TRAIN_END).height == 0


# ---------------------------------------------------------------------------
# concentration helpers (share guard + realized-DD-day locator)
# ---------------------------------------------------------------------------

def test_pnl_share_guard():
    assert _pnl_share(25.0, 100.0) == 25.0
    assert _pnl_share(50.0, 0.0) is None       # zero total -> not interpretable
    assert _pnl_share(50.0, -100.0) is None    # negative total -> not interpretable
    assert _pnl_share(-10.0, 100.0) == -10.0   # a losing top day is a valid negative share


def test_realized_dd_day_locates_the_trough():
    """The drawdown-driving day is where the cumulative-P&L curve troughs -- NOT necessarily the
    top-P&L day. Here day 1 makes +100, day 2 loses -300 (the trough), day 3 recovers +50."""
    rows = [
        (0, 300, 1, 100.0, 1),                         # 2024-01-01: +100 (top P&L day)
        (24 * 3600, 24 * 3600 + 300, 1, -300.0, 1),    # 2024-01-02: -300 (DD trough)
        (48 * 3600, 48 * 3600 + 300, 1, 50.0, 1),      # 2024-01-03: +50
    ]
    sub = _trades(rows)
    assert _realized_dd_day(sub) == "2024-01-02"


# ---------------------------------------------------------------------------
# causality of the cap (no look-ahead)
# ---------------------------------------------------------------------------

def test_cap_decisions_do_not_depend_on_future_trades():
    """The greedy cap must be causal: appending trades that arrive LATER cannot change which
    of the earlier trades were taken. This is the deployable / no-look-ahead guarantee."""
    early = _trades([(0, 10, 1, 1, 1), (5, 15, 1, 1, 1), (6, 16, 1, 1, 1), (30, 40, 1, 1, 1)])
    future = _trades([(0, 10, 1, 1, 1), (5, 15, 1, 1, 1), (6, 16, 1, 1, 1), (30, 40, 1, 1, 1),
                      (31, 41, 1, 1, 1), (32, 42, 1, 1, 1), (33, 43, 1, 1, 1)])
    for n in (1, 2, 3):
        taken_early = set(cap_at_n(early, n)["entry_start_time"].to_list())
        taken_future = set(cap_at_n(future, n)["entry_start_time"].to_list())
        early_starts = set(early["entry_start_time"].to_list())
        # decisions on the ORIGINAL (early) trades are identical with or without the future ones
        assert taken_early == (taken_future & early_starts)


def test_cap_monotone_in_n_and_peak_respected():
    dense = _trades([(i, 100, 1, 1, 1) for i in range(8)])  # 8 all open together
    counts = {n: len(cap_at_n(dense, n)) for n in (1, 3, 5)}
    assert counts[5] >= counts[3] >= counts[1]
    for n in (1, 3, 5):
        assert concurrency_profile(cap_at_n(dense, n))["peak"] <= n
    assert concurrency_profile(cap_at_n(dense, 1))["peak"] == 1


def test_policies_labels_and_membership():
    df = _trades([(i, 50, 1, 1, 1) for i in range(6)])
    pol = policies(df)
    labels = [lbl for lbl, _ in pol]
    assert labels[0].startswith("uncapped")
    assert any("N=1" in l and "no concurrency" in l for l in labels)
    # exactly uncapped + one policy per configured cap, nothing else
    assert len(pol) == 1 + len(CAPS)
    # the uncapped baseline keeps everything; each cap keeps <= uncapped
    n_uncapped = len(pol[0][1])
    assert n_uncapped == len(df)
    for lbl, sub in pol[1:]:
        assert len(sub) <= n_uncapped


def test_driver_uses_no_lookahead_oracle():
    """The central no-look-ahead promise: the driver must NOT reference the most-violent-per-
    episode oracle (`dedupe_per_episode`) anywhere. Guards against a future edit re-introducing
    look-ahead selection."""
    src = Path(ctc.__file__).read_text()
    # the name may appear in prose explaining it is EXCLUDED; what must never appear is a call
    # or an import of it (that would be actual look-ahead selection).
    assert "dedupe_per_episode(" not in src
    assert "import dedupe_per_episode" not in src and ", dedupe_per_episode" not in src
    assert not hasattr(ctc, "dedupe_per_episode")  # not pulled into the module namespace


# ---------------------------------------------------------------------------
# cap selection rule (dev-only)
# ---------------------------------------------------------------------------

def _row(policy, net, t_nw, sharpe, mtm):
    return {"policy": policy, "net_bps": net, "t_nw": t_nw, "ann_sharpe": sharpe,
            "mtm_dd_usd": mtm}


def test_select_prefers_higher_sharpe_among_significant():
    rows = [
        _row("uncapped", 50, 4.0, 3.0, -20000),
        _row("cap N=5", 40, 3.0, 3.5, -8000),   # highest Sharpe & significant -> winner
        _row("cap N=3", 35, 2.5, 3.2, -6000),
        _row("cap N=1", 30, 1.2, 4.5, -2000),   # highest Sharpe but t_NW < 2 -> excluded
    ]
    sel = select_cap(rows)
    assert sel["policy"] == "cap N=5"
    assert sel["gated"] is True


def test_select_tie_breaks_on_smaller_drawdown():
    rows = [
        _row("uncapped", 50, 4.0, 3.0, -20000),
        _row("cap N=3", 40, 3.0, 3.0, -6000),   # same Sharpe, smaller |DD| -> winner
    ]
    sel = select_cap(rows)
    assert sel["policy"] == "cap N=3"


def test_select_falls_back_when_none_significant():
    rows = [
        _row("uncapped", 5, 1.0, 1.0, -20000),
        _row("cap N=3", 4, 0.5, 1.5, -6000),    # best Sharpe in the ungated fallback
    ]
    sel = select_cap(rows)
    assert sel["policy"] == "cap N=3"
    assert sel["gated"] is False


# ---------------------------------------------------------------------------
# cost decomposition additivity (reused from pnl_decomposition)
# ---------------------------------------------------------------------------

def _decomp_frame():
    """One long and one short round trip with explicit mids/vwaps so the components are
    hand-checkable. `net_pnl` is set to the dollar value the mid/vwap path implies (long leg
    790 bps of $50k = $3,950; short leg 365 bps = $1,825) so net_reconstructed reconciles to
    net_realized. Columns match what add_decomposition reads."""
    return pl.DataFrame({
        "direction": [1, -1],
        "decision": [100.0, 200.0],       # mid at decision (denominator)
        "entry": [100.5, 199.0],          # mid at entry
        "exit": [110.0, 190.0],           # mid at exit
        "entry_vwap": [101.0, 198.5],     # fill worse than mid on both entries
        "exit_vwap": [109.0, 191.0],      # fill worse than mid on both exits
        "net_pnl": [3950.0, 1825.0],      # $ P&L consistent with the linear mid/vwap path
        "entry_quantity": [500.0, -500.0],
    })


def test_components_sum_to_reconstructed():
    """The 6 additive components must reconstruct net exactly (validates the with_columns
    wiring, independent of any inverse-contract approximation)."""
    out = pnld.add_decomposition(_decomp_frame())
    recon = (out["gross_mid_to_mid"] - out["latency_slippage"] - out["spread_entry"]
             - out["spread_exit"] - out["fee_entry"] - out["fee_exit"])
    assert np.allclose(recon.to_numpy(), out["net_reconstructed"].to_numpy(), atol=1e-9)


def test_component_values_match_hand_computation_long_leg():
    """Long leg: gross_mid = 1e4*(exit-decision)/decision; spread terms positive (fills worse
    than mid); fees = 5 bps each."""
    out = pnld.add_decomposition(_decomp_frame())
    r = out.row(0, named=True)
    assert r["gross_mid_to_mid"] == pytest.approx(1e4 * (110.0 - 100.0) / 100.0)   # +1000 bps
    assert r["latency_slippage"] == pytest.approx(1e4 * (100.5 - 100.0) / 100.0)   # +50 bps
    assert r["spread_entry"] == pytest.approx(1e4 * (101.0 - 100.5) / 100.0)       # +50 bps
    assert r["spread_exit"] == pytest.approx(1e4 * (110.0 - 109.0) / 100.0)        # +100 bps
    assert r["fee_entry"] == pytest.approx(5.0) and r["fee_exit"] == pytest.approx(5.0)
    # net = 1000 - 50 - 50 - 100 - 5 - 5 = 790 bps
    assert r["net_reconstructed"] == pytest.approx(790.0)


def test_component_values_match_hand_computation_short_leg():
    """Short leg (direction = -1): a falling mid is a GAIN; the sign flip is handled by d."""
    out = pnld.add_decomposition(_decomp_frame())
    r = out.row(1, named=True)
    # gross = 1e4 * (-1) * (190 - 200) / 200 = +500 bps
    assert r["gross_mid_to_mid"] == pytest.approx(1e4 * -1 * (190.0 - 200.0) / 200.0)
    # spread_entry = 1e4 * -1 * (entry_vwap - entry)/denom = -1*(198.5-199.0)/200 -> +25 bps
    assert r["spread_entry"] == pytest.approx(1e4 * -1 * (198.5 - 199.0) / 200.0)
    assert r["spread_exit"] == pytest.approx(1e4 * -1 * (190.0 - 191.0) / 200.0)
    assert r["net_reconstructed"] == pytest.approx(365.0)


def test_reconstruction_reconciles_to_realized():
    """The MEANINGFUL decomposition check (beyond internal additivity): the mid/vwap-path
    reconstruction must match the realized net (bps of traded notional) when net_pnl is
    consistent with that path. Guards the net_realized wiring the runtime `gap` print relies on."""
    out = pnld.add_decomposition(_decomp_frame())
    assert np.allclose(out["net_reconstructed"].to_numpy(),
                       out["net_realized"].to_numpy(), atol=1e-9)
