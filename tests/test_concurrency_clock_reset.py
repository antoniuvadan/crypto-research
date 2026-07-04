"""
Tests for the clock-reset bounded-cap driver (strategies/concurrency_clock_reset.py):
run chaining / build-to-N / clock extension, causality (no look-ahead), exit re-simulation
correctness against a hand computation, and the hold-time bookkeeping.

Run from the repo root under the polars env:
    ~/miniforge3/envs/mscf/bin/python -m pytest tests/test_concurrency_clock_reset.py -q
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

from backtester import _datetime_to_ns, DEFAULT_LATENCY
from concurrency_clock_reset import build_clock_reset_trades, hold_stats, FEE_RATE, CONTRACT

UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)
HOLD = timedelta(minutes=5)


def _events(rows):
    """rows: (offset_seconds, direction, entry_vwap, entry_quantity). One frozen entry per event;
    the builder reuses these and re-simulates only the exit."""
    recs = []
    for off, d, vwap, qty in rows:
        t = BASE + timedelta(seconds=off)
        recs.append({
            "decision_time": t, "liquidation_time": t - timedelta(seconds=5),
            "direction": d, "entry_start_time": t + DEFAULT_LATENCY,
            "entry_end_time": t + DEFAULT_LATENCY + timedelta(milliseconds=50),
            "entry_quantity": float(qty), "entry_vwap": float(vwap), "displacement_bps": 50.0,
        })
    return pl.DataFrame(recs)


def _tape(prints):
    """prints: list of (offset_seconds, price, qty, is_buyer_maker). Builds the 5-tuple the sweep
    consumes, with ns keys via _datetime_to_ns (self-consistent with the sweep's start-time key)."""
    prints = sorted(prints, key=lambda p: p[0])
    times = [BASE + timedelta(seconds=p[0]) for p in prints]
    return (
        np.array([_datetime_to_ns(t) for t in times], dtype=np.int64),
        times,
        np.array([p[1] for p in prints], dtype=float),
        np.array([p[2] for p in prints], dtype=float),
        np.array([p[3] for p in prints], dtype=bool),
    )


def _dense_exit_tape(price=100.0, is_buyer_maker=True, span_s=3 * 3600, step_s=10, qty=1e6):
    """Abundant same-type liquidity every step_s over the range, so every exit fills. For a LONG
    book the exit is a SELL (matches taker-sell prints is_buyer_maker=True)."""
    return _tape([(t, price, qty, is_buyer_maker) for t in range(0, span_s, step_s)])


# ---------------------------------------------------------------------------
# run chaining / gaps / build-to-N / clock extension
# ---------------------------------------------------------------------------

def test_single_event_run_is_a_plain_5min_trade():
    ev = _events([(0, 1, 100.0, 500.0)])
    out = build_clock_reset_trades(ev, n=5, tape=_dense_exit_tape(price=110.0))
    assert len(out) == 1
    r = out.row(0, named=True)
    assert r["n_units"] == 1 and r["run_events"] == 1 and r["n_resets"] == 0
    assert r["hold_minutes"] == pytest.approx(5.0)  # exit 5min after the lone signal
    # exit clock = decision + 5min; exit sweep starts after latency
    assert r["exit_start_time"] == BASE + HOLD + DEFAULT_LATENCY


def test_signals_within_5min_chain_gap_splits():
    # events at 0s, 120s (gap 2min <=5min -> chain), 600s (gap 8min from 120s -> new run)
    ev = _events([(0, 1, 100.0, 500.0), (120, 1, 100.0, 500.0), (600, 1, 100.0, 500.0)])
    out = build_clock_reset_trades(ev, n=5, tape=_dense_exit_tape())
    runs = out.group_by("run_id").agg(pl.col("run_events").first(), pl.len().alias("units")).sort("run_id")
    assert runs.height == 2
    assert runs["run_events"].to_list() == [2, 1]   # first run chains two, second is a singleton
    assert runs["units"].to_list() == [2, 1]


def test_build_to_N_then_extend_only():
    # 6 events 60s apart -> all chain (gaps 1min); N=3 caps units at 3 but the clock keeps resetting
    ev = _events([(60 * i, 1, 100.0, 500.0) for i in range(6)])
    out = build_clock_reset_trades(ev, n=3, tape=_dense_exit_tape())
    assert out["run_id"].n_unique() == 1
    r = out.row(0, named=True)
    assert r["n_units"] == 3            # size capped at N
    assert len(out) == 3               # exactly N unit-rows emitted
    assert r["run_events"] == 6 and r["n_resets"] == 5
    # exit clock = last signal (300s) + 5min = 600s -> hold = 10min from the first signal
    assert r["hold_minutes"] == pytest.approx(10.0)
    assert out["exit_start_time"].unique().to_list() == [BASE + timedelta(seconds=600) + DEFAULT_LATENCY]


def test_directions_form_independent_runs():
    ev = _events([(0, 1, 100.0, 500.0), (30, -1, 100.0, 500.0), (60, 1, 100.0, 500.0)])
    # long book exits SELL (is_buyer_maker True); short book exits BUY (is_buyer_maker False)
    tape = _tape([(t, 100.0, 1e6, bm) for t in range(0, 3600, 10) for bm in (True, False)])
    out = build_clock_reset_trades(ev, n=5, tape=tape)
    longs = out.filter(pl.col("direction") == 1)
    shorts = out.filter(pl.col("direction") == -1)
    assert longs["run_id"].n_unique() == 1 and shorts["run_id"].n_unique() == 1
    assert len(longs) == 2 and len(shorts) == 1  # the two longs chain, the short is its own run


# ---------------------------------------------------------------------------
# causality (no look-ahead)
# ---------------------------------------------------------------------------

def test_appending_future_events_does_not_change_earlier_runs():
    early = _events([(0, 1, 100.0, 500.0), (120, 1, 100.0, 500.0)])
    future = _events([(0, 1, 100.0, 500.0), (120, 1, 100.0, 500.0),
                      (5000, 1, 100.0, 500.0), (5100, 1, 100.0, 500.0)])
    tape = _dense_exit_tape(span_s=3 * 3600 + 6000)
    a = build_clock_reset_trades(early, n=5, tape=tape).sort("decision_time")
    b = build_clock_reset_trades(future, n=5, tape=tape).sort("decision_time")
    # the first run's rows (decision_time 0 and 120s) are byte-identical with or without the future
    cols = ["decision_time", "n_units", "run_events", "hold_minutes", "exit_start_time", "net_pnl"]
    a_first = a.head(2).select(cols)
    b_first = b.filter(pl.col("decision_time") <= BASE + timedelta(seconds=120)).select(cols)
    assert a_first.equals(b_first)


# ---------------------------------------------------------------------------
# exit re-simulation correctness (hand computation)
# ---------------------------------------------------------------------------

def test_exit_pnl_matches_hand_computation_long():
    # single long unit, entry 100, 500 contracts; exit tape prints 110 -> exit_vwap 110
    ev = _events([(0, 1, 100.0, 500.0)])
    out = build_clock_reset_trades(ev, n=1, tape=_dense_exit_tape(price=110.0))
    r = out.row(0, named=True)
    assert r["exit_vwap"] == pytest.approx(110.0)
    # gross = 500 * 100 * (110/100 - 1) = 5000; fees = (500+500)*100*5bps = 50; net = 4950
    assert r["gross_pnl"] == pytest.approx(5000.0)
    assert r["fees_paid"] == pytest.approx(50.0)
    assert r["net_pnl"] == pytest.approx(4950.0)
    assert r["exit_quantity"] == pytest.approx(-500.0)


def test_aggregated_exit_vwap_walks_the_tape():
    # two long units (500 + 500 = 1000 contracts) share one exit sweep; the tape offers 600 @110
    # then 600 @112, so a 1000-lot sell VWAPs (600*110 + 400*112)/1000 = 110.8 for BOTH units.
    ev = _events([(0, 1, 100.0, 500.0), (60, 1, 100.0, 500.0)])
    exit_ns = 60 + 300  # last signal 60s + 5min = 360s; +latency
    off = 361
    tape = _tape([(off, 110.0, 600.0, True), (off + 1, 112.0, 600.0, True)])
    out = build_clock_reset_trades(ev, n=5, tape=tape)
    assert len(out) == 2
    vwaps = out["exit_vwap"].unique().to_list()
    assert len(vwaps) == 1 and vwaps[0] == pytest.approx(110.8)


def test_aggregated_exit_distinct_entry_vwaps_give_distinct_gross():
    """Two units with DIFFERENT entry prices share one exit VWAP -> different per-unit gross
    (the shared-exit accounting must key gross off each unit's own entry)."""
    ev = _events([(0, 1, 100.0, 500.0), (60, 1, 105.0, 500.0)])
    out = build_clock_reset_trades(ev, n=5, tape=_dense_exit_tape(price=110.0)).sort("entry_vwap")
    assert out["exit_vwap"].unique().to_list() == [pytest.approx(110.0)]
    g = out["gross_pnl"].to_list()
    assert g[0] == pytest.approx(500 * 100 * (110.0 / 100.0 - 1.0))   # entry 100 -> +5000
    assert g[1] == pytest.approx(500 * 100 * (110.0 / 105.0 - 1.0))   # entry 105 -> ~+2381
    assert g[0] != pytest.approx(g[1])


def test_partial_exit_fill_scales_pro_rata():
    """Thin exit window: a 1000-lot book meets only 600 lots of same-side liquidity -> the exit
    is incomplete and each unit closes pro-rata (fill_ratio=0.6), with fees scaled to what closed."""
    ev = _events([(0, 1, 100.0, 500.0), (30, 1, 100.0, 500.0)])  # book_qty = 1000
    exit_off = 30 + 300  # last signal 30s + 5min = 330s; sweep starts after latency
    tape = _tape([(exit_off + 1, 110.0, 600.0, True)])  # only 600 lots available to sell into
    out = build_clock_reset_trades(ev, n=5, tape=tape)
    assert len(out) == 2
    for r in out.iter_rows(named=True):
        assert r["exit_complete"] is False
        assert r["exit_quantity"] == pytest.approx(-300.0)          # 500 * 0.6
        assert r["gross_pnl"] == pytest.approx(300 * 100 * (110.0 / 100.0 - 1.0))  # 300 closed
        assert r["fees_paid"] == pytest.approx((500 + 300) * 100 * FEE_RATE)       # fees scale


def test_run_with_no_exit_liquidity_is_dropped_others_survive():
    """A run whose exit finds no same-side liquidity is dropped (no phantom fill); unrelated runs
    are unaffected. Also exercises the empty-tail guard in the builder."""
    ev = _events([(0, 1, 100.0, 500.0), (10000, 1, 100.0, 500.0)])  # two far-apart long runs
    # sell-side liquidity only around the FIRST run's exit (~305s); nothing near the second (~10305s)
    tape = _tape([(t, 100.0, 1e6, True) for t in range(300, 400, 5)])
    out = build_clock_reset_trades(ev, n=5, tape=tape)
    assert len(out) == 1
    assert out["decision_time"].to_list() == [BASE]  # only the first run survived


def test_short_leg_pnl():
    # single short unit: entry 200, -500 contracts; exit BUY fills at 190 -> gain.
    ev = _events([(0, -1, 200.0, -500.0)])
    tape = _dense_exit_tape(price=190.0, is_buyer_maker=False)  # BUY exit matches taker-buy prints
    out = build_clock_reset_trades(ev, n=1, tape=tape)
    r = out.row(0, named=True)
    # gross = -500 * 100 * (190/200 - 1) = -500*100*(-0.05) = +2500
    assert r["gross_pnl"] == pytest.approx(2500.0)
    assert r["exit_quantity"] == pytest.approx(500.0)  # buy to close a short


# ---------------------------------------------------------------------------
# hold-time diagnostic bookkeeping
# ---------------------------------------------------------------------------

def test_hold_stats():
    ev = _events([(0, 1, 100.0, 500.0), (120, 1, 100.0, 500.0), (600, 1, 100.0, 500.0)])
    out = build_clock_reset_trades(ev, n=5, tape=_dense_exit_tape())
    hs = hold_stats(out)
    assert hs["runs"] == 2
    assert hs["multi_event_run_%"] == pytest.approx(50.0)  # one 2-event run, one singleton
    # run1 hold = (120+300-0)/60 = 7min; run2 hold = 5min; median of {7,5} = 6
    assert hs["median_hold_min"] == pytest.approx(6.0)
    assert hs["max_hold_min"] == pytest.approx(7.0)
