"""
Tests for the two new mechanisms in the OOS risk audit:

  1. `participation` cap in `_sweep_fill_from_agg_trades` (backtester.py): the cap must
     be an exact no-op at 1.0, must degrade the VWAP when it forces consumption of worse
     prints, and must leave a fill incomplete when the tape inside the window is too thin.

  2. `mtm_max_drawdown` (strategies/backtest_oos_risk.py): marking concurrently-open,
     simultaneously-underwater positions to the L1 mid must produce a strictly deeper
     drawdown than the realized-close cumsum, and the curve must end at the realized total.

Run from the repo root under the polars env:
    ~/miniforge3/envs/mscf/bin/python -m pytest tests/test_oos_risk.py -q
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

import backtester as bt
from backtester import BookView, _sweep_fill_from_agg_trades
from backtest_oos_risk import mtm_max_drawdown, realized_close_dd

UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _tape(offsets_s, prices, quantities, is_buyer_maker):
    """Build the parallel arrays a sweep consumes, from second-offsets off BASE."""
    times = [BASE + timedelta(seconds=s) for s in offsets_s]
    times_ns = np.array([int(t.timestamp() * 1e9) for t in times], dtype=np.int64)
    return dict(
        trade_times_ns=times_ns,
        trade_times=times,
        prices=np.array(prices, dtype=float),
        quantities=np.array(quantities, dtype=float),
        is_buyer_maker=np.array(is_buyer_maker, dtype=bool),
    )


# ---------------------------------------------------------------------------
# 1. participation cap
# ---------------------------------------------------------------------------

# A taker BUY (signed>0) matches taker-buy prints (is_buyer_maker == False). Prices rise
# over time, so being forced to consume later prints is adverse (a worse VWAP).
BUY_TAPE = _tape(
    offsets_s=[1, 2, 3, 4, 5],
    prices=[100.0, 101.0, 102.0, 103.0, 104.0],
    quantities=[10.0, 10.0, 10.0, 10.0, 10.0],
    is_buyer_maker=[False, False, False, False, False],
)
START = BASE  # sweep consumes everything at/after BASE


def test_participation_1_is_exact_noop():
    """participation=1.0 must reproduce the uncapped fill byte-for-byte."""
    default = _sweep_fill_from_agg_trades(15.0, START, **BUY_TAPE)
    explicit = _sweep_fill_from_agg_trades(15.0, START, **BUY_TAPE, participation=1.0)
    assert default.filled_quantity == explicit.filled_quantity == 15.0
    assert default.avg_price == explicit.avg_price
    # 10@100 + 5@101 = 1505 / 15
    assert explicit.avg_price == pytest.approx((10 * 100 + 5 * 101) / 15)
    assert explicit.is_complete


def test_participation_cap_degrades_vwap():
    """A cap forces consumption of later, worse-priced prints -> strictly worse VWAP."""
    full = _sweep_fill_from_agg_trades(15.0, START, **BUY_TAPE, participation=1.0)
    capped = _sweep_fill_from_agg_trades(15.0, START, **BUY_TAPE, participation=0.5)
    # p=0.5 -> 5 per print: 5@100 + 5@101 + 5@102 = 1515 / 15 = 101.0
    assert capped.filled_quantity == 15.0
    assert capped.is_complete
    assert capped.avg_price == pytest.approx((5 * 100 + 5 * 101 + 5 * 102) / 15)
    assert capped.avg_price > full.avg_price  # a buyer paying more is worse


def test_participation_cap_leaves_fill_incomplete():
    """Too thin a tape inside the window -> a partial (missed-size) fill, not a free entry."""
    # p=0.1 -> 1 contract per print; 5 prints => only 5 of 15 filled.
    res = _sweep_fill_from_agg_trades(15.0, START, **BUY_TAPE, participation=0.1)
    assert res.filled_quantity == pytest.approx(5.0)
    assert not res.is_complete
    # what did fill still fills at the honest VWAP of the 5 earliest prints
    assert res.avg_price == pytest.approx((100 + 101 + 102 + 103 + 104) / 5)


def test_participation_cap_symmetric_for_sell():
    """A taker SELL matches taker-sell prints; falling prices are the adverse direction."""
    sell_tape = _tape(
        offsets_s=[1, 2, 3],
        prices=[100.0, 99.0, 98.0],
        quantities=[10.0, 10.0, 10.0],
        is_buyer_maker=[True, True, True],  # taker sells (buyer is maker)
    )
    full = _sweep_fill_from_agg_trades(-15.0, START, **sell_tape, participation=1.0)
    capped = _sweep_fill_from_agg_trades(-15.0, START, **sell_tape, participation=0.5)
    assert full.avg_price == pytest.approx((10 * 100 + 5 * 99) / 15)
    assert capped.avg_price == pytest.approx((5 * 100 + 5 * 99 + 5 * 98) / 15)
    assert capped.avg_price < full.avg_price  # a seller receiving less is worse


# ---------------------------------------------------------------------------
# 2. mark-to-market drawdown
# ---------------------------------------------------------------------------

class _FakeBook:
    """BookProvider-compatible stand-in over one in-memory BookView (as_of + window)."""

    def __init__(self, view: BookView) -> None:
        self.view = view

    def as_of(self, t):
        return self.view.as_of(t)

    def window(self, start, end):
        return self.view.slice_window(start, end)


def _flat_book_with_dip(dip_mid: float, dip_lo_s: int, dip_hi_s: int,
                        span_s: int = 120, base_mid: float = 100.0) -> _FakeBook:
    rows = []
    for s in range(span_s + 1):
        mid = dip_mid if dip_lo_s <= s <= dip_hi_s else base_mid
        rows.append({
            "event_time_datetime": BASE + timedelta(seconds=s),
            "best_bid_price": mid - 0.5, "best_bid_qty": 1.0,
            "best_ask_price": mid + 0.5, "best_ask_qty": 1.0,
            "mid_price": mid, "micro_price": mid,
        })
    # BookView.as_of/slice_window query in nanoseconds (via _datetime_to_ns), so the
    # frame's datetime column must be ns precision -- matching the real bookTicker parquet.
    frame = pl.DataFrame(rows).with_columns(
        pl.col("event_time_datetime").cast(pl.Datetime("ns", "UTC"))
    )
    return _FakeBook(BookView.from_frame(frame))


def _two_overlapping_longs() -> pl.DataFrame:
    """Two +10-contract reversion longs entered ~flat at 100, both held across a dip to
    90, both exiting ~flat. Each round-trip nets ~0 (minus fees), so the realized-close
    curve barely moves -- but both are -$100 underwater at the mid=90 trough at once."""
    fee = abs(10) * bt.CONTRACT_NOTIONAL_USD * bt.TAKER_FEE_RATE  # 0.5 per leg
    def trade(dt_s, ee_s, xe_s):
        return {
            "decision_time": BASE + timedelta(seconds=dt_s),
            "entry_start_time": BASE + timedelta(seconds=dt_s),
            "entry_end_time": BASE + timedelta(seconds=ee_s),
            "exit_start_time": BASE + timedelta(seconds=xe_s - 1),
            "exit_end_time": BASE + timedelta(seconds=xe_s),
            "entry_vwap": 100.0, "exit_vwap": 100.0,
            "entry_quantity": 10.0, "exit_quantity": -10.0,
            "gross_pnl": 0.0, "net_pnl": -2 * fee,  # 0 gross - entry_fee - exit_fee
        }
    return pl.DataFrame([trade(9, 10, 100), trade(11, 12, 102)])


def test_mtm_dd_strictly_worse_than_realized_when_positions_overlap_underwater():
    trades = _two_overlapping_longs()
    book = _flat_book_with_dip(dip_mid=90.0, dip_lo_s=50, dip_hi_s=52)

    realized_dd = realized_close_dd(trades)          # ~ -2 * fee summed = -2.0
    dd = mtm_max_drawdown(trades, book)

    total_net = float(trades["net_pnl"].sum())
    # Curve integrity: once everything is closed the equity equals the realized total.
    assert dd["final_equity_usd"] == pytest.approx(total_net, abs=1e-6)
    # Both longs -$100 at mid=90 => ~-$200 unrealized, plus entry fees already paid.
    assert dd["max_dd_usd"] == pytest.approx(-201.0, abs=1.0)
    # The whole point: MtM DD is far deeper than the realized-close DD.
    assert dd["max_dd_usd"] < realized_dd
    assert dd["max_dd_usd"] < -100.0 and realized_dd > -5.0


def test_mtm_dd_no_swing_when_mid_never_leaves_entry():
    """If the mid never moves off entry_vwap, there is no unrealized swing; the MtM
    drawdown collapses to just the realized fee bleed (a guard against phantom DD)."""
    trades = _two_overlapping_longs()
    book = _flat_book_with_dip(dip_mid=100.0, dip_lo_s=50, dip_hi_s=52)  # no dip
    dd = mtm_max_drawdown(trades, book)
    assert dd["max_dd_usd"] == pytest.approx(realized_close_dd(trades), abs=1.0)
    assert dd["max_dd_usd"] > -2.5  # only fees, no price swing


# --- flexible fixtures for the coverage cases the review asked for -------------------

def _book_with_mids(mids_by_s: dict[int, float], span_s: int = 200,
                    base_mid: float = 100.0, start: datetime = BASE) -> _FakeBook:
    """Book ticks every second; mid forward-filled from `mids_by_s` (second-offset -> mid)."""
    rows = []
    cur = base_mid
    for s in range(span_s + 1):
        if s in mids_by_s:
            cur = mids_by_s[s]
        rows.append({
            "event_time_datetime": start + timedelta(seconds=s),
            "best_bid_price": cur - 0.5, "best_bid_qty": 1.0,
            "best_ask_price": cur + 0.5, "best_ask_qty": 1.0,
            "mid_price": cur, "micro_price": cur,
        })
    frame = pl.DataFrame(rows).with_columns(
        pl.col("event_time_datetime").cast(pl.Datetime("ns", "UTC")))
    return _FakeBook(BookView.from_frame(frame))


def _trade(entry_qty, exit_qty, entry_vwap, exit_vwap, dt_s, ee_s, xe_s,
           start: datetime = BASE):
    closed = min(abs(entry_qty), abs(exit_qty))
    signed_closed = closed if entry_qty > 0 else -closed
    gross = signed_closed * bt.CONTRACT_NOTIONAL_USD * (exit_vwap / entry_vwap - 1.0)
    entry_fee = abs(entry_qty) * bt.CONTRACT_NOTIONAL_USD * bt.TAKER_FEE_RATE
    exit_fee = abs(exit_qty) * bt.CONTRACT_NOTIONAL_USD * bt.TAKER_FEE_RATE
    return {
        "decision_time": start + timedelta(seconds=dt_s),
        "entry_start_time": start + timedelta(seconds=dt_s),
        "entry_end_time": start + timedelta(seconds=ee_s),
        "exit_start_time": start + timedelta(seconds=xe_s - 1),
        "exit_end_time": start + timedelta(seconds=xe_s),
        "entry_vwap": float(entry_vwap), "exit_vwap": float(exit_vwap),
        "entry_quantity": float(entry_qty), "exit_quantity": float(exit_qty),
        "gross_pnl": float(gross), "net_pnl": float(gross - entry_fee - exit_fee),
    }


def test_mtm_short_position_sign():
    """A short (+entry_quantity<0) loses as the mid RISES; catches any abs()/sign slip."""
    trades = pl.DataFrame([_trade(-10, +10, 100.0, 100.0, 9, 10, 100)])
    book = _book_with_mids({50: 120.0, 53: 100.0})  # mid spikes up while short is open
    dd = mtm_max_drawdown(trades, book)
    assert dd["max_dd_usd"] == pytest.approx(-200.5, abs=1.0)  # -10*(120-100) - entry fee


def test_mtm_mixed_offset_nets_exposure():
    """Simultaneous +10 and -10 net to zero exposure; a mid dip must NOT create DD beyond
    the fee bleed (catches accumulation errors that don't cancel across legs)."""
    trades = pl.DataFrame([
        _trade(+10, -10, 100.0, 100.0, 9, 10, 100),
        _trade(-10, +10, 100.0, 100.0, 9, 11, 101),
    ])
    book = _book_with_mids({50: 90.0, 53: 100.0})  # big dip, but exposure is flat
    dd = mtm_max_drawdown(trades, book)
    assert dd["max_dd_usd"] == pytest.approx(-2.0, abs=0.6)  # ~4 fee legs, no price swing


def test_mtm_peak_then_trough_within_one_segment():
    """One open long, mid 100->130->70: DD is measured from the +300 peak to the -300
    trough (=-600). This is the case that actually exercises np.maximum.accumulate; a
    naive (arr - global_running_peak) without accumulate would still pass the basic tests
    but give the wrong answer here."""
    trades = pl.DataFrame([_trade(+10, -10, 100.0, 100.0, 9, 10, 100)])
    book = _book_with_mids({40: 130.0, 50: 70.0, 60: 100.0})
    dd = mtm_max_drawdown(trades, book)
    assert dd["max_dd_usd"] == pytest.approx(-600.0, abs=1.5)


def test_mtm_trough_before_higher_peak():
    """Mid 100->80->140->60 under one long: the max DD is from the 140 peak to the 60
    trough (=-800), NOT from the initial 100. Proves the carried running peak is right."""
    trades = pl.DataFrame([_trade(+10, -10, 100.0, 100.0, 9, 10, 100)])
    book = _book_with_mids({30: 80.0, 45: 140.0, 55: 60.0, 65: 100.0})
    dd = mtm_max_drawdown(trades, book)
    assert dd["max_dd_usd"] == pytest.approx(-800.0, abs=1.5)


def test_mtm_partial_exit_preserves_final_equity_invariant():
    """Under a participation cap the exit can fill less than the entry (entry 15, exit 10).
    The MtM curve must still end at sum(net_pnl): A/B remove the full entered size while
    realized books gross on the closed min only."""
    trades = pl.DataFrame([_trade(+15, -10, 100.0, 101.0, 9, 10, 100)])
    book = _book_with_mids({50: 95.0, 53: 101.0})
    dd = mtm_max_drawdown(trades, book)
    assert dd["final_equity_usd"] == pytest.approx(float(trades["net_pnl"].sum()), abs=1e-6)


def test_mtm_position_open_across_midnight():
    """A position held across a UTC midnight must mark correctly (timestamps cross a day
    boundary; ns math must not wrap)."""
    start = datetime(2024, 3, 1, 23, 58, 0, tzinfo=UTC)  # opens 2 min before midnight
    trades = pl.DataFrame([_trade(+10, -10, 100.0, 100.0, 0, 30, 300, start=start)])
    book = _book_with_mids({150: 85.0, 200: 100.0}, span_s=360, start=start)  # dip after midnight
    dd = mtm_max_drawdown(trades, book)
    assert dd["max_dd_usd"] == pytest.approx(-150.5, abs=1.0)  # -10*(100-85) - entry fee
