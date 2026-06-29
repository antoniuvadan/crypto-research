#!/usr/bin/env python3
"""
Maker-entry fill model for the 5-min tail reversion (Finding 7 successor).

Findings 1-4 establish that the 10 bps round-trip *taker* fee is the binding
friction on the +23.7 bps net edge; Finding 5 shows we *want* to hold (no hurry to
get filled). That is exactly where a passive (maker) entry is worth modelling: post
a resting limit at the touch instead of crossing the spread, pay 2 bps instead of 5
on the entry leg -- but only fill when the tape comes to you, and miss the events
where price reverts away from your limit immediately.

This driver simulates, per >=98th tail event over the full train window:
  * a TAKER entry (existing `_sweep_fill_from_agg_trades`) -> reproduces the
    Finding-7 baseline (fills ~every event);
  * a MAKER entry (`_maker_fill_from_agg_trades`, strict-cross queue assumption):
    rest a bid (long reversion) / ask (short) at the decision-time touch; fill only
    if a taker prints through it before the 5-min exit. No-fills are dropped.
Both then exit with a taker sweep at decision + 5min + latency (maker-in/taker-out).

The honest comparison credits the missed maker trades at zero P&L (you posted, did
not fill, held no position): the headline `net_bps (all)` is the mean over *every
attempted event*, so the entry-fee saving and the forfeited winners sit on one
ledger. `net_bps (filled)` is the diagnostic over fills only.

Fee modes reported (fills are fee-independent, so computed off the same sim):
  taker/taker (baseline)   5 + 5 bps   -- reproduces Finding 7
  maker-in / taker-out     2 + 5 bps   -- the realistic headline
  maker/maker (fee bound)  2 + 2 bps   -- optimistic bookend: credits a maker EXIT
                                          fee without modelling passive-exit fill
                                          risk (same taker-exit fills, 2 bps charged)

The causal trailing-median displacement trim filter (Finding 7, fixed from dev) is
applied on top; results are split DEV (2023-06-25 -> 2024-02-24) vs within-train
HOLDOUT (2024-02-25 -> 2024-06-24). The reserved test period is NOT touched.

Usage:
  python backtest_reversion_maker.py
"""

from __future__ import annotations

# This module lives in strategies/; put the repo root on sys.path so the shared
# engine (backtester, significance) resolves when run as a script from the repo root.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

import backtester as bt
import backtest_momentum as bm
from backtester import (
    BookProvider,
    _linear_contract_pnl_usd,
    _maker_fill_from_agg_trades,
    _sweep_fill_from_agg_trades,
)
from significance import hac_stats

START_DATE = date(2023, 6, 25)
END_DATE = date(2024, 6, 24)  # full train; threshold needs history across DEV_END
DEV_END = datetime(2024, 2, 25, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2024, 6, 25, tzinfo=timezone.utc)

HOLDING = timedelta(minutes=5)
LATENCY = bt.DEFAULT_LATENCY  # 300ms
TRADE_NOTIONAL_USD = 50_000.0
CONTRACT_NOTIONAL_USD = bt.CONTRACT_NOTIONAL_USD  # 100
TARGET_CONTRACTS = TRADE_NOTIONAL_USD / CONTRACT_NOTIONAL_USD
PERCENTILE = 0.98
SIGNAL_SIGN = -1  # reversion: trade against the liquidating flow
STRICT_CROSS = True  # conservative maker queue assumption

# Causal trim filter (Finding 7, fixed from dev) -- identical params to apply_trim_filter.
PRE_LOOKBACK = timedelta(seconds=5)
TRAIL_WINDOW = timedelta(days=30)
MIN_HISTORY = 15

MAKER = bt.MAKER_FEE_RATE  # 2 bps
TAKER = bt.TAKER_FEE_RATE  # 5 bps

OUT = Path("data/results/reversion_maker_trades.csv")


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def _net_bps(gross_pnl: float, entry_qty: float, exit_qty: float,
             entry_rate: float, exit_rate: float) -> float:
    """Net P&L in bps of *target* notional (common denominator across fills and
    misses, so partial fills are penalised and misses can be credited at zero)."""
    fees = abs(entry_qty) * CONTRACT_NOTIONAL_USD * entry_rate
    fees += abs(exit_qty) * CONTRACT_NOTIONAL_USD * exit_rate
    return 1e4 * (gross_pnl - fees) / TRADE_NOTIONAL_USD


def simulate() -> pl.DataFrame:
    _eprint("loading liq + agg over full train ...")
    liq = bt.load_liq_snap(start_date=START_DATE, end_date=END_DATE).collect().sort("time_datetime")
    agg = bt.load_agg_trades(start_date=START_DATE, end_date=END_DATE).collect().sort("transact_time_datetime")
    agg_qty = bm._same_direction_aggregate_quantities(liq_df=liq, trades_df=agg, seconds_before=5, seconds_after=5)
    liq = liq.with_columns(pl.Series("agg_qty_5s_before_5s_after", agg_qty))

    events = bm.LiquidationMomentumStrategy._build_signal_events(
        liq_snap=liq,
        agg_trades=agg,
        percentile=PERCENTILE,
        trailing_window=timedelta(days=7),
        seconds_before=5,
        seconds_after=5,
        aggregate_quantity_col="agg_qty_5s_before_5s_after",
        upper_percentile=None,
        progress_label=None,
    )
    _eprint(f"built {len(events)} >=98th tail events")

    trade_times_ns = agg["transact_time_datetime"].cast(pl.Int64).to_numpy()
    trade_times = agg["transact_time_datetime"].to_list()
    prices = agg["price"].to_numpy()
    quantities = agg["quantity"].to_numpy()
    is_buyer_maker = agg["is_buyer_maker"].to_numpy()

    book = BookProvider()
    rows: list[dict[str, object]] = []

    for i, event in enumerate(events, start=1):
        if i % 100 == 0:
            _eprint(f"  event {i}/{len(events)}")
        traded_direction = SIGNAL_SIGN * event.direction  # +1 long, -1 short
        decision = event.decision_time
        entry_start = decision + LATENCY
        cap = decision + HOLDING + LATENCY  # = exit_start (FixedHorizon)
        exit_start = cap

        # Decision-time book: limit price (touch) + displacement for the trim filter.
        at_entry = book.as_of(entry_start)
        at_decision = book.as_of(decision)
        pre = book.as_of(event.liquidation_time - PRE_LOOKBACK)
        if at_entry is None or at_decision is None or pre is None:
            continue
        disp_bps = traded_direction * 1e4 * (pre.mid_price - at_decision.mid_price) / at_decision.mid_price
        limit_price = at_entry.bid_price if traded_direction > 0 else at_entry.ask_price

        signed_entry = TARGET_CONTRACTS * traded_direction

        # --- TAKER entry (baseline reproduction) ---
        tk_entry = _sweep_fill_from_agg_trades(
            signed_entry, entry_start, trade_times_ns, trade_times,
            prices, quantities, is_buyer_maker, max_end_time=cap,
        )
        # --- MAKER entry (passive, may not fill) ---
        mk_entry = _maker_fill_from_agg_trades(
            signed_entry, limit_price, entry_start, trade_times_ns, trade_times,
            prices, quantities, is_buyer_maker, max_end_time=cap, strict_cross=STRICT_CROSS,
        )

        def close(entry_fill):
            if entry_fill.filled_quantity == 0 or entry_fill.avg_price is None:
                return 0.0, 0.0, 0.0  # entry_qty, exit_qty, gross_pnl
            ex = _sweep_fill_from_agg_trades(
                -entry_fill.filled_quantity, exit_start, trade_times_ns, trade_times,
                prices, quantities, is_buyer_maker,
            )
            if ex.filled_quantity == 0 or ex.avg_price is None:
                return 0.0, 0.0, 0.0
            closed = min(abs(entry_fill.filled_quantity), abs(ex.filled_quantity))
            signed_closed = closed if entry_fill.filled_quantity > 0 else -closed
            gross = _linear_contract_pnl_usd(signed_closed, entry_fill.avg_price, ex.avg_price, CONTRACT_NOTIONAL_USD)
            return entry_fill.filled_quantity, -signed_closed, gross

        tk_eq, tk_xq, tk_gross = close(tk_entry)
        mk_eq, mk_xq, mk_gross = close(mk_entry)

        rows.append({
            "decision_time": decision,
            "direction": traded_direction,
            "displacement_bps": disp_bps,
            "limit_price": limit_price,
            "tk_filled": tk_eq != 0.0,
            "tk_entry_qty": tk_eq, "tk_exit_qty": tk_xq, "tk_gross_pnl": tk_gross,
            "mk_filled": mk_eq != 0.0,
            "mk_complete": bool(mk_entry.is_complete) and mk_eq != 0.0,
            "mk_entry_qty": mk_eq, "mk_exit_qty": mk_xq, "mk_gross_pnl": mk_gross,
        })

    return pl.DataFrame(rows).sort("decision_time")


def apply_trim(df: pl.DataFrame) -> pl.DataFrame:
    """Causal trailing-median displacement keep mask over PAST events in [t-W, t)."""
    times_ns = df["decision_time"].cast(pl.Int64).to_numpy()
    d = df["displacement_bps"].to_numpy()
    w_ns = int(TRAIL_WINDOW.total_seconds() * 1e9)
    keep = np.zeros(len(d), dtype=bool)
    for i in range(len(d)):
        lo = int(np.searchsorted(times_ns, times_ns[i] - w_ns, side="left"))
        past = d[lo:i]
        keep[i] = True if len(past) < MIN_HISTORY else (d[i] > float(np.median(past)))
    return df.with_columns(keep=pl.Series(keep))


# (label, which entry fill, entry fee, exit fee)
MODES = [
    ("taker/taker (baseline)", "tk", TAKER, TAKER),
    ("maker-in / taker-out", "mk", MAKER, TAKER),
    ("maker/maker (fee bound)", "mk", MAKER, MAKER),
]


def _mode_series(df: pl.DataFrame, which: str, entry_rate: float, exit_rate: float) -> np.ndarray:
    """Per-event net bps, time-ordered, zero for events the entry never filled
    (you posted, did not fill -> no position -> 0 P&L). This IS the strategy's
    realised return series, so its mean and HAC t are the honest headline."""
    eq = df[f"{which}_entry_qty"].to_numpy()
    xq = df[f"{which}_exit_qty"].to_numpy()
    gross = df[f"{which}_gross_pnl"].to_numpy()
    filled = df[f"{which}_filled"].to_numpy()
    out = np.zeros(len(df))
    for i in range(len(df)):
        if filled[i]:
            out[i] = _net_bps(gross[i], eq[i], xq[i], entry_rate, exit_rate)
    return out


def report(df: pl.DataFrame, title: str) -> None:
    print(f"\n=== {title}  (n attempted = {len(df)}) ===")
    print(f"{'mode':26s} {'fill%':>6s} {'net_bps(filled)':>16s} {'net_bps(all)':>13s} {'t_NW(all)':>10s}")
    for label, which, er, xr in MODES:
        s = _mode_series(df, which, er, xr)
        filled = df[f"{which}_filled"].to_numpy()
        n_fill = int(filled.sum())
        fill_rate = n_fill / len(df) if len(df) else float("nan")
        on_filled = float(s[filled].mean()) if n_fill else float("nan")
        all_mean = float(s.mean())
        hac = hac_stats(s)
        print(f"{label:26s} {fill_rate*100:5.0f}% {on_filled:16.2f} {all_mean:13.2f} "
              f"{hac['t_nw']:9.2f}  (L={int(hac['nw_lag'])})")


def main() -> None:
    df = simulate()
    df = apply_trim(df)
    df = df.filter(pl.col("decision_time") < HOLDOUT_END)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT)
    _eprint(f"wrote {OUT}")

    dev = df.filter(pl.col("decision_time") < DEV_END)
    hold = df.filter(pl.col("decision_time") >= DEV_END)

    print("\nMaker-entry fill model -- 5-min tail reversion, $50k, strict-cross queue")
    print("net_bps(all) credits non-filled maker events at 0 P&L (the honest headline).")

    report(dev, "DEV 2023-06-25 -> 2024-02-24  (ALL events)")
    report(dev.filter(pl.col("keep")), "DEV  filtered (causal displacement trim)")
    report(hold, "HOLDOUT 2024-02-25 -> 2024-06-24  (ALL events)  [within-train OOS]")
    report(hold.filter(pl.col("keep")), "HOLDOUT  filtered (causal displacement trim)  [within-train OOS]")

    print(f"\nWrote {OUT}.  TEST PERIOD (2024-06-25 -> 2024-10-14) untouched.")


if __name__ == "__main__":
    main()
