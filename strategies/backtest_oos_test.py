#!/usr/bin/env python3
"""
TRUE out-of-sample test of the frozen candidate on the reserved test period.

Strategy (frozen at Finding 8, de-risked at Finding 9 -- NOTHING re-tuned here):
  - >=98th-pct cascade tail (trailing-7d threshold on same-direction +/-5s flow)
  - reversion (signal_direction_sign = -1), 5-min fixed hold, taker fills
  - causal trim: keep an event iff its decision-time displacement exceeds the
    trailing-30d median of past tail-event displacements (min_history 15)

Reserved TEST period: 2024-06-25 -> 2024-10-14 (never touched in F1-F9).

Warmup: data is loaded from 2024-04-15 so the trailing-7d event threshold and the
trailing-30d trim median are fully warmed for every test-period event (the trim
history of an early-test event legitimately includes late-train tail events, which
is the live/causal behaviour). Only events with decision_time in the test period are
reported.

One book pass per trade yields BOTH the displacement (for the trim filter) and the
mid-path decomposition (mid_decision, mid_entry, mid_exit) so the conservative
fill floor can be computed exactly, without a second pass.

Produces a very thorough report: per-trade edge (gross/net bps, IID + Newey-West HAC
SEs and t-stats), conservative-fill floor, long/short split, dollar P&L and ROI on a
peak-concurrent-capital base at $50k and $100k, per-trade and annualized Sharpe, max
drawdown, hit rate, and the trade distribution -- compared against the within-train
expectation (F7/F9) and the F4 regime prior.

Usage:
  python strategies/backtest_oos_test.py        # run from the repo root
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
from backtester import BookProvider
from significance import hac_stats

LOAD_START = date(2024, 4, 15)          # warmup buffer (>=30d trim + 7d threshold)
TEST_START = datetime(2024, 6, 25, tzinfo=timezone.utc)
TEST_END = datetime(2024, 10, 15, tzinfo=timezone.utc)  # inclusive of 2024-10-14
LOAD_END = date(2024, 10, 14)
DISP_FROM = datetime(2024, 5, 20, tzinfo=timezone.utc)   # compute mids from here on

HOLDING = timedelta(minutes=5)
LATENCY = bt.DEFAULT_LATENCY
PRE_LOOKBACK = timedelta(seconds=5)
TRAIL_WINDOW = timedelta(days=30)
MIN_HISTORY = 15
PERCENTILE = 0.98
CONTRACT = bt.CONTRACT_NOTIONAL_USD     # $100
ROUND_TRIP_FEE_BPS = 10.0

OUT = Path("data/results/oos_test_trades.csv")


def _eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def run_size(liq: pl.DataFrame, agg: pl.DataFrame, notional: float) -> pl.DataFrame:
    res = bm.run_liquidation_momentum_model_c_backtests(
        liq_snap=liq, agg_trades=agg,
        holding_periods=(HOLDING,),
        trade_notional_usd=notional,
        signal_direction_sign=-1, percentile=0.98, upper_percentile=None,
        summary_csv_path=None, trades_csv_path=None, show_progress=False,
    )
    return res["trades"].with_columns(
        pl.col("decision_time").str.to_datetime(time_zone="UTC"),
        pl.col("liquidation_time").str.to_datetime(time_zone="UTC"),
        pl.col("entry_start_time").str.to_datetime(time_zone="UTC"),
        pl.col("exit_start_time").str.to_datetime(time_zone="UTC"),
    ).sort("decision_time")


def add_book_features(df: pl.DataFrame) -> pl.DataFrame:
    """One causal book pass: displacement (for trim) + mid path (for the floor)."""
    book = BookProvider()
    n = len(df)
    disp = np.full(n, np.nan)
    gross_mid = np.full(n, np.nan)
    latency_slip = np.full(n, np.nan)
    for i, t in enumerate(df.iter_rows(named=True)):
        if i % 100 == 0:
            _eprint(f"  book features {i}/{n}")
        if t["decision_time"] < DISP_FROM:
            continue
        d = float(t["direction"])
        m_dec = book.as_of(t["decision_time"])
        m_pre = book.as_of(t["liquidation_time"] - PRE_LOOKBACK)
        m_ent = book.as_of(t["decision_time"] + LATENCY)
        m_exit = book.as_of(t["decision_time"] + HOLDING + LATENCY)
        if m_dec is None or m_pre is None:
            continue
        disp[i] = d * 1e4 * (m_pre.mid_price - m_dec.mid_price) / m_dec.mid_price
        if m_ent is not None:
            latency_slip[i] = d * 1e4 * (m_ent.mid_price / m_dec.mid_price - 1.0)
        if m_exit is not None:
            gross_mid[i] = d * 1e4 * (m_exit.mid_price / m_dec.mid_price - 1.0)
    return df.with_columns(
        displacement_bps=pl.Series(disp),
        gross_mid_bps=pl.Series(gross_mid),
        latency_slip_bps=pl.Series(latency_slip),
    )


def causal_keep(df: pl.DataFrame) -> np.ndarray:
    times_ns = df["decision_time"].cast(pl.Int64).to_numpy()
    d = df["displacement_bps"].to_numpy()
    w_ns = int(TRAIL_WINDOW.total_seconds() * 1e9)
    keep = np.zeros(len(d), dtype=bool)
    for i in range(len(d)):
        if np.isnan(d[i]):
            continue
        lo = int(np.searchsorted(times_ns, times_ns[i] - w_ns, side="left"))
        past = d[lo:i]
        past = past[~np.isnan(past)]
        keep[i] = True if len(past) < MIN_HISTORY else (d[i] > float(np.median(past)))
    return keep


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _net_bps(df: pl.DataFrame, notional: float) -> np.ndarray:
    return (1e4 * df["net_pnl"] / notional).to_numpy()


def _gross_bps(df: pl.DataFrame, notional: float) -> np.ndarray:
    return (1e4 * df["gross_pnl"] / notional).to_numpy()


def _hac_line(label: str, y: np.ndarray) -> None:
    if len(y) == 0:
        print(f"  {label:24s} n=0")
        return
    s = hac_stats(y)
    print(f"  {label:24s} n={int(s['n']):4d}  mean {s['mean_bps']:+7.2f} bps   "
          f"SE_iid {s['se_iid']:5.2f} (t {s['t_iid']:+5.2f})   "
          f"SE_NW {s['se_nw']:5.2f} (t {s['t_nw']:+5.2f}, L={int(s['nw_lag'])})")


def peak_capital(df: pl.DataFrame, notional: float) -> tuple[int, float]:
    """Peak simultaneously-open positions (sweep line over [entry, exit]) and the
    capital base = peak * per-trade notional."""
    if len(df) == 0:
        return 0, 0.0
    opens = df["entry_start_time"].to_numpy()
    closes = (df["decision_time"].dt.offset_by("5m").dt.offset_by(f"{int(LATENCY.total_seconds()*1000)}ms")).to_numpy()
    events = [(t, +1) for t in opens] + [(t, -1) for t in closes]
    events.sort(key=lambda e: (e[0], -e[1]))  # opens before closes at a tie
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak, peak * notional


def dollar_block(df: pl.DataFrame, notional: float, span_days: float) -> None:
    pnl = df["net_pnl"].to_numpy()
    gross = df["gross_pnl"].to_numpy()
    total = float(pnl.sum())
    pk, pk_cap = peak_capital(df, notional)
    roi = total / pk_cap if pk_cap else float("nan")
    ann = roi * 365.0 / span_days if span_days else float("nan")
    turnover = len(df) * notional
    print(f"\n  -- ${notional/1000:.0f}k / trade --")
    print(f"  trades                {len(df)}")
    print(f"  total NET P&L         ${total:,.0f}")
    print(f"  total GROSS P&L       ${float(gross.sum()):,.0f}   (fees ${float(gross.sum()-total):,.0f})")
    print(f"  mean net / trade      ${total/len(df):,.0f}")
    print(f"  peak concurrent pos   {pk}  ->  peak capital ${pk_cap:,.0f}")
    print(f"  ROI on peak capital   {roi*100:+.2f}%  over {span_days:.0f}d  "
          f"(~{ann*100:+.1f}%/yr, simple)")
    print(f"  gross notional turnover ${turnover:,.0f}")


def sharpe_block(df: pl.DataFrame, notional: float, net_bps: np.ndarray) -> None:
    # Per-trade Sharpe (unitless, per round-trip).
    pt = float(net_bps.mean() / net_bps.std(ddof=1)) if len(net_bps) > 1 else float("nan")
    # Annualized Sharpe from a daily realized-P&L series on the peak-capital base,
    # over ALL calendar days in the test span (idle days count as 0 return).
    pk, pk_cap = peak_capital(df, notional)
    closes = df["decision_time"].dt.offset_by("5m").dt.date()
    daily = (
        df.with_columns(close_date=closes)
        .group_by("close_date").agg(pl.col("net_pnl").sum())
        .sort("close_date")
    )
    all_days = pl.date_range(TEST_START.date(), (TEST_END - timedelta(days=1)).date(),
                             interval="1d", eager=True)
    cal = pl.DataFrame({"close_date": all_days}).join(daily, on="close_date", how="left").fill_null(0.0)
    dret = (cal["net_pnl"] / pk_cap).to_numpy() if pk_cap else np.array([0.0])
    ann = float(dret.mean() / dret.std(ddof=1) * np.sqrt(365)) if dret.std(ddof=1) > 0 else float("nan")
    active = int((cal["net_pnl"] != 0).sum())
    print(f"  per-trade Sharpe      {pt:+.3f}  (mean/std of per-trade net bps)")
    print(f"  annualized Sharpe     {ann:+.2f}  (daily P&L / peak cap, all {len(cal)} cal days, "
          f"{active} active; x sqrt365)")


def max_drawdown(df: pl.DataFrame) -> float:
    eq = np.cumsum(df.sort("decision_time")["net_pnl"].to_numpy())
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min()) if len(eq) else 0.0


def report(test_all: pl.DataFrame, notional_primary: float, test100: pl.DataFrame) -> None:
    kept = test_all.filter(pl.col("keep"))
    span_days = (TEST_END - TEST_START).days

    print("\n" + "#" * 80)
    print("# TRUE OUT-OF-SAMPLE TEST -- reserved period 2024-06-25 -> 2024-10-14")
    print("# Frozen strategy: trimmed 5-min >=98th taker tail reversion. NOTHING re-tuned.")
    print("#" * 80)

    print(f"\nSelection: {len(test_all)} >=98th tail events in the test period; "
          f"trim keeps {len(kept)} ({len(kept)/len(test_all)*100:.0f}%).")

    print("\n=== 1. PER-TRADE EDGE (bps of traded notional; IID + Newey-West HAC) ===")
    print("\n GROSS (fill-to-fill, before fees):")
    _hac_line("test all", _gross_bps(test_all, notional_primary))
    _hac_line("test FILTERED (strat)", _gross_bps(kept, notional_primary))
    print("\n NET (after 10 bps round-trip taker fee):")
    _hac_line("test all", _net_bps(test_all, notional_primary))
    _hac_line("test FILTERED (strat)", _net_bps(kept, notional_primary))
    print("\n NET by direction (filtered):")
    _hac_line("long", _net_bps(kept.filter(pl.col("direction") > 0), notional_primary))
    _hac_line("short", _net_bps(kept.filter(pl.col("direction") < 0), notional_primary))

    print("\n=== 2. CONSERVATIVE FILL FLOOR (filtered; substitute flat s bps/side taker) ===")
    print("   floor = gross_mid - latency_slip - 2s - 10   (F9 method, exact mids)")
    have = kept.filter(pl.col("gross_mid_bps").is_not_null() & pl.col("latency_slip_bps").is_not_null())
    for s in (2.0, 3.0, 4.0):
        floor = (have["gross_mid_bps"] - have["latency_slip_bps"] - 2 * s - ROUND_TRIP_FEE_BPS).to_numpy()
        st = hac_stats(floor)
        print(f"   s={s:.0f} bps/side: mean {st['mean_bps']:+6.2f}  SE_NW {st['se_nw']:4.2f}  "
              f"t_NW {st['t_nw']:+.2f}")
    gm = have["gross_mid_bps"].to_numpy()
    print(f"   (sanity: realized-fill gross mean {_gross_bps(have, notional_primary).mean():+.2f} "
          f"vs mid-path gross {gm.mean():+.2f} bps)")

    print("\n=== 3. DOLLAR P&L, CAPITAL, ROI (filtered strategy) ===")
    dollar_block(kept, notional_primary, span_days)
    dollar_block(test100.filter(pl.col("keep")), 100_000.0, span_days)

    print("\n=== 4. RISK / SHARPE (filtered, ${:.0f}k) ===".format(notional_primary/1000))
    sharpe_block(kept, notional_primary, _net_bps(kept, notional_primary))
    print(f"  max drawdown (equity) ${max_drawdown(kept):,.0f}")
    nb = _net_bps(kept, notional_primary)
    wins = (nb > 0).mean()
    print(f"  hit rate              {wins*100:.0f}%   ({int((nb>0).sum())}/{len(nb)} trades net-positive)")
    print(f"  best / worst trade    {nb.max():+.0f} / {nb.min():+.0f} bps   "
          f"(median {np.median(nb):+.0f}, mean {nb.mean():+.0f})")

    print("\n=== 5. VERDICT vs WITHIN-TRAIN EXPECTATION ===")
    net_filt = _net_bps(kept, notional_primary)
    s = hac_stats(net_filt)
    print(f"  OOS filtered net      {s['mean_bps']:+.2f} bps   t_NW {s['t_nw']:+.2f}")
    print( "  within-train holdout  +51.01 bps (t_NW 3.22) optimistic / +41-45 floor (F7/F9)")
    print( "  regime prior (F4)     edge concentrated in the Dec23-Mar24 bull; the test")
    print( "                        window is a different regime -- a genuine test.")


def main() -> None:
    _eprint("loading liq + agg (warmup 2024-04-15 -> test end 2024-10-14) ...")
    liq = bt.load_liq_snap(start_date=LOAD_START, end_date=LOAD_END).collect().sort("time_datetime")
    agg = bt.load_agg_trades(start_date=LOAD_START, end_date=LOAD_END).collect().sort("transact_time_datetime")
    agg_qty = bm._same_direction_aggregate_quantities(liq_df=liq, trades_df=agg, seconds_before=5, seconds_after=5)
    liq = liq.with_columns(pl.Series("agg_qty_5s_before_5s_after", agg_qty))

    _eprint("running model-C 5min tail reversion at $50k and $100k ...")
    t50 = run_size(liq, agg, 50_000.0)
    t100 = run_size(liq, agg, 100_000.0)
    _eprint(f"built {len(t50)} tail events over [warmup, test end]")

    _eprint("book features (displacement + mid path) ...")
    t50 = add_book_features(t50)
    keep = causal_keep(t50)
    t50 = t50.with_columns(keep=pl.Series(keep))
    # transfer keep/displacement to the $100k set (trim is fill-independent)
    t100 = t100.join(t50.select("decision_time", "keep"), on="decision_time", how="inner")

    test50 = t50.filter((pl.col("decision_time") >= TEST_START) & (pl.col("decision_time") < TEST_END))
    test100 = t100.filter((pl.col("decision_time") >= TEST_START) & (pl.col("decision_time") < TEST_END))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    test50.write_csv(OUT)
    _eprint(f"wrote {OUT} ({len(test50)} test-period trades)")

    report(test50, 50_000.0, test100)
    print(f"\nWrote {OUT}.")


if __name__ == "__main__":
    main()
