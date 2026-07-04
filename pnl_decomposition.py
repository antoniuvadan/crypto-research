#!/usr/bin/env python3
"""
Step 1 — Decompose Model C net P&L into additive components.

For every completed round trip in liquidation_momentum_model_c_trades.csv, split
the net return (bps, sign-corrected by trade direction) into:

    net = gross_mid_to_mid          (signal: mid at decision -> mid at exit horizon)
          - latency_slippage        (mid drift during the 300ms decision->entry delay)
          - spread_entry            (entry fill vs mid at entry: spread + book-walk)
          - spread_exit             (exit fill vs mid at exit: spread + book-walk)
          - fee_entry               (5 bps taker)
          - fee_exit                (5 bps taker)

Each term is additive in price space, so the bps terms (all divided by the same
per-trade denominator, mid_decision) sum exactly to the reconstructed net.

Mids come from the L1 bookTicker parquet, as-of joined (last quote at or before
each reference timestamp). Processed one calendar day at a time to bound memory.

Usage:
  python pnl_decomposition.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

DEFAULT_TRADES_PATH = Path("data/results/liquidation_momentum_model_c_trades.csv")
BOOK_TICKER_DIR = Path("data/BTCUSD_PERP-bookTicker")
RESULTS_DIR = Path("data/results")
FEE_RATE = 0.0005  # 5 bps taker per leg
HOLDING_ORDER = ["5s", "10s", "30s", "1min", "2min", "5min", "30min", "60min"]


def _book_file(d: date) -> Path:
    return BOOK_TICKER_DIR / f"BTCUSD_PERP-bookTicker-{d.isoformat()}.parquet"


def _epoch_ns(col: str) -> pl.Expr:
    return pl.col(col).dt.epoch("ns").alias("ts_ns")


def add_decomposition(df: pl.DataFrame) -> pl.DataFrame:
    """Add the 6 additive bps components + reconstruction to a frame that already carries the
    per-trade reference mids in columns `decision`/`entry`/`exit` (plus `entry_vwap`,
    `exit_vwap`, `direction`, `net_pnl`, `entry_quantity`).

    Split out of `main()` so other callers (e.g. the concurrency cap study) reuse the exact
    formula rather than duplicating it. All price terms share the denominator `decision`
    (mid at decision) so they are additive: `net_reconstructed == net_realized` to L1 resolution.
    """
    d = pl.col("direction").cast(pl.Float64)  # +1 long, -1 short
    denom = pl.col("decision")  # common per-trade denominator for additivity
    bps = 1e4
    return df.with_columns(
        gross_mid_to_mid=bps * d * (pl.col("exit") - pl.col("decision")) / denom,
        latency_slippage=bps * d * (pl.col("entry") - pl.col("decision")) / denom,
        spread_entry=bps * d * (pl.col("entry_vwap") - pl.col("entry")) / denom,
        spread_exit=bps * d * (pl.col("exit") - pl.col("exit_vwap")) / denom,
        fee_entry=pl.lit(bps * FEE_RATE),
        fee_exit=pl.lit(bps * FEE_RATE),
    ).with_columns(
        net_reconstructed=(
            pl.col("gross_mid_to_mid")
            - pl.col("latency_slippage")
            - pl.col("spread_entry")
            - pl.col("spread_exit")
            - pl.col("fee_entry")
            - pl.col("fee_exit")
        ),
        # Realized net straight from the backtest, in bps of traded notional.
        net_realized=bps * pl.col("net_pnl") / (pl.col("entry_quantity").abs() * 100.0),
    )


def lookup_mids(lookups: pl.DataFrame) -> pl.DataFrame:
    """
    lookups: columns [tid, role, ts (Datetime UTC)].
    Returns [tid, role, mid] via backward as-of join on each day's bookTicker.

    Epoch-ns integer keys sidestep any tz dtype mismatch between the CSV
    (tz-aware UTC) and the parquet (tz-naive UTC wall-clock).
    """
    lookups = lookups.with_columns(
        _epoch_ns("ts"),
        pl.col("ts").dt.date().alias("date"),
    )
    out: list[pl.DataFrame] = []
    dates = lookups["date"].unique().sort().to_list()
    for i, d in enumerate(dates, start=1):
        path = _book_file(d)
        day_lookups = lookups.filter(pl.col("date") == d).sort("ts_ns")
        if not path.exists():
            print(f"  [warn] no bookTicker file for {d}; {len(day_lookups)} lookups dropped", file=sys.stderr)
            continue
        book = (
            pl.read_parquet(path, columns=["event_time_datetime", "mid_price"])
            .with_columns(_epoch_ns("event_time_datetime"))
            .select(["ts_ns", "mid_price"])
            .sort("ts_ns")
        )
        joined = day_lookups.join_asof(book, on="ts_ns", strategy="backward")
        out.append(joined.select(["tid", "role", "mid_price"]))
        print(f"  [{i}/{len(dates)}] {d}: {len(day_lookups)} lookups", file=sys.stderr)

    return pl.concat(out).rename({"mid_price": "mid"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trades", type=Path, default=DEFAULT_TRADES_PATH,
        help="Model C trades CSV to decompose (default: momentum).",
    )
    parser.add_argument(
        "--label", default="pnl_decomposition",
        help="Output basename: data/results/{label}_{trades,summary}.csv.",
    )
    args = parser.parse_args()

    if not args.trades.exists():
        sys.exit(f"trades file not found: {args.trades}")

    trades = pl.read_csv(args.trades).with_row_index("tid")
    for c in ("decision_time", "entry_start_time", "exit_start_time"):
        trades = trades.with_columns(
            pl.col(c).str.to_datetime(time_zone="UTC").alias(c)
        )

    # Three mid lookups per trade.
    role_cols = {
        "decision": "decision_time",
        "entry": "entry_start_time",
        "exit": "exit_start_time",
    }
    lookups = pl.concat(
        [
            trades.select(
                "tid",
                pl.lit(role).alias("role"),
                pl.col(col).alias("ts"),
            )
            for role, col in role_cols.items()
        ]
    )

    print(f"Resolving {len(lookups):,} mid lookups across days...", file=sys.stderr)
    mids = lookup_mids(lookups)
    wide = mids.pivot(values="mid", index="tid", on="role")

    df = trades.join(wide, on="tid", how="left")

    # Drop trades missing any reference mid (so the decomposition stays exact).
    n_before = len(df)
    df = df.drop_nulls(subset=["decision", "entry", "exit"])
    dropped = n_before - len(df)
    if dropped:
        print(f"[warn] dropped {dropped}/{n_before} trades missing a reference mid", file=sys.stderr)

    df = add_decomposition(df)

    component_cols = [
        "gross_mid_to_mid",
        "latency_slippage",
        "spread_entry",
        "spread_exit",
        "fee_entry",
        "fee_exit",
        "net_reconstructed",
        "net_realized",
    ]

    table = (
        df.group_by("trade_notional_usd", "holding_period")
        .agg(
            pl.len().alias("n"),
            *[pl.col(c).mean().alias(c) for c in component_cols],
        )
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

    print("\n=== Model C net P&L decomposition (mean bps per trade, sign-corrected) ===")
    print("net = gross_mid_to_mid - latency_slippage - spread_entry - spread_exit - fee_entry - fee_exit")
    print("(costs are reported as positive numbers that are subtracted)\n")
    print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trades_out = RESULTS_DIR / f"{args.label}_trades.csv"
    summary_out = RESULTS_DIR / f"{args.label}_summary.csv"
    df.write_csv(trades_out)
    table.write_csv(summary_out)
    print(f"\nWrote {trades_out} and {summary_out}")


if __name__ == "__main__":
    main()
