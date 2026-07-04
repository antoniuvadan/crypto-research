#!/usr/bin/env python3
"""
Generate the frozen strategy's TRIM-FILTERED trades over the TRAINING window
(2023-06-25 -> 2024-06-24), for the concurrency risk diagnostic.

Same frozen config as the OOS driver (>=98th tail, 5-min taker reversion, causal 30d
displacement trim) -- reuses `run_size`, `add_book_features`, `causal_keep` verbatim, only
the date window and the book-pass start (`disp_from`) change. Train is NOT the spent test
holdout, so this is free to compute. Cached to a CSV so the heavy book pass runs once.

Usage:
  python strategies/gen_train_trades.py        # run from the repo root
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

import backtester as bt
import backtest_momentum as bm
from backtest_oos_test import run_size, add_book_features, causal_keep

TRAIN_START = date(2023, 6, 25)
TRAIN_END = date(2024, 6, 24)
DISP_FROM_TRAIN = datetime(2023, 6, 25, tzinfo=timezone.utc)  # book pass over all train events
OUT = Path("data/results/train_filtered_trades.csv")


def _eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def main() -> None:
    _eprint(f"loading liq + agg over train {TRAIN_START} -> {TRAIN_END} ...")
    liq = bt.load_liq_snap(start_date=TRAIN_START, end_date=TRAIN_END).collect().sort("time_datetime")
    agg = bt.load_agg_trades(start_date=TRAIN_START, end_date=TRAIN_END).collect().sort("transact_time_datetime")
    agg_qty = bm._same_direction_aggregate_quantities(liq_df=liq, trades_df=agg, seconds_before=5, seconds_after=5)
    liq = liq.with_columns(pl.Series("agg_qty_5s_before_5s_after", agg_qty))

    _eprint("running frozen model-C 5min tail reversion at $50k over train ...")
    t = run_size(liq, agg, 50_000.0)
    _eprint(f"built {len(t)} tail events over train")

    _eprint("book features (displacement + mid path) over all train events ...")
    t = add_book_features(t, disp_from=DISP_FROM_TRAIN)
    keep = causal_keep(t)
    t = t.with_columns(keep=pl.Series(keep))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.write_csv(OUT)
    kept = t.filter(pl.col("keep"))
    net_bps = (1e4 * kept["net_pnl"] / 50_000.0)
    _eprint(f"wrote {OUT}: {len(t)} tail events, {len(kept)} kept ({len(kept)/len(t)*100:.0f}%)")
    _eprint(f"[smoke] within-train kept net {net_bps.mean():+.2f} bps  "
            f"(F7/F9 within-train region ~ +51 optimistic / +41-45 floor)")


if __name__ == "__main__":
    main()
