#!/usr/bin/env python3
"""
Download + process ETHUSD_PERP (Binance COIN-M) daily data from Binance Vision,
mirroring the existing BTCUSD_PERP files exactly (naming, schema, format).

Three datasets over the BTC study period (2023-06-25 -> 2024-10-14):
  liquidationSnapshot -> data/ETHUSD_PERP-liquidationSnapshot/*.parquet
  aggTrades           -> data/ETHUSD_PERP-aggTrades/*.parquet
  bookTicker          -> data/ETHUSD_PERP-bookTicker/*.parquet  AND  *.zip

Per BTC convention: liq/agg keep only the parquet (zip deleted after processing);
bookTicker keeps BOTH the raw .zip and the processed .parquet.

Derived columns added to match the BTC parquet schema:
  liq  : time_datetime            = to_datetime(time, ms, UTC)
  agg  : transact_time_datetime   = to_datetime(transact_time, ms, UTC)
  book : mid_price = (bid+ask)/2
         micro_price = (bid*ask_qty + ask*bid_qty)/(bid_qty+ask_qty)
         event_time_datetime = to_datetime(event_time, ms, UTC)

Resumable: skips any date whose parquet already exists. Runs each dataset with a
thread pool (download+process per date), smallest dataset first.

Usage:
  python download_eth.py [--workers 6] [--datasets liquidationSnapshot,aggTrades,bookTicker]
"""

from __future__ import annotations

import argparse
import io
import sys
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests

S3_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CDN_BASE = "https://data.binance.vision"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
SYMBOL = "ETHUSD_PERP"
DATE_START = date(2023, 6, 25)
DATE_END = date(2024, 10, 14)
DATA_DIR = Path("data")

# Raw Binance CSV column names per dataset (used if a file ships without a header).
COLS = {
    "liquidationSnapshot": ["time", "side", "order_type", "time_in_force",
                            "original_quantity", "price", "average_price",
                            "order_status", "last_fill_quantity", "accumulated_fill_quantity"],
    "aggTrades": ["agg_trade_id", "price", "quantity", "first_trade_id",
                  "last_trade_id", "transact_time", "is_buyer_maker"],
    "bookTicker": ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price",
                   "best_ask_qty", "transaction_time", "event_time"],
}
KEEP_ZIP = {"liquidationSnapshot": False, "aggTrades": False, "bookTicker": True}


def eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def prefix(ds: str) -> str:
    return f"data/futures/cm/daily/{ds}/{SYMBOL}/"


def out_dir(ds: str) -> Path:
    return DATA_DIR / f"{SYMBOL}-{ds}"


def list_keys(ds: str) -> list[str]:
    keys: list[str] = []
    marker = None
    while True:
        params: dict = {"prefix": prefix(ds), "delimiter": "/"}
        if marker:
            params["marker"] = marker
        r = requests.get(S3_BASE, params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        batch = [c.findtext(f"{S3_NS}Key") for c in root.findall(f"{S3_NS}Contents")
                 if c.findtext(f"{S3_NS}Key", "").endswith(".zip")]
        keys.extend(k for k in batch if k)
        if root.findtext(f"{S3_NS}IsTruncated", "false").lower() != "true":
            break
        marker = root.findtext(f"{S3_NS}NextMarker") or (batch[-1] if batch else None)
        if not marker:
            break
    return keys


def key_date(key: str) -> date | None:
    p = key.split("/")[-1].removesuffix(".zip").split("-")
    try:
        return date(int(p[-3]), int(p[-2]), int(p[-1]))
    except (ValueError, IndexError):
        return None


def read_csv_bytes(raw: bytes, ds: str) -> pd.DataFrame:
    """Read a raw Binance daily CSV, whether or not it ships a header row."""
    first = raw[:64].lstrip().split(b",")[0].decode("utf-8", "ignore").strip().strip('"')
    has_header = not first.lstrip("-").isdigit()  # first field is text -> header row
    if has_header:
        return pd.read_csv(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw), header=None, names=COLS[ds])


def enrich(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    if ds == "liquidationSnapshot":
        df["time_datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    elif ds == "aggTrades":
        if df["is_buyer_maker"].dtype != bool:
            df["is_buyer_maker"] = (
                df["is_buyer_maker"].astype(str).str.lower().map({"true": True, "false": False})
            )
        df["transact_time_datetime"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    elif ds == "bookTicker":
        df["mid_price"] = (df["best_bid_price"] + df["best_ask_price"]) / 2.0
        df["micro_price"] = (
            df["best_bid_price"] * df["best_ask_qty"] + df["best_ask_price"] * df["best_bid_qty"]
        ) / (df["best_bid_qty"] + df["best_ask_qty"])
        df["event_time_datetime"] = pd.to_datetime(df["event_time"], unit="ms", utc=True)
    return df


def fetch_and_process(key: str, ds: str, session: requests.Session) -> tuple[str, bool, str]:
    fname = key.split("/")[-1]
    d = out_dir(ds)
    parquet = d / fname.replace(".zip", ".parquet")
    zip_path = d / fname
    if parquet.exists():
        return fname, True, "skip (parquet exists)"
    try:
        # Download the zip (stream to disk for bookTicker so the raw file is kept).
        if not (zip_path.exists() and zip_path.stat().st_size > 0):
            with session.get(f"{CDN_BASE}/{key}", stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = zip_path.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                tmp.rename(zip_path)
        # Extract CSV in-memory, enrich, write parquet.
        with zipfile.ZipFile(zip_path) as z:
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
            raw = z.read(csv_name)
        df = enrich(read_csv_bytes(raw, ds), ds)
        df.to_parquet(parquet, index=False)
        if not KEEP_ZIP[ds]:
            zip_path.unlink(missing_ok=True)
        return fname, True, ""
    except Exception as e:  # noqa: BLE001 - report and continue
        return fname, False, repr(e)


def run_dataset(ds: str, workers: int) -> None:
    out_dir(ds).mkdir(parents=True, exist_ok=True)
    eprint(f"\n[{ds}] listing keys ...")
    keys = [k for k in list_keys(ds) if (d := key_date(k)) and DATE_START <= d <= DATE_END]
    eprint(f"[{ds}] {len(keys)} files in range {DATE_START}..{DATE_END}")
    done = fail = 0
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_and_process, k, ds, session): k for k in keys}
        for i, fut in enumerate(as_completed(futs), 1):
            fname, ok, msg = fut.result()
            done += ok
            fail += (not ok)
            if not ok:
                eprint(f"  FAIL {fname}: {msg}")
            if i % 25 == 0 or i == len(keys):
                eprint(f"[{ds}] {i}/{len(keys)} done ({fail} failed)")
    eprint(f"[{ds}] complete: {done} ok, {fail} failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", default="liquidationSnapshot,aggTrades,bookTicker")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        ds = ds.strip()
        if ds not in COLS:
            eprint(f"unknown dataset {ds!r}, skipping")
            continue
        run_dataset(ds, args.workers)
    eprint("\nAll requested datasets processed.")


if __name__ == "__main__":
    main()
