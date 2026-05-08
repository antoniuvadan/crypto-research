#!/usr/bin/env python3
"""
Download BTCUSD_PERP liquidationSnapshot zip files from Binance Vision for a given date range.
For each zip: extract CSV, add time_datetime, save as parquet, delete the CSV.

Usage:
  python downloader.py [--output-dir ./data] [--workers 4]
"""

import argparse
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
S3_BASE    = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CDN_BASE   = "https://data.binance.vision"
PREFIX     = "data/futures/cm/daily/liquidationSnapshot/BTCUSD_PERP/"
S3_NS      = "{http://s3.amazonaws.com/doc/2006-03-01/}"
CHUNK_SIZE = 1 << 16   # 64 KB

DATE_START = date(2023, 6, 25)
DATE_END   = date(2024, 10, 14)
# ─────────────────────────────────────────────────────────────────────────────


def list_all_keys(prefix: str) -> list[str]:
    keys, marker = [], None
    with tqdm(desc="Listing files", unit=" page", leave=False) as pbar:
        while True:
            params: dict = {"prefix": prefix, "delimiter": "/"}
            if marker:
                params["marker"] = marker
            r = requests.get(S3_BASE, params=params, timeout=30)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            batch = [
                c.findtext(f"{S3_NS}Key")
                for c in root.findall(f"{S3_NS}Contents")
                if c.findtext(f"{S3_NS}Key", "").endswith(".zip")
            ]
            keys.extend(batch)
            pbar.update(1)
            if root.findtext(f"{S3_NS}IsTruncated", "false").lower() != "true":
                break
            marker = root.findtext(f"{S3_NS}NextMarker") or (batch[-1] if batch else None)
            if not marker:
                break
    return keys


def key_date(key: str) -> date | None:
    stem = key.split("/")[-1].removesuffix(".zip")
    parts = stem.split("-")
    try:
        return date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
    except (ValueError, IndexError):
        return None


def filter_keys(keys: list[str]) -> list[str]:
    return [k for k in keys if (d := key_date(k)) is not None and DATE_START <= d <= DATE_END]


def download_one(key: str, out_dir: Path, session: requests.Session) -> tuple[str, bool, str]:
    filename = key.split("/")[-1]
    dest = out_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return filename, True, "skipped (already exists)"
    url = f"{CDN_BASE}/{key}"
    tmp = dest.with_suffix(".part")
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                desc=filename, total=total, unit="B",
                unit_scale=True, unit_divisor=1024, leave=False, ncols=90,
            ) as bar:
                for chunk in r.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)
        return filename, True, ""
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return filename, False, str(e)


def process_zip(zp: Path) -> None:
    """Extract CSV from zip, enrich it, save as parquet, delete the CSV."""
    parquet_path = zp.with_suffix(".parquet")
    if parquet_path.exists():
        return

    with zipfile.ZipFile(zp) as z:
        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        z.extractall(zp.parent, members=csv_names)

    for csv_name in csv_names:
        csv_path = zp.parent / csv_name
        df = pd.read_csv(csv_path)

        df["time_datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True)

        df.to_parquet(parquet_path, index=False)
        csv_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Download and process BTCUSD_PERP liquidationSnapshot from Binance Vision")
    parser.add_argument("--output-dir", default="/Users/vadanantoniu/Documents/research/MFin/crypto/data/BTCUSD_PERP-liquidationSnapshot")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir : {out_dir.resolve()}")
    print(f"Date range : {DATE_START} → {DATE_END} (inclusive)")
    print(f"Workers    : {args.workers}\n")

    print("Step 1/3 — Listing available files...")
    all_keys = list_all_keys(PREFIX)
    keys = filter_keys(all_keys)
    print(f"  Found {len(all_keys)} total zips, {len(keys)} in date range.\n")
    if not keys:
        print("No files to download.")
        return

    print("Step 2/3 — Downloading...")
    failed = []
    session = requests.Session()
    with tqdm(total=len(keys), desc="Overall", unit=" file", ncols=90) as overall:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_one, k, out_dir, session): k for k in keys}
            for fut in as_completed(futures):
                fname, ok, msg = fut.result()
                if not ok:
                    failed.append((fname, msg))
                    tqdm.write(f"  FAILED  {fname}: {msg}")
                elif msg:
                    tqdm.write(f"  SKIPPED {fname}")
                overall.update(1)
    if failed:
        print(f"\n{len(failed)} downloads failed:")
        for fname, err in failed:
            print(f"  {fname}: {err}")

    print("\nStep 3/3 — Processing zips (extract → enrich → parquet → delete CSV)...")
    zips = sorted(out_dir.glob("BTCUSD_PERP-liquidationSnapshot-*.zip"))
    for zp in tqdm(zips, unit=" file", ncols=90):
        process_zip(zp)

    print("\nDone.")


if __name__ == "__main__":
    main()
