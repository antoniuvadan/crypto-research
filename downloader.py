#!/usr/bin/env python3
"""
Download all zip files from:
  https://data.binance.vision/?prefix=data/futures/cm/daily/liquidationSnapshot/BTCUSD_PERP/

Usage:
  python download_binance_liq.py [--output-dir ./data] [--workers 4]
"""

import argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
S3_BASE      = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CDN_BASE     = "https://data.binance.vision"
PREFIX       = "data/futures/cm/daily/liquidationSnapshot/BTCUSD_PERP/"
S3_NS        = "{http://s3.amazonaws.com/doc/2006-03-01/}"
CHUNK_SIZE   = 1 << 16   # 64 KB
# ─────────────────────────────────────────────────────────────────────────────


def list_all_keys(prefix: str) -> list[str]:
    """Page through S3 XML listing and return all .zip keys."""
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

            is_truncated = root.findtext(f"{S3_NS}IsTruncated", "false").lower()
            if is_truncated != "true":
                break
            marker = root.findtext(f"{S3_NS}NextMarker") or (batch[-1] if batch else None)
            if not marker:
                break

    return keys


def download_one(key: str, out_dir: Path, session: requests.Session) -> tuple[str, bool, str]:
    """Download a single zip. Returns (filename, success, error_msg)."""
    filename = key.split("/")[-1]
    dest = out_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        return filename, True, "skipped (already exists)"

    url = f"{CDN_BASE}/{key}"
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f, tqdm(
                desc=filename,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
                ncols=90,
            ) as bar:
                for chunk in r.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    bar.update(len(chunk))
            tmp.rename(dest)
        return filename, True, ""
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return filename, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Download BTCUSD_PERP liquidation snapshots from Binance Vision")
    parser.add_argument("--output-dir", default="/Users/vadanantoniu/Documents/research/MFin/crypto/data/BTCUSD_PERP-liquidationSnapshot", help="Directory to save zip files")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download threads (default: 4)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir : {out_dir.resolve()}")
    print(f"Workers    : {args.workers}")
    print()

    # 1. List all zips
    print("Step 1/2 — Listing available files...")
    keys = list_all_keys(PREFIX)
    if not keys:
        print("No zip files found. Check the prefix or your connection.")
        return
    print(f"  Found {len(keys)} zip files.\n")

    # 2. Download
    print("Step 2/2 — Downloading...")
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

    print()
    if failed:
        print(f"Done. {len(keys) - len(failed)}/{len(keys)} succeeded. {len(failed)} failed:")
        for fname, err in failed:
            print(f"  {fname}: {err}")
    else:
        print(f"Done. All {len(keys)} files downloaded to {out_dir.resolve()}")


if __name__ == "__main__":
    main()