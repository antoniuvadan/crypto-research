#!/usr/bin/env python3
"""
Probabilistic Sharpe Ratio PSR(0) and Deflated Sharpe Ratio (DSR) for the reversion strategies.

PSR(0) = P(true Sharpe > 0), corrected for finite sample + non-normality (skew/kurtosis). DSR =
PSR at SR* = the expected max Sharpe of N trials, i.e. corrected for selection bias / how many
configs were searched. Both are computed on the PER-TRADE net returns; because trades overlap
(up to ~18 concurrent), the honest sample size is the number of independent EPISODES, not trades,
so PSR/DSR are reported at BOTH raw n and n_eff = episodes.

Caveats (documented in the printout): DSR's inputs N (# trials) and V (cross-trial SR variance)
are judgement calls, so DSR is shown across an N grid with V estimated from the train-policy
Sharpe dispersion. DSR corrects IN-SAMPLE selection; the OOS result was a single frozen run on a
reserved set, so for it PSR(0) is the honest statistic and a train-N DSR would double-count.

Usage (repo root):
  ~/miniforge3/envs/mscf/bin/python strategies/psr_dsr_report.py
"""

from __future__ import annotations

import sys as _sys
from datetime import datetime, timedelta, timezone
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys

import numpy as np
import polars as pl

from significance import psr, dsr, expected_max_sharpe
from concurrency_analysis import load, net_bps, cap_at_n, assign_episodes, OOS_CSV, TRAIN_CSV

UTC = timezone.utc
HOLDOUT_START = datetime(2024, 2, 25, tzinfo=UTC)
TRAIN_END = datetime(2024, 6, 25, tzinfo=UTC)
N_GRID = (10, 30, 100)
CAPS = (5, 3, 1)


def _eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def n_episodes(df: pl.DataFrame) -> int:
    return int(assign_episodes(df, timedelta(0)).max()) + 1 if len(df) else 0


def strategies() -> list[tuple[str, pl.DataFrame]]:
    out: list[tuple[str, pl.DataFrame]] = []
    if OOS_CSV.exists():
        out.append(("OOS uncapped (F10)", load(OOS_CSV)))
    if TRAIN_CSV.exists():
        train = load(TRAIN_CSV)
        dev = train.filter(pl.col("decision_time") < HOLDOUT_START)
        hold = train.filter((pl.col("decision_time") >= HOLDOUT_START)
                            & (pl.col("decision_time") < TRAIN_END))
        for win, w in (("dev", dev), ("holdout", hold)):
            out.append((f"TRAIN {win} uncapped", w))
            for n in CAPS:
                out.append((f"TRAIN {win} greedy N={n}", cap_at_n(w, n)))
    return out


def main() -> None:
    strat = strategies()
    # per-trade Sharpe of each strategy (feeds PSR, and its dispersion feeds the DSR benchmark V)
    recs = []
    for label, df in strat:
        y = net_bps(df)
        ne = n_episodes(df)
        p_raw = psr(y, sr_star=0.0)
        p_eff = psr(y, sr_star=0.0, n_eff=ne)
        recs.append({"label": label, "y": y, "n_eff": ne, "sr": p_raw["sr_hat"],
                     "skew": p_raw["skew"], "kurt": p_raw["kurtosis"],
                     "n": len(df), "psr_raw": p_raw["psr"], "psr_eff": p_eff["psr"]})

    # V = variance of the per-trade Sharpe across the TRAIN trials (a proxy for the search
    # dispersion; the real search had more configs, so this is a lower bound on V).
    train_srs = [r["sr"] for r in recs if r["label"].startswith("TRAIN")]
    var_sr = float(np.var(train_srs, ddof=1))

    pl.Config.set_tbl_rows(50); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(240)
    pl.Config.set_float_precision(3)

    print("\n=== PSR(0) = P(true Sharpe > 0), per-trade returns ===")
    print("psr_raw uses n = #trades; psr_eff uses n_eff = #episodes (honest for overlapping trades).")
    print("PSR(0) > 0.975 ~ significant at 5% one-sided. skew<0 / kurt>3 lower PSR.\n")
    ptab = pl.DataFrame([{k: r[k] for k in
                          ("label", "n", "n_eff", "sr", "skew", "kurt", "psr_raw", "psr_eff")}
                         for r in recs])
    print(ptab)

    print(f"\n=== DSR = PSR at SR* (expected max Sharpe of N trials); n_eff basis ===")
    print(f"V (cross-trial per-trade SR variance, from {len(train_srs)} train policies) = {var_sr:.4f}")
    print("SR* and DSR shown across a trial-count grid (N is a judgement input):")
    for N in N_GRID:
        print(f"   N={N:>3}: SR* = {expected_max_sharpe(N, var_sr):+.3f}")
    print("NOTE: DSR corrects IN-SAMPLE selection. For 'OOS uncapped' the reserved-set hold-out")
    print("already controls selection, so PSR(0) is its honest statistic; a train-N DSR on it")
    print("would DOUBLE-COUNT the correction (shown greyed by intent -- read PSR(0) for OOS).\n")
    drows = []
    for r in recs:
        row = {"label": r["label"], "sr": r["sr"], "n_eff": r["n_eff"], "psr0_eff": r["psr_eff"]}
        for N in N_GRID:
            row[f"dsr_N{N}"] = dsr(r["y"], n_trials=N, var_sr=var_sr, n_eff=r["n_eff"])["psr"]
        drows.append(row)
    print(pl.DataFrame(drows))

    out = _Path("data/results/psr_dsr_report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(drows).write_csv(out)
    _eprint(f"\nwrote {out}")


if __name__ == "__main__":
    main()
