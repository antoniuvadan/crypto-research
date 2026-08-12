#!/usr/bin/env python3
"""
The definitive out-of-sample Sharpe ratio of the frozen reversion strategy, with its conventions
stated at the point of use, two independent uncertainty estimates, and the matching drawdown.

The repo already contains the pieces (F10's annualized 4.46, the audit's block-bootstrap CI, F14's
Lo/PSR work) but they are scattered across three drivers and computed only on the OPTIMISTIC
realized-sweep fill basis. This driver consolidates them and puts the headline on the CONSERVATIVE
fill floor, which F9/F10 called "the number to believe".

No new backtests: every statistic is computed from already-realized trade CSVs. Nothing is
re-fitted, no parameter is tuned here, and the floor s-grid is frozen from F9. The spent test set
is not re-executed.

CONVENTIONS (all four are choices; each is stated so none is hidden):
  rf = 0        Futures are self-financing -- P&L over notional is already an excess return, and
                the margin base can sit in bills. Standard for margined futures. NOT a shortcut
                needing correction: the cash-drag row in the output is the separate counterfactual
                "what if the peak capital earned nothing?", and must not be averaged with it.
  cost basis    PRIMARY = conservative floor at s = 3 bps/side:
                    net_bps = gross_mid_bps - latency_slip_bps - 2s - 10   (F9/F10 method)
                Realized-sweep fills and the participation-capped stress are shown alongside.
  day count     ALL calendar days in the test span, idle days = 0 return. More conservative than
                dropping the idle days (F10: 4.46 vs 4.97) and matches how an allocation is held.
  annualization x sqrt(365), crypto being 24/7 -- but this is the WEAKEST of the four. F14 read
                Lo's eta ~ sqrt(365) as confirming the naive rule; that evidence is withdrawn
                here. On a series that is 80% idle days the autocorrelation estimate has no power
                (a true daily AR(1) of 0.5 still measures rho1 ~ +0.07 under this mask), and eta
                itself swings 17.9-23.9 across truncations. Lo is UNINFORMATIVE here, not
                confirmatory. F14's mechanism argument (5-min holds close intraday, so overlap
                washes out at the daily level) is untouched.

SCALE INVARIANCE: the Sharpe does not depend on the capital base. Scaling every daily P&L by a
constant leaves mean/std unchanged. The $900k peak-concurrent base affects the drawdown PERCENTAGE
and the ROI, never the Sharpe. (It does affect the cash-drag counterfactual, which is why that row
is labelled with its base.)

WHAT THIS DRIVER CONCLUDES: that the annualized Sharpe should NOT be the headline claim. Computing
it carefully is what shows why -- it moves 5.16 -> 4.15 -> 2.53 with the window end date alone, one
day is ~39% of the P&L, and on the traded-day sample its 90% interval includes zero. The quotable
results are the per-trade edge, its Newey-West t, PSR(0), and the mark-to-market drawdown. That is
the same place F14 landed; this driver corroborates it rather than overturning it.

Usage (repo root):
  ~/miniforge3/envs/mscf/bin/python strategies/oos_sharpe_report.py
  ~/miniforge3/envs/mscf/bin/python strategies/oos_sharpe_report.py --mtm   # + book pass for MtM DD
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from significance import hac_stats, lo_annualized_sharpe, psr
from concurrency_analysis import NOTIONAL, OOS_CSV, assign_episodes, effective_n, load
from sharpe_lo_adjustment import TEST_END, TEST_START, daily_pnl
from backtest_oos_risk import block_bootstrap_sharpe_dist, realized_close_dd
from backtest_oos_test import peak_capital

Q = 365                      # periods/yr: crypto trades 24/7
Z90 = 1.6448536269514722     # norm.ppf(0.95) -> two-sided 90%
ROUND_TRIP_FEE_BPS = 10.0    # 5 bps/leg taker
FLOOR_S_BPS = (2.0, 3.0, 4.0)
HEADLINE_S_BPS = 3.0
RF_ANNUAL = 0.0525           # ~3M T-bill over the test window (mid-2024); counterfactual only
BOOT_BLOCK = 7               # days; circular block bootstrap
LO_MAX_LAGS = (1, 5, 10, 20)  # F14's truncation grid -- keep all four; L=20 is the one that
                              # most contradicts the "eta ~ sqrt(365)" reading, so dropping it
                              # would make the annualizer look better behaved than it is

STRESS_CSV = Path("data/results/oos_test_slippage_stress_trades.csv")
OUT_CSV = Path("data/results/oos_sharpe_report.csv")

# Frozen figures this driver must reproduce (F10 / OOS_RISK_AUDIT). Guards, not inputs.
FROZEN_NET_BPS = 48.85
FROZEN_IR_TRADE = 0.467
FROZEN_ANN_SHARPE = 4.46
FROZEN_MTM_DD_USD = -22_770.0


def _eprint(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Cost bases -- each restates `net_pnl` under a different fill assumption
# ---------------------------------------------------------------------------

def floor_frame(oos: pl.DataFrame, s_bps: float) -> pl.DataFrame:
    """F9/F10 conservative fill floor: throw away the realized sweep VWAPs and re-price both legs
    off the EXACT mid path at a flat `s` bps/side taker spread.

        net_bps = gross_mid_bps - latency_slip_bps - 2*s - round_trip_fee

    `gross_mid_bps` is the decision-mid-to-exit-mid move and `latency_slip_bps` the drift over the
    300ms execution delay, so subtracting both leaves a pure mid-to-mid P&L charged a fixed spread.
    Returns the frame with `net_pnl` (USD) restated; all other columns are untouched, so episode
    and concurrency accounting still work."""
    bps = (pl.col("gross_mid_bps") - pl.col("latency_slip_bps")
           - 2.0 * s_bps - ROUND_TRIP_FEE_BPS)
    return oos.with_columns(net_pnl=bps / 1e4 * NOTIONAL)


def cost_bases(oos: pl.DataFrame, stress: pl.DataFrame | None) -> list[tuple[str, pl.DataFrame]]:
    """(label, frame) per fill assumption, headline first then increasing conservatism."""
    out: list[tuple[str, pl.DataFrame]] = [
        # "[PRIMARY BASIS]", not "[HEADLINE]": the CSV label travels without the caveats, and a
        # cell reading HEADLINE is exactly the artifact that would get quoted on its own.
        (f"floor s={HEADLINE_S_BPS:.0f}bps  [PRIMARY BASIS]", floor_frame(oos, HEADLINE_S_BPS)),
        ("realized-sweep fills (F10)", oos),
    ]
    if stress is not None:
        for p in (0.20, 0.10):
            sub = stress.filter((pl.col("participation") - p).abs() < 1e-9)
            if len(sub):
                out.append((f"participation cap {p:.2f}", sub))
    out += [(f"floor s={s:.0f}bps", floor_frame(oos, s))
            for s in FLOOR_S_BPS if s != HEADLINE_S_BPS]
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def net_bps_of(df: pl.DataFrame) -> np.ndarray:
    """Per-trade net return in bps of the INTENDED notional -- the $50k the strategy meant to
    trade, not the filled quantity (the F9/F10 convention).

    This would matter if any trade were left short, but empirically none is: every trade in every
    basis fills exactly 500 contracts, including the participation-capped ones. (`entry_complete`
    reads False on ~40 capped trades, but the shortfall is ~3e-13 contracts -- a floating-point
    artifact of `filled >= requested` in `backtester.py`, not a real partial fill. The
    participation cap degraded fill VWAPs; it never left a trade unfilled.)

    The sort is load-bearing: `hac_stats` computes a Newey-West SE, which is defined on a
    time-ordered series and silently wrong on an unordered one."""
    return (1e4 * df.sort("decision_time")["net_pnl"] / NOTIONAL).to_numpy()


def daily_max_dd(daily: np.ndarray) -> float:
    """Max drawdown of the daily equity curve, measured from a zero starting equity so that an
    opening loss counts as a drawdown (np.maximum.accumulate alone would anchor the first peak at
    equity[0] and miss it). Coarser than the per-trade realized-close DD and far coarser than the
    mark-to-market DD -- it cannot see a trough that opens and closes inside one day."""
    if len(daily) == 0:
        return 0.0
    eq = np.concatenate([[0.0], np.cumsum(daily)])
    return float((eq - np.maximum.accumulate(eq)).min())


def n_episodes(df: pl.DataFrame) -> int:
    """Independent cascade episodes: overlapping/back-to-back trades chained into one (F14's
    effective sample for the per-trade series)."""
    return int(assign_episodes(df, timedelta(0)).max()) + 1 if len(df) else 0


def gaussian_sharpe_se(sr: float, n: int) -> float:
    """Standard error of a Sharpe estimate under IID normal returns (Jobson-Korkie / Lo 2002):

        SE(SR) = sqrt( (1 + SR^2 / 2) / (n - 1) )

    Neither closed-form SE is trustworthy on this series and it is worth being explicit about why,
    because they disagree by 2.5x:

      * The moment-corrected SE (`significance.psr`) needs believable 3rd/4th sample moments. The
        daily series is ~80% exact zeros, which drives skew ~ +7.5 and kurtosis ~ 68. Those enter
        the variance as `1 - g3*SR + (g4-1)/4*SR^2`, whose skew term goes NEGATIVE (-0.63) and is
        rescued only by the kurtosis term -- shrinking the SE ~6x and pinning PSR(0) at 1.000 for
        every cost basis.
      * This Gaussian SE does not fix that; it merely ASSERTS g3=0, g4=3 when the sample says 7.5
        and 68. It is not "more robust", it is differently wrong -- and ON THE OOS DAILY SERIES it
        is empirically the FURTHER of the two from the bootstrap's estimate of the sampling SD.

    That last comparison is a fact about THIS series, not a general property of zero-inflation:
    on other zero-inflated series (including synthetic ones with a large adverse day) the Gaussian
    SE comes out too NARROW instead. The general claim is only that both closed forms are
    mis-calibrated under zero-inflation; the direction has to be measured each time.

    So neither is primary. The block bootstrap resamples the observed zero-inflated structure and
    is the arbiter; these two are reported as bracketing references (`boot_sd` in `sharpe_stats`
    shows how far off each is)."""
    return float(np.sqrt((1.0 + sr ** 2 / 2.0) / (n - 1))) if n > 1 else float("nan")


def sharpe_stats(daily: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    """Annualized Sharpe of a daily P&L series with every uncertainty estimate this data supports.

    boot   -- 7-day circular blocks, which preserve short-horizon dependence AND the zero-inflated
              structure. THE PRIMARY INTERVAL, reported as the basic (reverse-percentile) form
              [2*SR_hat - q_hi, 2*SR_hat - q_lo]: on this series the bootstrap median sits above
              the point estimate (resamples drawing more active days score higher), so the raw
              percentile interval is centred too high and the basic form removes that shift.
              `boot_sd` is the empirical sampling SD of SR_ann -- the arbiter between the two
              closed forms below.
    gauss  -- sqrt(Q)*(SR +/- z*SE); assumes IID normality.
    moment -- the PSR/Mertens interval. See `gaussian_sharpe_se` for why neither closed form is
              quotable on a zero-inflated series.

    On the OOS daily series specifically, gauss runs ~1.7x wider and moment ~0.7x narrower than
    boot_sd -- but that DIRECTION is a property of this series, not of the estimators, so the
    printed output computes both ratios rather than asserting them.
    """
    sd = float(daily.std(ddof=1))
    sr_d = float(daily.mean()) / sd if sd > 0 else float("nan")
    ann = np.sqrt(Q) * sr_d

    se_g = gaussian_sharpe_se(sr_d, len(daily))
    p = psr(daily, sr_star=0.0)
    dist = block_bootstrap_sharpe_dist(daily, block=BOOT_BLOCK, n_boot=n_boot, seed=seed)
    lo_b, mid_b, hi_b = (float(np.percentile(dist, 5)), float(np.percentile(dist, 50)),
                         float(np.percentile(dist, 95)))
    return {
        "boot_sd": float(np.std(dist, ddof=1)),
        "se_gauss_ann": np.sqrt(Q) * se_g,
        "se_moment_ann": np.sqrt(Q) * p["se_sr"],
        "sr_daily": sr_d,
        "sr_ann": ann,
        "ci_lo_gauss": np.sqrt(Q) * (sr_d - Z90 * se_g),
        "ci_hi_gauss": np.sqrt(Q) * (sr_d + Z90 * se_g),
        "ci_lo_boot_basic": 2.0 * ann - hi_b,
        "ci_hi_boot_basic": 2.0 * ann - lo_b,
        "ci_lo_boot_pct": lo_b,
        "ci_hi_boot_pct": hi_b,
        "boot_median": mid_b,
        "ci_lo_moment": np.sqrt(Q) * (sr_d - Z90 * p["se_sr"]),
        "ci_hi_moment": np.sqrt(Q) * (sr_d + Z90 * p["se_sr"]),
        "se_gauss": se_g,
        "se_moment": p["se_sr"],
        "skew_daily": p["skew"],
        "kurt_daily": p["kurtosis"],
    }


def window_sensitivity(df: pl.DataFrame) -> list[tuple[str, int, float]]:
    """Annualized Sharpe as a function of where the window ENDS.

    SR_ann = sqrt(365/T) * sum(x) / sqrt(sum(x^2) - sum(x)^2/T), so padding the series with idle
    days shrinks it roughly as 1/sqrt(T). That matters enormously here: the strategy's last kept
    trade is 2024-09-06, so ~34% of the published window is a period in which the signal was
    switched off. The published number is neither the best nor the worst point on this curve -- the
    window was pre-registered, which is what makes it legitimate -- but a reader who is not shown
    this curve cannot tell that the headline is a function of an arbitrary boundary."""
    # CLOSE time, not decision time: daily_pnl books a trade on decision+5min, so a late-night
    # decision closes on the next date. Using the decision max would put that close outside the
    # calendar range and silently drop the trade (the left-join behaviour pinned in the tests).
    last = df["decision_time"].dt.offset_by("5m").max()
    ends = [(f"to last trade ({last:%Y-%m-%d})", last + timedelta(days=1)),
            ("AS PUBLISHED (2024-10-14)", TEST_END),
            ("+ 6 more idle months", TEST_START + timedelta(days=294))]
    out = []
    for lbl, end in ends:
        d = daily_pnl(df, TEST_START, end)
        sd = float(d.std(ddof=1))
        out.append((lbl, len(d), np.sqrt(Q) * float(d.mean()) / sd if sd > 0 else float("nan")))
    return out


def concentration(daily: np.ndarray) -> dict[str, float]:
    """How much of the result is one day. The Sharpe of a series dominated by a single observation
    is not a property of a process, it is a property of that observation.

    The counterfactual ZEROES the top day rather than deleting it: deleting would shorten the
    window to 111 days and so mix the concentration effect with the length effect that
    `window_sensitivity` exists to isolate. Zeroing also matches the driver's stated
    'idle days = 0' convention -- a day the strategy did not trade is a zero-return day, not an
    absent one."""
    total = float(daily.sum())
    order = np.argsort(daily)
    top = int(order[-1])
    without = daily.copy()
    without[top] = 0.0
    sd = float(without.std(ddof=1))
    return {
        "top_day_idx": float(top),
        "top_day_usd": float(daily[top]),
        "top_day_share": float(daily[top]) / total if total else float("nan"),
        "top3_share": float(daily[order[-3:]].sum()) / total if total else float("nan"),
        "sr_ann_ex_top": np.sqrt(Q) * float(without.mean()) / sd if sd > 0 else float("nan"),
    }


F14_KISH_N_EFF = 16   # F14 §4's published Kish rung, used only as a cross-check target below.


def kish_n_eff(df: pl.DataFrame) -> float:
    """Kish effective sample size over episode clusters: n_eff = n^2 / sum(cluster_size^2).

    The most conservative rung of the ladder, and the one that penalises the 18-trade
    2024-08-05 episode hardest (a cluster of size k contributes k^2 to the denominator, so one
    big pile-on collapses n_eff far below the episode count). Computed with the repo's own
    `concurrency_analysis.effective_n` rather than quoted: it returns 15.96 here, which
    independently recovers F14's published 16 -- the same kind of cross-check as `psr0_episodes`
    recovering F14's 0.968."""
    return float(effective_n(assign_episodes(df, timedelta(0)))["n_eff"])


def psr_ladder(nb: np.ndarray, n_ep: int, n_traded: int,
               n_kish: float) -> list[tuple[str, float, float]]:
    """PSR(0) on the PER-TRADE series across every defensible effective-sample choice, coarsest
    last. F14 §4's recommendation is to headline the traded-day basis (daily independence is the
    one thing actually verified) and use episodes as the cross-check, so the whole ladder is
    printed rather than the single most flattering rung.

    Caveat carried in the printout: skew/kurtosis are always estimated from all n trades while
    sqrt(n_eff - 1) uses the coarser count -- standard practice, but the moments are not
    independently re-estimated at the coarser unit."""
    return [(lbl, float(n), psr(nb, sr_star=0.0, n_eff=float(n))["psr"])
            for lbl, n in (("raw trades", len(nb)), ("episodes", n_ep),
                           ("traded days", n_traded), ("Kish", n_kish))]


def basis_row(label: str, df: pl.DataFrame, peak_cap: float,
              n_boot: int, seed: int, is_realized: bool = False) -> dict[str, object]:
    """One row of the sensitivity table. `is_realized` marks the realized-sweep path, the only
    basis for which a mark-to-market drawdown exists (see `ret_over_mtm_dd` below)."""
    nb = net_bps_of(df)
    hac = hac_stats(nb)
    daily = daily_pnl(df, TEST_START, TEST_END)
    s = sharpe_stats(daily, n_boot, seed)
    dd = daily_max_dd(daily)
    total = float(daily.sum())
    return {
        "cost_basis": label,
        "n_trades": len(df),
        "active_days": int(np.count_nonzero(daily)),
        "cal_days": len(daily),
        "mean_net_bps": hac["mean_bps"],
        "t_nw": hac["t_nw"],
        "ir_per_trade": float(nb.mean() / nb.std(ddof=1)) if len(nb) > 1 else float("nan"),
        "sr_daily": s["sr_daily"],
        "sr_ann": s["sr_ann"],
        "ci90_gauss_lo": s["ci_lo_gauss"],
        "ci90_gauss_hi": s["ci_hi_gauss"],
        "ci90_boot_lo": s["ci_lo_boot_basic"],
        "ci90_boot_hi": s["ci_hi_boot_basic"],
        # PSR(0) on the PER-TRADE series at n_eff = episodes (F14's basis). The daily-series PSR is
        # degenerate here (zero-inflated moments -> 1.000 for every basis), so it is not reported.
        "psr0_episodes": psr(nb, sr_star=0.0, n_eff=float(n_episodes(df)))["psr"],
        "total_net_usd": total,
        # NAMED for the drawdown definition they use. The daily-equity DD is the SHALLOWEST of the
        # three (blind to intraday troughs), so `ret_over_dd_daily` is ~30x the ratio computed
        # against the honest mark-to-market DD -- hence the explicit column name and the
        # `ret_over_mtm_dd` companion, which is the one to read.
        "dd_daily_usd": dd,
        "dd_daily_pct_peak_cap": dd / peak_cap * 100.0 if peak_cap else float("nan"),
        "ret_over_dd_daily": total / abs(dd) if dd < 0 else float("nan"),
        # ONLY defined for the realized-fill path. The MtM drawdown marks an actual mid path; a
        # per-trade cost offset (the floor and participation bases) does not define one, so there
        # is no MtM curve to divide by. Imputing the realized -$22,770 to them would not merely be
        # unknown, it would be FLATTERING: writing equity_floor(t) = equity_real(t) - c*N(t) with
        # c = (2s + fee) * notional and N non-decreasing gives
        #     DD_floor <= DD_real - c*(N(trough) - N(peak)),
        # i.e. strictly deeper whenever a trade is entered between the peak and the trough -- and
        # 20 trades were entered on 2024-08-05, the day carrying 86% of the drawdown. So the
        # imputed ratio is an upper bound, ~3-7% optimistic. NaN is the honest entry.
        "ret_over_mtm_dd": total / abs(FROZEN_MTM_DD_USD) if is_realized else float("nan"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def headline_block(df: pl.DataFrame, peak_cap: float, peak_n: int,
                   n_boot: int, seed: int) -> None:
    """Everything that qualifies the headline number: effective-sample sensitivity, the
    annualization check, the collateral cash-drag counterfactual, and the drawdown."""
    daily = daily_pnl(df, TEST_START, TEST_END)
    nb = net_bps_of(df)
    s = sharpe_stats(daily, n_boot, seed)
    active = int(np.count_nonzero(daily))

    print("\n" + "=" * 78)
    print(f"PRIMARY COST BASIS -- conservative fills (s={HEADLINE_S_BPS:.0f}bps/side)")
    print("=" * 78)
    print(f"  window                2024-06-25 -> 2024-10-14 ({len(daily)} calendar days, "
          f"{active} active)")
    print(f"  trades                {len(df)}   ({n_episodes(df)} independent episodes)")
    print(f"  mean net              {nb.mean():+.2f} bps/trade   per-trade IR {nb.mean()/nb.std(ddof=1):+.3f}")
    print(f"  SR_daily              {s['sr_daily']:+.4f}")
    print(f"  ANNUALIZED SHARPE     {s['sr_ann']:+.2f}   (x sqrt(365), rf = 0, idle days = 0)")
    # The disqualification travels WITH the number. Anyone who screenshots this block, greps for
    # "SHARPE", or reads the CSV must hit it here, not 90 lines down in the closing section.
    print("     *** DO NOT QUOTE THIS AS A HEADLINE SHARPE -- see (1)-(4) below and the closing")
    print("         section. Quote the per-trade edge, PSR(0) and the drawdown instead. ***")
    print(f"  MtM max drawdown      ${FROZEN_MTM_DD_USD:,.0f} = "
          f"{FROZEN_MTM_DD_USD/peak_cap*100:+.2f}% of ${peak_cap:,.0f} peak capital")
    print("                        (realized-fill path; see the ladder below)")

    # --- what the point estimate is actually made of. This comes FIRST, before the intervals,
    # --- because these two facts limit the number more than any confidence interval does.
    print("\n  -- (1) the number is a function of where the window ENDS --")
    for lbl, t, sr in window_sensitivity(df):
        print(f"     {lbl:<32} T={t:3d} days   ann Sharpe {sr:+.2f}")
    print("     The strategy's last kept trade is 2024-09-06: the trim gated out every later")
    print("     event (Sep 7/51 kept, Oct 0/30), so ~34% of the published window is a stretch in")
    print("     which the signal was switched off. The window was PRE-REGISTERED, so this is not")
    print("     cherry-picking -- but the headline is still a boundary artifact to this degree.")

    c = concentration(daily)
    print("\n  -- (2) the number is dominated by a single day --")
    print(f"     largest day is {c['top_day_share']*100:.0f}% of total P&L; top 3 days are "
          f"{c['top3_share']*100:.0f}%")
    print(f"     drop that one day and the Sharpe goes {s['sr_ann']:+.2f} -> "
          f"{c['sr_ann_ex_top']:+.2f}")
    print("     (2024-08-05, the one-directional cascade the book pyramided ~18x long into; per")
    print("      F14 it is also 86% of the mark-to-market drawdown. It supplies a quarter of the")
    print("      edge AND most of the variance, so it pushes the Sharpe DOWN while making it up.)")

    # --- the interval, by three methods that disagree; the disagreement IS the finding ---
    print("\n  -- (3) 90% confidence intervals: three methods, no consensus --")
    print(f"     bootstrap, basic   [{s['ci_lo_boot_basic']:+.2f}, {s['ci_hi_boot_basic']:+.2f}]   "
          f"<- PRIMARY: resamples the observed zero-inflated structure")
    print(f"     Gaussian SE        [{s['ci_lo_gauss']:+.2f}, {s['ci_hi_gauss']:+.2f}]   "
          f"asserts normality; ann SE {s['se_gauss_ann']:.2f}")
    print(f"     moment-corrected   [{s['ci_lo_moment']:+.2f}, {s['ci_hi_moment']:+.2f}]   "
          f"ann SE {s['se_moment_ann']:.2f}; skew {s['skew_daily']:+.2f} / kurt "
          f"{s['kurt_daily']:.1f} from {len(daily)-active}/{len(daily)} exact zeros")
    print(f"     empirical sampling SD of the annualized Sharpe (bootstrap) = {s['boot_sd']:.2f}")
    print(f"       -> the Gaussian SE ({s['se_gauss_ann']:.2f}) is ~"
          f"{s['se_gauss_ann']/s['boot_sd']:.1f}x too WIDE; the moment SE "
          f"({s['se_moment_ann']:.2f}) is ~{s['se_moment_ann']/s['boot_sd']:.1f}x too NARROW.")
    print("       Neither closed form is quotable: one asserts moments the sample contradicts,")
    print("       the other trusts moments the zero-inflation makes meaningless.")
    print(f"     (raw percentile interval [{s['ci_lo_boot_pct']:+.2f}, {s['ci_hi_boot_pct']:+.2f}] "
          f"is centred {s['boot_median']-s['sr_ann']:+.2f} too high -- not quoted.)")

    # --- effective sample: the intervals above all assume 112 independent observations ---
    print("\n  -- (4) effective sample: every interval above assumes 112 independent days --")
    for lbl, n_used in (("all calendar days", len(daily)), ("traded days only ", active)):
        se = gaussian_sharpe_se(s["sr_daily"], n_used)
        lo_i, hi_i = np.sqrt(Q) * (s["sr_daily"] - Z90 * se), np.sqrt(Q) * (s["sr_daily"] + Z90 * se)
        flag = "   <- SPANS ZERO" if lo_i < 0 else ""
        print(f"     daily, {lbl}  n {n_used:4d}   ann CI [{lo_i:+.2f}, {hi_i:+.2f}]{flag}")
    for lbl, n_used, p0 in psr_ladder(nb, n_episodes(df), active, kish_n_eff(df)):
        print(f"     per-trade PSR(0), {lbl:<12} n {n_used:4.0f}   {p0:.3f}"
              + ("   <- F14's recommended basis" if lbl == "traded days" else ""))
    print("     On the traded-day sample -- the one whose independence is actually verified, and")
    print("     F14's own recommendation -- the 90% interval on the Sharpe INCLUDES ZERO and")
    print("     PSR(0) drops below the 0.95 line. That is the honest resolution of this exercise:")
    print("     the per-trade edge is significant, the annualized Sharpe MAGNITUDE is not")
    print("     estimable from 22 active days.")
    print("     (Ladder caveat: skew/kurtosis come from all n trades while sqrt(n_eff-1) uses the")
    print("      coarser count -- standard, but the moments are not re-estimated at that unit.)")

    # --- annualization: is sqrt(365) the right factor? ---
    print(f"\n  -- (5) is sqrt(365) the right annualizer? Lo (2002) eta vs sqrt(365) = "
          f"{np.sqrt(Q):.2f} --")
    for L in LO_MAX_LAGS:
        r = lo_annualized_sharpe(daily, q=Q, max_lag=L)
        print(f"     max_lag {L:2d}:  rho1 {r['rho1']:+.3f}   eta {r['eta']:6.2f}   "
              f"SR_ann_Lo {r['sr_ann_lo']:+.2f}   (deflation {r['deflation']:.3f})")
    print("     rho1 ~ 0, but this statistic HAS NO POWER on a series that is 80% zeros: under")
    print("     the observed idle-day mask, a true daily AR(1) of 0.5 still measures rho1 ~ +0.07")
    print("     (5-95%: -0.05..+0.25), so the observed -0.014 is consistent with substantial true")
    print("     autocorrelation. Read this as 'Lo's correction is uninformative here', NOT as")
    print("     'independence confirmed'. eta also swings 17.9-23.9 across truncations, i.e. the")
    print("     annualizer itself is uncertain by ~25% before any sampling error (F14).")

    # --- the collateral cash-drag counterfactual (NOT an rf correction) ---
    rf_daily = peak_cap * RF_ANNUAL / Q
    excess = daily - rf_daily
    sd_e = float(excess.std(ddof=1))
    sr_rf = np.sqrt(Q) * float(excess.mean()) / sd_e if sd_e > 0 else float("nan")
    print(f"\n  -- (6) collateral cash-drag counterfactual --")
    print(f"     rf = 0 is not a shortcut: a futures book is self-financing, so P&L over notional")
    print(f"     IS an excess return when collateral sits in bills. The row below is the DIFFERENT")
    print(f"     question 'what if the ${peak_cap:,.0f} peak base earned nothing?' -- a cash-drag")
    print(f"     counterfactual, not a risk-free correction, and not to be averaged with the above.")
    print(f"     charging {RF_ANNUAL*100:.2f}%/yr on that base costs ${rf_daily*len(daily):,.0f} "
          f"over the window -> ann Sharpe {sr_rf:+.2f}  (vs {s['sr_ann']:+.2f})")


def drawdown_ladder(oos: pl.DataFrame, mtm_usd: float | None, mtm_source: str,
                    peak_cap: float, peak_n: int) -> None:
    """Three drawdown definitions on the realized-fill path, coarsest first. Reported together
    because they differ by ~3.5x and quoting only the shallowest was the audit's finding."""
    daily = daily_pnl(oos, TEST_START, TEST_END)
    print("\n" + "=" * 78)
    print("DRAWDOWN LADDER -- realized-sweep path, $50k/trade")
    print("=" * 78)
    print(f"  base: {peak_n} peak concurrent positions x $50k = ${peak_cap:,.0f}")
    for lbl, val, note in (
        ("daily equity", daily_max_dd(daily), "coarsest: blind to intraday troughs"),
        ("realized-close, per trade", realized_close_dd(oos), "books P&L only when a leg closes"),
        ("MARK-TO-MARKET", mtm_usd, mtm_source),
    ):
        if val is None:
            continue
        print(f"  {lbl:<26} ${val:>10,.0f}   {val/peak_cap*100:+6.2f}% of peak capital   {note}")
    print("\n  The MtM figure is the honest one: it marks the up-to-18 simultaneously open legs,")
    print("  which the other two cannot see. It is computed on the REALIZED-fill path only -- a")
    print("  per-trade cost offset (the floor bases) does not define a marked path, so no MtM")
    print("  drawdown is imputed to them.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtm", action="store_true",
                    help="recompute the mark-to-market drawdown from the book (slow: full book pass)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    oos = load(OOS_CSV)
    stress = load(STRESS_CSV) if STRESS_CSV.exists() else None
    if stress is None:
        _eprint(f"({STRESS_CSV} not found -- participation rows skipped)")

    peak_n, peak_cap = peak_capital(oos, NOTIONAL)

    rows = [basis_row(lbl, df, peak_cap, args.n_boot, args.seed,
                      is_realized=lbl.startswith("realized-sweep"))
            for lbl, df in cost_bases(oos, stress)]
    table = pl.DataFrame(rows)

    # ---- regression guard: the realized-fill row MUST reproduce the frozen F10 figures. It is
    # the control proving this driver restates the frozen result rather than re-deriving it.
    ctl = next(r for r in rows if r["cost_basis"].startswith("realized-sweep"))
    assert abs(ctl["mean_net_bps"] - FROZEN_NET_BPS) < 0.05, \
        f"control mean {ctl['mean_net_bps']:.2f} != frozen {FROZEN_NET_BPS}"
    assert abs(ctl["ir_per_trade"] - FROZEN_IR_TRADE) < 0.005, \
        f"control IR {ctl['ir_per_trade']:.3f} != frozen {FROZEN_IR_TRADE}"
    assert abs(ctl["sr_ann"] - FROZEN_ANN_SHARPE) < 0.05, \
        f"control ann Sharpe {ctl['sr_ann']:.2f} != frozen {FROZEN_ANN_SHARPE}"
    _eprint(f"[guard OK] realized-fill control reproduces F10: {ctl['mean_net_bps']:+.2f} bps, "
            f"IR {ctl['ir_per_trade']:+.3f}, ann Sharpe {ctl['sr_ann']:+.2f}")

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_float_precision(3)

    print("\n" + "#" * 78)
    print("# OUT-OF-SAMPLE SHARPE -- frozen trimmed 5-min tail reversion, BTCUSD_PERP")
    print("# reserved test window 2024-06-25 -> 2024-10-14. No new backtests, nothing re-fitted.")
    print("#" * 78)

    headline_block(floor_frame(oos, HEADLINE_S_BPS), peak_cap, peak_n, args.n_boot, args.seed)

    print("\n" + "=" * 78)
    print("SENSITIVITY -- the same Sharpe under every other fill assumption")
    print("=" * 78)
    print("sr_ann = sqrt(365) * SR_daily on daily P&L, all calendar days, rf = 0.")
    print("ci90_boot = PRIMARY (basic/centring-corrected blocks); ci90_gauss = normality-assuming")
    print("reference, empirically ~1.7x too wide. psr0_episodes = PSR(0) on per-trade returns at")
    print("n_eff = episodes; the more conservative traded-day rung is in the headline ladder.")
    print("Sharpe is scale-invariant: the $900k base moves the dd columns, never sr_ann.")
    print("Every caveat in the headline block (window boundary, one-day dominance, effective")
    print("sample) applies to EVERY row here -- the rows differ only in fill assumption.\n")
    print(table.select("cost_basis", "n_trades", "active_days", "mean_net_bps", "t_nw",
                       "ir_per_trade", "sr_ann", "ci90_boot_lo", "ci90_boot_hi",
                       "ci90_gauss_lo", "ci90_gauss_hi", "psr0_episodes"))
    print(table.select("cost_basis", "total_net_usd", "dd_daily_usd", "dd_daily_pct_peak_cap",
                       "ret_over_dd_daily", "ret_over_mtm_dd"))
    print("ret_over_dd_daily uses the SHALLOWEST drawdown definition and is ~30x flattering.")
    print("ret_over_mtm_dd is the one to read, and is defined ONLY for the realized-fill row --")
    print("a per-trade cost offset does not define a marked path, so the other rows are null")
    print("rather than borrowing the realized -$22,770 (which would be a flattering upper bound).")

    mtm_usd, mtm_source = FROZEN_MTM_DD_USD, "quoted from OOS_RISK_AUDIT (run --mtm to recompute)"
    if args.mtm:
        from backtester import BookProvider
        from backtest_oos_risk import mtm_max_drawdown
        _eprint("marking the book for the MtM drawdown (slow) ...")
        dd = mtm_max_drawdown(oos, BookProvider())
        total = float(oos["net_pnl"].sum())
        assert abs(dd["final_equity_usd"] - total) < 1.0, "MtM curve integrity"
        mtm_usd = dd["max_dd_usd"]
        mtm_source = f"recomputed, {int(dd['n_ticks']):,} book ticks marked"
        assert abs(mtm_usd - FROZEN_MTM_DD_USD) < 50.0, \
            f"MtM DD {mtm_usd:,.0f} drifted from the audit's {FROZEN_MTM_DD_USD:,.0f}"

    drawdown_ladder(oos, mtm_usd, mtm_source, peak_cap, peak_n)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    # Both interpolated from the computed values -- see the desync note below.
    head_df = floor_frame(oos, HEADLINE_S_BPS)
    win = window_sensitivity(head_df)
    head_daily = daily_pnl(head_df, TEST_START, TEST_END)
    # EXCLUDES the raw-trades rung: F14 says "never raw trades" (it treats 105 overlapping trades
    # as independent), so letting its 0.999 set the top of the quoted range would advertise the
    # one rung the report disclaims. Range spans episodes / traded days / Kish.
    psr_range = [p for lbl, _, p in psr_ladder(net_bps_of(head_df), n_episodes(head_df),
                                               int(np.count_nonzero(head_daily)),
                                               kish_n_eff(head_df))
                 if lbl != "raw trades"]
    print("  * DO NOT QUOTE THE ANNUALIZED SHARPE AS A HEADLINE. It is reported here because it")
    print("    was asked for and because the conventions behind it deserve to be written down,")
    print("    not because it is estimable from this sample. Four independent reasons:")
    # Interpolated from window_sensitivity, never hard-coded: a literal here silently desynced
    # from the computed table once already (it read 2.27, a figure for a window length the code
    # does not run).
    print("      - it moves " + " -> ".join(f"{sr:.2f}" for _, _, sr in win)
          + " purely with the window end date (T = "
          + ", ".join(str(t) for _, t, _ in win) + " days);")
    print("      - one day (2024-08-05) is ~39% of the P&L;")
    print("      - on the traded-day sample its 90% interval INCLUDES ZERO;")
    print("      - the three CI methods disagree by more than the point estimate is worth.")
    print("  * QUOTE INSTEAD, all on the conservative floor basis: the per-trade edge")
    print(f"    (+41.13 bps, Newey-West t 2.18), PSR(0) {min(psr_range):.2f}-{max(psr_range):.2f} "
          f"across the DEFENSIBLE effective-sample")
    print("    choices (episodes / traded days / Kish -- never the raw-trade rung, which treats")
    print("    105 overlapping trades as independent),")
    print("    and the -$22,770 / -2.53% mark-to-market drawdown. Those survive every correction")
    print("    applied above. This is also exactly where F14 landed -- 'report significance, not")
    print("    the flattered Sharpe magnitude' -- and this driver does not overturn it.")
    print("  * One asset, one 3.7-month regime, n=105 cascades, 22 active days. One clean OOS.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_CSV)
    _eprint(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
