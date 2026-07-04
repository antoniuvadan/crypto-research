# Concurrency Analysis — should the reversion algo ever hold 18 positions at once?

**Date:** 2026-07-03
**Question:** why does the frozen liquidation-cascade reversion strategy reach a peak of **18
simultaneously-open positions**, and should it? Analysed with deliberate skepticism (the case
for allowing burst concentration *and* the case against), statistically, over both the reserved
OOS window and the full training window.
**Artifacts:** `strategies/concurrency_analysis.py` (battery), `strategies/gen_train_trades.py`
(train trade set), `tests/test_concurrency_analysis.py` (20 tests). Reuses the prior audit's
`mtm_max_drawdown` / `realized_close_dd` (`backtest_oos_risk.py`), `peak_capital`
(`backtest_oos_test.py`), `hac_stats` (`significance.py`). Reviewed adversarially; findings folded in.

---

## TL;DR — verdict

**No. The algo should not run to 18 concurrent.** The 18-stack is **one-directional** (never
hedged — 100% of open time), **redundant** (the positions in a cascade are the same signal
fired repeatedly, not independent bets), and it concentrates risk catastrophically: **86% of
the entire out-of-sample drawdown comes from a single day** (2024-08-05, the global risk-off
crash), into which the strategy pyramided **18× long**. Crucially, the stacking **does not add
per-trade edge** — the edge lives in the *most-violent* event of each cascade, which a single
position already captures. **Recommendation: cap concurrency and select one entry per cascade
by displacement, then re-size up** — with an explicit causal caveat: the idealised "keep the
most-violent event" result is an oracle upper bound, not directly deployable.

---

## 1. Why it reaches 18 — the mechanism

Concurrency = **event clustering × fixed 5-min hold × no position management**. The engine
(`run_liquidation_momentum_model_c_backtests`, `backtest_momentum.py`) opens a fresh
fixed-notional 5-min round-trip on **every** ≥98th-pct tail event that passes the trim, with
**no cap, no netting, no cooldown, no size decay** — "concurrency" is never enforced, only
measured after the fact by `peak_capital`. A liquidation cascade emits many tail events within
minutes; during a one-way cascade they are all the same direction; each opens a 5-min position
→ they stack. The OOS peak of 18 is **2024-08-05 01:10 UTC**; the train peak of 18 is
**2023-10-01**. In **both** windows, whenever the book is open it is **100% one-directional**
(duration-weighted) — the strategy is never internally hedged; concurrency is pure directional
leverage, not diversification.

## 2. Is the "18" statistical double-counting? Mostly no — and this is where I correct a naïve prior

Clustering trades into **episodes** (chains whose 5-min holds overlap; robust to a gap
tolerance τ ∈ {0, 60s, 300s}) gives a Kish effective sample size:

| Set | trades | episodes | Kish N_eff | largest episode |
|---|---|---|---|---|
| OOS | 105 | 29 | 16.0 | 18 |
| Train | 503 | ~119 | ~73 | 18–24 |

It is tempting to say "n=105 is really ~16, so the edge is fragile." **That is wrong, and the
skeptical read cuts the other way.** Kish N_eff assumes within-cluster P&L correlation ρ=1; it
is a *worst-case floor*, not the effective sample. **Directional overlap (100% one-way) is not
the same as P&L correlation** — the trades stacked in a cascade win and lose fairly
independently. The honest dependence-robust estimators of the OOS edge:

| estimator | t (OOS) | t (train) |
|---|---|---|
| trade-level IID | +4.78 | +8.64 |
| cluster-by-episode | **+4.69** | +4.09 |
| cluster-by-day | **+4.89** | +4.81 |
| Newey-West (conservative headline) | +2.62 | +4.74 |
| episode block-bootstrap 90% CI (mean bps) | [+31, +64], frac≤0 = 0.000 | [+24, +56], 0.000 |

Cluster-robust t (4.69) ≈ IID t (4.78) and is **stronger** than the conservative Newey-West
2.62; resampling whole episodes still rejects ≤0. So the concurrency does **not** cripple the
sample. **The problem with concurrency is not statistical significance — it is risk
concentration and capital.** (Newey-West stays the conservative headline; its auto lag L=4 is
below the 18-trade max episode span, so it cannot fully absorb the longest cluster — another
reason not to lean on it alone.)

## 3. Risk and P&L are concentrated in a handful of cascades

| Set | active days | top day | top-day P&L share | top-3 share |
|---|---|---|---|---|
| OOS | 22 | 2024-08-05 | **36%** ($9,299 / $25,647) | 54% |
| Train | 85 | 2024-01-09 | 15% | 40% |

**Drawdown attribution (OOS, mark-to-market):** full **−$22,770**; **excluding 2024-08-05 →
−$3,277**. The Aug-5 crash accounts for **−$19,493 = 86%** of the entire drawdown (the collapse
on exclusion confirms Aug-5 is the drawdown-driving day, not merely the top-P&L day). The
−2.53% headline risk figure is, to first order, *one day of pyramided long exposure into a
crash* — and the −$22,770 itself assumes the reversion eventually arrives, a fat-tailed bet.

**But the edge is not only that day.** Leave-one-day-out: dropping Aug-5 moves the mean edge
from +48.85 to **+38.46 bps**, with **cluster-t +4.67** (and trade-level t_NW rises to +5.37,
since the overlapping Aug-5 trades had been adding serially-correlated noise NW penalised). The
strategy is still positive and significant without its worst-risk day — concurrency is where
the *risk* concentrates, not where all the *edge* lives.

## 4. Counterfactual position caps — the policy evidence

Applied as a **pure filter on the already-realized frozen trades** (no new fills; the spent OOS
holdout is untouched — a risk overlay, not re-tuning). Sharpe is the prior audit's
known-flattered √365 daily convention (relative cross-policy signal only); drawdown is
mark-to-market (OOS).

**OOS:**

| Policy | trades | net bps | t_NW | total $ | peak cap | Sharpe | MtM DD | MtM DD % |
|---|---|---|---|---|---|---|---|---|
| uncapped (N=∞) | 105 | 48.85 | 2.62 | 25,647 | $900k | 4.46 | −22,770 | −2.53% |
| cap N=5 (greedy) | 87 | 22.53 | 1.53 | 9,801 | $250k | 2.64 | −11,034 | −4.41% |
| cap N=3 (greedy) | 68 | 20.42 | 1.33 | 6,943 | $150k | 2.29 | −7,882 | −5.25% |
| cap N=2 (greedy) | 53 | 22.90 | 1.54 | 6,069 | $100k | 2.79 | −5,740 | −5.74% |
| cap N=1 (greedy, keep FIRST) | 29 | 22.91 | 1.48 | 3,323 | $50k | 2.84 | −2,915 | −5.83% |
| **1-per-episode ORACLE (keep MOST-VIOLENT)** | 29 | **50.55** | **5.68** | 7,329 | $50k | 5.61 | **−1,270** | −2.54% |

Train reproduces the pattern: uncapped +39.75/t_NW 4.74; greedy cap-N=1 +16.5/t_NW 2.12;
oracle 1-per-episode +45.99/t_NW 4.76, both at $50k peak capital.

Two things jump out:
- **Greedy caps destroy the edge** (t_NW falls below 2). Keeping the *first* event of a cluster
  keeps the *wrong* one — reversion edge builds as the cascade intensifies, so the early event
  is the low-edge one.
- **Selecting the most-violent event per cascade improves everything at once**: per-trade edge
  +50.55 (> the uncapped 48.85), t_NW 5.68 (>> 2.62), **1/18th the deployed capital** ($50k vs
  $900k), and MtM drawdown −$1,270 vs −$22,770 (18× less dollar risk). Stacking the other 76
  trades added capital and one-way crash risk to capture *the same signal* repeatedly.

### The causal caveat — do not bank the +50.55

"1-per-episode keeping the max-displacement trade" **peeks at the whole cascade to pick its
most-violent event** — it is an **oracle upper bound, not a live-deployable result** (verified:
in 50% of multi-trade episodes the max-displacement trade is *not* the first arrival, so
picking it requires seeing future trades). The honest **achievable band** is bracketed by:
- **greedy-first (causal): +22.91 bps, t_NW 1.48** — take the first, block for the hold; and
- **oracle-max (not causal): +50.55 bps, t_NW 5.68**.

A live rule (e.g. a **displacement threshold** estimated on train — enter only on the first
event whose displacement clears a bar, then lock out) lands between the two, **likely nearer
+22.91**, and must be developed and validated on **fresh data** (the OOS test set is spent). The
gap between the brackets is itself the finding that *edge rises with cascade progression*.

## 5. Skeptic's both-sides ledger

**Reasons to ALLOW burst concentration:**
- The most-violent cascade events carry the highest per-trade edge; the uncapped 48.85 bps is
  *above* every causal single-cap variant precisely because it includes those violent trades.
- Fixed per-trade sizing is scale-free; "18 × $50k" is a capital-deployment choice, not
  intrinsic to the alpha; and the edge's *significance* survives the overlap (§2).

**Reasons NOT to (stronger, and they win):**
- The stack is **one-directional and directionally perfectly correlated** — leverage, not
  diversification. "18 positions" is one bet sized 18×.
- **86% of the OOS drawdown is one crash day**; the strategy pyramids **long into a crash** that
  could keep running — the −$22,770 assumes the reversion arrives (fat-tailed).
- The redundant stacked trades **do not add per-trade edge** — the edge is in the violent event,
  captured by one position; stacking only adds capital and tail risk.
- Concurrency inflates the **capital base** the ROI is quoted on ($900k peak vs a realistic
  position-limited book) — capital efficiency is overstated.

## 6. Recommendation

1. **Cap concurrency.** There is no diversification benefit to a one-way stack; a hard limit
   (target 1 position per cascade, at most N ≤ 3) removes the crash-pyramiding tail with no loss
   of per-trade edge — *provided the right trade is selected*.
2. **Select by displacement, not arrival.** Greedy first-come throws the edge away; the winner
   is the most-violent event per cascade. Because that selection is not causal as-measured,
   **develop a causal displacement-gated entry** (threshold / cooldown-with-floor) and validate
   it on **new data** before trusting anything above the causal ~+23 bps floor.
3. **Re-size, don't shrink.** With capital freed ~18×, size each retained (scale-free ~+23–50
   bps) trade up to redeploy — capturing the edge at a fraction of the concentration risk.
4. **Re-report Sharpe + max drawdown under the cap** (per CLAUDE.md rigor); the uncapped
   headline overstates capital efficiency and hides that its risk is essentially one day.

**Sensitivity:** every conclusion above is stable across the episode gap τ ∈ {0, 60s, 300s} and
holds in both the OOS and training windows.
