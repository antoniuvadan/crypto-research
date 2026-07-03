# OOS Reversion Result — Skeptical Risk Audit

**Date:** 2026-07-03
**Scope:** audit the frozen mean-reversion strategy's out-of-sample result (research.md
**Finding 10**, reserved test window **2024-06-25 → 2024-10-14**, filtered net **+48.85 bps**,
**t_NW 2.62**, n=105) for lookahead bias / improper alpha methodology that could inflate
performance, then report the true maximum drawdown.
**Artifacts:** `strategies/backtest_oos_risk.py` (MtM drawdown + honest Sharpe + slippage
stress), `tests/test_oos_risk.py` (12 tests), `participation` param on
`_sweep_fill_from_agg_trades` (`backtester.py`) threaded through the Model-C runner
(`backtest_momentum.py`), `data/results/oos_test_slippage_stress_trades.csv`.

---

## TL;DR

1. **No lookahead bias.** The causal path is clean end-to-end (verified by direct reading
   plus two independent search passes). The engine never consumes data timestamped after a
   decision. The overstatement risk is **optimistic fills + statistical framing**, not
   time-travel.
2. **The "2.5 Sharpe" does not exist.** Reported figures are per-trade Sharpe **0.467** and
   annualized **4.46**; the ~2.5 is the Newey-West **t-stat (2.62)**, a significance stat.
3. **Max drawdown was understated ~3.5×.** The reported "shallow" realized-close −$6,571
   (−0.73% of peak capital) ignores concurrent open positions. The true **mark-to-market
   drawdown is −$22,770 = −2.53% of peak capital** (size-invariant).
4. **The edge is robust.** Under a 10× participation squeeze on fills it still returns
   **+46.8 bps, t_NW 2.54** — not a liquidity-capture artifact. But it rests on n=105
   cascades in one 3.7-month window: one clean OOS, not a proven Sharpe engine.

---

## 1. Deliverable: maximum drawdown over the OOS period

| Definition | $ ($50k/trade) | $ ($100k/trade) | **% of peak capital** |
|---|---|---|---|
| Realized-close (frozen report's method) | −$6,571 | −$13,113 | **−0.73%** |
| **Mark-to-market (honest)** | **−$22,770** | **−$45,536** | **−2.53%** |

**Why the frozen number is wrong.** `max_drawdown` in `backtest_oos_test.py:218-221` is a
`cumsum` of per-trade `net_pnl` ordered by `decision_time` — it books P&L only when a
round-trip *closes* and never marks the **up-to-18 simultaneously-open positions** to
market. During a cascade that runs against the reversion book, the whole book is underwater
at once; the realized-close curve cannot see that.

**The correction.** `mtm_max_drawdown` (`backtest_oos_risk.py`) reconstructs a mark-to-mid
equity curve: realized cash on closed legs + unrealized MtM on every open leg, marked to the
L1 mid at **every book tick inside a busy span** (1.73M ticks). Within a constant open-set
segment, equity is linear in the mid — `equity = realized + A·mid − B` with
`A = Σ signed_qty·$100/entry_vwap`, `B = Σ signed_qty·$100` — so it is exact to L1 resolution.
Curve integrity is asserted (final equity == realized total = +$25,647).

**On the denominator.** Peak deployed capital = 18 concurrent positions × $50k = **$900k**,
the same base the frozen report used for ROI. The strategy is capital-inefficient and bursty
(only 22 of 112 days active), so against a tighter working-capital base — a position limit
well below 18, or average-deployed rather than peak — the drawdown **percentage would be
proportionally larger**. −2.53% is honest for the peak-capital convention, not a claim that a
realistically-sized book only draws down 2.5%.

---

## 2. Findings

### A. No hard lookahead — the causal path is clean

Verified that no decision consumes data timestamped after its decision instant:

- Signal fires at `liq_time + 5s`, exactly when the ±5s flow window closes; `seconds_after`
  is coupled to the window reach in every call site. `backtest_momentum.py:235, 275`
- Trailing-7d 98th-pct threshold uses `values[left:right]` — strictly-earlier events, current
  excluded. `backtester.py:404-419`
- Trailing-30d trim median uses only past displacements `d[lo:i]`. `backtest_oos_test.py:123-135`
- Displacement/trim inputs read book mids at/before `decision_time`. `backtest_oos_test.py:106-111`
- Entry sweep starts at `decision+300ms`, exit at `decision+hold+300ms`; both
  `searchsorted(side="left")`, so pre-decision trades are never consumed.
  `backtest_momentum.py:498, 528`; `backtester.py:516`
- `BookView.as_of` is a strict backward as-of (no forward peek). `backtester.py:722-727`

### B. Fill optimism — mostly immaterial in practice

- **B1 — 100% volume capture / no own-size impact.** The sweep takes `min(remaining,
  print_qty)` of every same-side aggressor print at its exact price, with zero impact and no
  queue competition. `backtester.py:523-539`. It *does* implicitly pay the spread (a taker buy
  matches only ask-side aggressor prints), so the round-trip pays ~full spread vs mid; it omits
  only impact from the order's own size. **Latency ≠ slippage:** the 300ms latency models
  *time drift* during the execution delay, not the cost of consuming liquidity. Slippage is
  stress-tested separately (§4 and the F9 flat-bps floor).
- **B2 — 5-min entry fill window.** Measured immaterial: control fills complete in **median
  114ms** (p99 1.3s), 100% complete.
- **B3 — unbounded exit window.** Measured immaterial: 100% exit completion.

### C. Statistical framing inflates the annualized Sharpe

- **C1 — annualized Sharpe 4.46 is a fragile convention** (`backtest_oos_test.py:195-215`):
  √365 (calendar) annualization on a **22-active-day / 112-day** bursty sample, and treating
  overlapping same-cascade positions as independent.
  **Self-correction:** I initially hypothesized the idle-day zero-padding *inflates* the
  number. The data shows the **opposite** — dropping zero days gives **+4.97 > +4.46**, so the
  padding slightly *deflated* it. The inflation is the annualization convention and the
  independence assumption, **not** the padding.
- **C2 — overlap/concentration.** Peak **18 concurrent** positions, long-biased (64/41), all
  riding the same cascade retracements; the √365 IID assumption is not valid.
- **C3 — single 3.7-month window, n=105.** t_NW 2.62 ≈ 1% significance; the point estimate has
  a wide CI. Not a "Sharpe-rich" strategy.
- **C4 (minor) — linear PnL on an inverse contract.** `_linear_contract_pnl_usd`
  (`backtester.py`) uses `exit/entry − 1`; a COIN-M inverse perp is `1/entry − 1/exit`.
  Negligible at 5-min bps scale.

### D. Max drawdown understated — see §1.

---

## 3. Honest Sharpe restatement

| Measure | Value | Note |
|---|---|---|
| **Per-trade IR (robust primitive)** | **+0.467** | mean +48.85 bps, **t_NW +2.62** (n=105) |
| Annualized, as-coded | +4.46 | √365 IID, 22/112 active days |
| Annualized, active-days-only | +4.97 | zero days dropped |
| Annualized, 90% bootstrap CI | **[+3.46, +7.18]**, median +4.88 | 7-day circular block bootstrap |

The bootstrap CI is the width of the **as-coded (flattered) estimator** and inherits its
upward bias — read it as "even generously framed, the point estimate is this uncertain."
**The honest anchor is the per-trade IR 0.467 / t_NW 2.62.** The annualized 4.46 is an
optimistic convention, not a robust number.

---

## 4. Participation-cap slippage stress

The one mechanism the backtest lacked: it captured 100% of every print. `participation < 1`
caps the fraction of each aggressor print the order may take, so fills spread over the tape,
VWAPs drift, and thin windows leave partial/missed trades. (This re-executes fills, so it
**re-touches the spent holdout** — reported as a stress, not a fresh OOS. It models competition
from *other* traders; the strategy's own overlapping legs still each sweep the full tape.)

| α (max % of each print) | trades | entry complete | median fill | **net bps** | **t_NW** | MtM DD |
|---|---|---|---|---|---|---|
| 1.0 (control) | 105 | 100% | 114ms | +48.85 | 2.62 | −$22,770 |
| 0.20 | 105 | 62% | 453ms | **+47.62** | 2.58 | −$22,766 |
| 0.10 | 105 | 60% | 717ms | **+46.84** | 2.54 | −$22,765 |

Even taking only **10% of each print** (leaving ~40% of trades partially filled) the edge
holds at **+46.8 bps, t_NW 2.54**, right beside the F9 flat-bps floor (~+39–43). Strong
evidence the edge is **not** a liquidity-capture artifact.

---

## 5. Verification

- `participation=1.0` is a bit-exact no-op; the driver asserts the control reproduces the
  frozen headline (+48.85 bps, realized DD −$6,571, total +$25,647). All held.
- MtM curve integrity asserted (final equity == realized total); MtM DD ≤ realized-close DD
  (guaranteed only under the constant 5-min holding period — commented in code).
- **`tests/test_oos_risk.py`: 12/12 passing** — participation no-op / VWAP degradation /
  incomplete-fill / sell-side symmetry; MtM worse-than-realized, no-phantom-DD, short-sign,
  mixed-offset netting, peak-then-trough (exercises `np.maximum.accumulate`),
  trough-before-higher-peak, partial-exit invariant, cross-midnight.
- **Adversarial review:** no ship-blocking bugs; the one real latent finding (an unsorted
  Newey-West input in the stress table) was fixed, and its six requested coverage cases added.

---

## 6. Bottom line

- **The edge survives scrutiny** — no lookahead, robust to a 10× participation squeeze,
  +48.85 bps net / t_NW 2.62 OOS — but on **n=105 cascades in one 3.7-month regime**.
- **The risk was understated:** true mark-to-market drawdown is **−2.53% of peak capital
  (−$22,770)**, ~3.5× the reported −$6,571, and larger still on a realistically position-limited
  book.
- **The annualized Sharpe (4.46) is a fragile convention, not a "Sharpe 2.5+" claim.** Quote
  the per-trade IR **0.467 / t_NW 2.62** instead.
