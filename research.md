# Research findings

Dated, self-contained findings for the liquidation-cascade study on BTCUSD_PERP
(Binance COIN-M perpetual). Each finding is its own section.

---

## Finding 1 — Model C P&L decomposition: the momentum signal is dead (and inverted)

**Date:** 2026-06-27
**Artifact:** `pnl_decomposition.py` (reads
`data/results/liquidation_momentum_model_c_trades.csv` + L1 `bookTicker` parquet;
writes `data/results/pnl_decomposition_summary.csv` /
`data/results/pnl_decomposition_trades.csv`)
**Scope:** training window 2023-06-25 → 2024-06-24, all Model C round trips
(n ≈ 969 per horizon), trade sizes $50k and $100k.

### Method

For every completed Model C round trip, net return (in bps, sign-corrected by
trade direction) was split into additive components:

```
net = gross_mid_to_mid          (signal: mid at decision -> mid at exit horizon)
      - latency_slippage        (mid drift during the 300ms decision->entry delay)
      - spread_entry            (entry fill vs mid at entry: spread + book-walk)
      - spread_exit             (exit fill vs mid at exit: spread + book-walk)
      - fee_entry               (5 bps taker)
      - fee_exit                (5 bps taker)
```

The split is exact in price space, so the bps terms — all divided by the same
per-trade denominator (`mid_decision`) — sum to the reconstructed net. Reference
mids come from the L1 `bookTicker` parquet, as-of joined (last quote at or before
each reference timestamp: `decision_time`, `entry_start_time`, `exit_start_time`).
Reference timestamps:

- `decision_time` = liquidation_time + 5s (signal fully observed)
- `entry_start_time` = decision_time + 300ms latency
- `exit_start_time` = decision_time + holding_period + 300ms latency

**Validation:** `net_reconstructed` matches the backtest's `net_realized` to 3
decimals across all 10 (size × horizon) cells, confirming the accounting.

### Results (mean bps per trade, sign-corrected; costs shown as positives subtracted)

| horizon | gross_mid_to_mid | latency_slip | spread_entry | spread_exit | fees (rt) | net |
|---|---|---|---|---|---|---|
| 5s   | **−3.23**  | −0.24 | 1.03 | 0.84 | 10.0 | −11.13 |
| 10s  | **−6.18**  | −0.24 | 1.06 | 1.13 | 10.0 | −13.75 |
| 30s  | **−8.59**  | −0.24 | 1.07 | 1.15 | 10.0 | −16.14 |
| 1min | **−14.58** | −0.24 | 1.07 | 0.97 | 10.0 | −22.31 |
| 2min | **−15.72** | −0.24 | 1.07 | 1.17 | 10.0 | −23.24 |

($100k is within ~0.05 bps of $50k at every horizon — market impact / book-walk
is negligible at these sizes; the book is deep relative to the trade.)

### Verdict: the "signal is dead" branch — and the signal is backwards

`gross_mid_to_mid` is **negative at every horizon, before any friction**, and
gets monotonically worse with holding period (−3.2 bps at 5s → −15.7 bps at
2min). This is the decision-tree's first bucket: no exit refinement saves it; the
problem is event selection or the hypothesis itself.

The **sign and shape are a specific, actionable clue, not just "dead."** Trading
in the *same* direction as the liquidating flow (momentum) loses on the pure mid
path, and the loss grows with horizon. That is the signature of **mean-reversion**:
after a liquidation cascade the mid systematically retraces *against* the flow,
more so the longer you wait. Consistent with the 2026-05-29 journal note that
price mean-reverts at the 1–2 min frequency. **The momentum hypothesis is inverted.**

### What this rules out / establishes

- **Latency is not the problem.** `latency_slippage` is −0.24 bps — a tiny
  *favorable* drift during the 300ms delay. We are not entering after the
  dislocation peak. (Rules out the latency-refinement path.)
- **Friction is small and boring.** Round-trip friction ≈ 12 bps, of which
  **10 bps is taker fees** and only ~2 bps is spread + impact. Execution is not
  where an edge is leaking — there is no edge to leak under the momentum sign.

### Implication for a reversion flip

Flipping to **mean-reversion** (trade against the flow) flips `gross_mid_to_mid`:
+3.2 bps (5s) → +15.7 bps (2min). Against ~12 bps friction:

- 5s / 10s / 30s reversion **still does not clear friction** (+3.2 to +8.6 gross < 12).
- Only **1–2 min** clears it, and thinly (2min: +15.7 − 12 ≈ **+3.7 bps**
  before any conservatism — well inside noise at n ≈ 969).

So the table dictates: flip the hypothesis to reversion (Step 7), but a naïve
flip is marginal. To become real it needs **maker entry to kill the 10 bps taker
fee (Step 2)** and/or **event selection to isolate events with larger reversion
(Step 3)** — e.g. conditioning on liquidation magnitude or volatility regime.

### Next steps

1. Flip to reversion and re-decompose to confirm the +bps directly and see which
   horizons clear friction net.
2. Step 3 — filter events (liquidation magnitude / vol regime) to find where
   reversion is strongest.
3. Step 2 — model maker entry to remove the dominant 10 bps round-trip fee.

---

## Finding 2 — Reversion flip: net-positive at 1–2 min, but thin and fee-bound

**Date:** 2026-06-27
**Artifacts:** `backtest_reversion.py` (reuses
`backtester.run_liquidation_momentum_model_c_backtests` with
`signal_direction_sign=-1`; writes `data/results/reversion_model_c_{summary,trades}.csv`)
and `pnl_decomposition.py --trades data/results/reversion_model_c_trades.csv
--label reversion_decomposition` (writes
`data/results/reversion_decomposition_{summary,trades}.csv`).
**Scope:** same training window and grid as Finding 1 — 2023-06-25 → 2024-06-24,
**identical 969-event set per horizon** (only the trade direction is flipped),
trade sizes $50k and $100k.

### Method

Flip the executed trade to trade *against* the liquidating flow
(`signal_direction_sign = -1`). The signal/event construction is unchanged, so
the same 969 events fire; entry/exit fills are re-simulated on the opposite side
of the tape (a reversion sell hits the bid where momentum bought the ask), and
`side`/`direction` in the trade rows record the executed trade. Re-ran the same
P&L decomposition.

### Results (mean bps per round trip, sign-corrected; $50k)

| horizon | momentum net (F1) | reversion net | reversion gross_mid | net Sharpe | ≈ t-stat |
|---|---|---|---|---|---|
| 5s   | −11.13 | −4.79 | +3.24  | −0.15 | — |
| 10s  | −13.75 | −1.98 | +6.18  | −0.05 | — |
| 30s  | −16.14 | +0.58 | +8.59  | +0.01 | 0.3 |
| 1min | −22.31 | +6.93 | +14.58 | +0.094 | ≈2.9 |
| 2min | −23.24 | +7.90 | +15.72 | +0.084 | ≈2.6 |

($100k near-identical: −4.62 / −1.87 / +0.79 / +7.16 / +8.09.) Decomposition
reconciles exactly (`net_reconstructed == net_realized` to 3 decimals).

### Significance — Newey-West, not just IID

The IID t-stat (`mean / [std/sqrt(n)]`) overstates significance: liquidation
events cluster and 1–2 min holds overlap, inducing positive serial correlation in
the per-trade return series. Newey-West (HAC, automatic bandwidth L=6) corrects
it. Tool: `significance.py`; output `data/results/{reversion,momentum}_significance.csv`.

| horizon | mean bps | t_IID | **t_NW** |
|---|---|---|---|
| 5s   | −4.79 | −4.78 | **−2.78** |
| 10s  | −1.98 | −1.53 | −0.79 |
| 30s  | +0.58 | +0.31 | +0.16 |
| 1min | +6.93 | +2.92 | **+1.43** |
| 2min | +7.90 | +2.60 | **+1.26** |

HAC SEs are ~1.7× (5s) → ~2.1× (2min) the IID SEs — the inflation grows with
horizon, exactly as overlap predicts. **The 1–2 min reversion edge is NOT
significant under Newey-West** (t≈1.3–1.5, p≈0.15): positive point estimate, but
zero is not rejected. L=6 is the standard auto rule and is conservative — a longer
bandwidth only widens the SEs. (Momentum, Finding 1, stays decisively negative
under HAC: t_NW −3.7 to −6.4.)

### Interpretation

- **`gross_mid_to_mid` flips sign almost exactly** vs Finding 1 (−3.23 → +3.24,
  …, −15.72 → +15.72). Clean confirmation that the mid retraces against the
  liquidation flow; reversion captures it. Edge grows monotonically with horizon
  and is consistent across both sizes — not a cherry-picked cell.
- **Fees unchanged at 10 bps round-trip** — still the largest single line and
  still what sinks the short horizons.
- **Net-positive in point estimate only at ≥1 min, and not significant there.**
  The gross edge (~+15 bps at 1–2 min) clears the 10 bps fee, but the resulting
  ~+7–8 bps net edge is within HAC error bars.
- `latency` flips to a tiny cost (+0.24 bps); spread/execution terms are small
  (~1–2 bps) and don't change the conclusion.

### How much to believe it

Weakly positive — do not trade on this as-is:
- **Not significant under Newey-West** (1–2 min t≈1.3–1.5). The earlier IID
  t≈2.6–2.9 was inflated by serial correlation.
- **In-sample** (training window only) — no out-of-sample confirmation yet.
- **Optimistic fills** — sweep assumes the historical tape was available and the
  order adds no impact.
- **Execution/spread terms cross two feeds** (aggTrades vs bookTicker) and are
  timing-sensitive around cascades — lean on `gross_mid_to_mid` (single-feed,
  robust), not the execution bps.

One-liner: the gross reversion edge is real and sizable at 1–2 min (~+15 bps) and
clears the 10 bps fee in expectation, but the net edge is indistinguishable from
zero once serial correlation is accounted for. The fee is the binding constraint,
so the edge can't survive honest error bars until it gets thicker.

### Next steps

1. **Step 2 — maker entry (highest leverage).** The 10 bps taker fee eats ~2/3 of
   the gross edge. Maker is 2 bps vs 5 bps; even maker-in/taker-out saves ~3 bps
   round-trip, which would push 30s clearly positive and 1–2 min toward ~+10 bps.
2. Out-of-sample / robustness: confirm on a held-out period before believing the
   1–2 min edge.
3. Step 3 — event selection (liquidation magnitude / vol regime) to concentrate
   the edge.

---

## Finding 3 — Reversion strengthens with cascade size: the inner region is the wrong place to trade

**Date:** 2026-06-28
**Artifacts:** `backtest_reversion_event_slices.py` (writes
`data/results/reversion_event_slices_summary.csv`). The signal builder
(`LiquidationMomentumStrategy._build_signal_events`) and the Model C runner now
take a percentile **band** (`percentile` lower + `upper_percentile`), selecting an
inner slice of the trailing-7d cascade-size distribution instead of the open-ended
right tail.
**Scope:** training window 2023-06-25 → 2024-06-24, reversion
(`signal_direction_sign=-1`), $50k, fixed-horizon exit across [5s,10s,30s,1min,2min].

### Motivation / hypothesis

The 98th-pct threshold was built for the *momentum* signal — it isolates the
open-ended right tail (largest cascades). For reversion the working heuristic was
that the right tail (≥98th) might be **regime-shifting, non-reverting** moves,
while the far left tail dislocates too little to clear fees — so the tradeable
reversion should live in the **inner region**. Tested three inner bands of the
trailing distribution, walking down from just below the tail toward the median:
[0.50,0.80], [0.80,0.95], [0.95,0.98].

### Results (per trade, same metric throughout; gross = realized fill-to-fill,
net = after 10 bps round-trip taker fee)

Gross reversion edge by band and horizon (bps):

| band | n (2min) | 5s | 10s | 30s | 1min | 2min |
|---|---|---|---|---|---|---|
| [0.50,0.80] | 12,318 | 2.42 | 2.43 | 2.54 | 2.62 | 3.08 |
| [0.80,0.95] | 6,096  | 2.90 | 2.73 | 3.21 | 1.54 | 3.69 |
| [0.95,0.98] | 1,205  | 2.96 | 4.53 | 6.26 | 5.09 | 6.80 |
| **[0.98,→] (tail)** | 969 | **5.21** | **8.02** | **10.58** | **16.93** | **17.90** |

Net (after fee), 2min: [0.50,0.80] **−6.92**, [0.80,0.95] **−6.31**, [0.95,0.98]
**−3.20**, tail **+7.90**. Every inner-region cell is net-negative at every
horizon; with n up to 12k the losses are overwhelmingly significant (t_IID ≈ −15
to −100). The tail row reproduces Finding 2 exactly (net +6.93 @1min, +7.90 @2min).

### Verdict: the heuristic is inverted

**Reversion strength increases monotonically with cascade size**, at every
horizon. At 2 min the gross edge runs +3.08 → +3.69 → +6.80 → **+17.90** bps as
the band climbs from [0.50,0.80] to the ≥98th tail. The largest cascades revert
the *most*, not the least — the opposite of the "big = non-reverting" guess. The
inner region reverts too weakly to clear the 10 bps fee anywhere, so it is
decisively unprofitable, not marginal. The only net-positive region remains the
extreme right tail at 1–2 min — i.e. the original 98th-pct threshold was already
selecting the right events; the problem (Findings 1–2) was never event selection,
it is the fee and the thin, not-yet-significant tail edge.

### Cross-cutting clue: gross grows with horizon in every band

Gross rises with holding period across the board, and steeply in the tail
(5.21 → 17.90 over 5s → 2min). The retracement is still building at 2 min — the
current horizon grid may be truncating it. This motivates testing **longer
horizons** (1m / 5m / 30m / 60m) on the tail, to see whether the gross edge keeps
growing enough to clear the fee with room to spare.

### Next steps

1. **Longer horizons on the tail** — extend the holding grid to [1m,5m,30m,60m]
   (replacing the short grid) and re-measure gross/net on the ≥98th events; the
   monotone-in-horizon gross suggests the edge may not have peaked by 2 min.
2. Maker entry (Finding 2 Step 2) remains the highest-leverage fee attack and is
   orthogonal to this — it lifts every cell by ~3–6 bps round-trip.
3. Drop the inner-region event-selection branch: it is ruled out here.

---

## Finding 4 — Longer horizons on the tail: a significant ~+24 bps reversion edge at 5 min

**Date:** 2026-06-28
**Artifacts:** `backtest_reversion_long_horizons.py` (writes
`data/results/reversion_long_horizons_{summary,trades}.csv`) and
`significance.py --trades …reversion_long_horizons_trades.csv` (Newey-West).
**Scope:** training window 2023-06-25 → 2024-06-24, reversion
(`signal_direction_sign=-1`), ≥98th-pct tail (the 969-event set from Findings 2–3),
$50k, fixed-horizon exit across **[1m, 5m, 30m, 60m]**.

### Motivation

Finding 3 showed the tail gross edge was still climbing at 2 min (the grid's cap),
so the 2-min horizon was likely truncating the retracement. Extend the grid.

### Results (per trade; gross = realized fill-to-fill, net after 10 bps fee)

| horizon | n | gross_bps | net_bps | t_IID | **t_NW** (L=6) | t_NW (L=100) |
|---|---|---|---|---|---|---|
| 1min  | 969 | 16.93 | 6.93  | 2.92 | 1.43 | 1.11 |
| **5min**  | 969 | **33.67** | **23.67** | 8.51 | **4.33** | **3.43** |
| 30min | 969 | 33.40 | 23.40 | 5.69 | 2.66 | 1.92 |
| 60min | 969 | 22.28 | 12.28 | 2.52 | 1.20 | 0.98 |

The gross edge does **not** plateau at 2 min — it nearly doubles by 5 min
(+16.9 → +33.7), holds through 30 min, then fades by 60 min. The shape (peak at
5–30 min, decay by 60 min) is a clean reversion signature: the bounce completes
around 5–30 min, after which the position just holds noise.

### This clears the Newey-West bar that killed the 1–2 min edge

`significance.py` (HAC, auto L=6): **5 min t_NW = 4.33**, 30 min = 2.66 — both
significant, where 1 min (1.43, the Finding 2 result) and 60 min (1.20) are not.
**Robust to bandwidth:** widening L to 100 leaves 5 min at **t_NW = 3.43** (30 min
falls to 1.92, 60 min ~1). The 5-min edge is the first configuration in the study
that survives an honest serial-correlation correction.

### It is not just bull-market beta

The tail is SELL-skewed (718 long vs 251 short trades), so a naive worry is that
long holds in a rising market manufacture the edge. Splitting by direction rejects
that — at 5 min **both sides profit** (long +27.2, short +13.5 bps net); a pure
long-drift artifact would make the shorts *lose*. By 30 min shorts (+25.2) even
beat longs (+22.8). Reversion works in both directions, not just the long book.

| horizon | long net (n=718) | short net (n=251) |
|---|---|---|
| 1min  | +14.93 | −15.94 |
| 5min  | +27.21 | +13.54 |
| 30min | +22.76 | +25.23 |
| 60min | +10.18 | +18.27 |

(The 1-min short book loses; the short-side reversion only turns on by 5 min — on
a small n, so treat the short side cautiously.)

### How much to believe it

This is the strongest result in the study so far — a sizable net edge that, unlike
F2, survives HAC and is not a directional artifact — **but it is not yet
de-risked:**
- **In-sample**, single training window. The bull-market-regime worry is *reduced*
  (both directions work) but only a held-out / bear-regime test settles it.
- **Optimistic fills** — sweep assumes the historical tape was available and the
  order adds no impact.
- **Not market-neutralized.** Both-directions-profit is strong evidence against a
  beta artifact, but subtracting the contemporaneous BTC return over each hold
  would make the alpha airtight.
- Short-side edge rests on n=251 and flips sign between 1 and 5 min.

One-liner: lengthening the hold to **5 min on the ≥98th tail** turns the thin,
not-significant 1–2 min reversion into a **+23.7 bps net edge with t_NW ≈ 4.3
(robust to bandwidth), profitable in both directions** — the 2-min grid was
truncating it. Now de-risk: out-of-sample, market-neutralization, and the maker-fee
attack (which would lift +23.7 toward ~+27 bps).

### Sub-period stability across train months (NOT true OOS)

The 5-min hold was chosen on the full training aggregate, so no train month is a
true out-of-sample test — that is reserved for the untouched test period
(2024-06-25 → 2024-10-14, for the final run). As a sub-period stability check, the
5-min tail trades were sliced by calendar month (the trailing-7d threshold is
causal, so each month's trades equal a standalone-with-warmup run):

| month | n | net_bps | t_NW | long_net | short_net |
|---|---|---|---|---|---|
| 2023-06 | 35  | −17.34 | −0.94 | −16.75 | −19.33 |
| 2023-07 | 45  | +12.37 | +1.48 | +9.12  | +38.30 |
| 2023-08 | 51  | −5.62  | −0.44 | +2.13  | −28.26 |
| 2023-09 | 30  | +15.10 | +3.08 | +13.74 | +18.86 |
| 2023-10 | 69  | +19.39 | +0.90 | +29.43 | +15.57 |
| 2023-11 | 64  | +20.44 | +2.59 | +41.39 | −4.84  |
| 2023-12 | 93  | +39.44 | +1.93 | +52.37 | +2.26  |
| 2024-01 | 89  | +65.67 | +3.65 | +61.10 | +94.97 |
| 2024-02 | 122 | +18.41 | +1.51 | +12.28 | +29.27 |
| 2024-03 | 134 | +46.61 | +3.05 | +50.54 | +31.01 |
| 2024-04 | 91  | +14.64 | +0.71 | +9.20  | +54.20 |
| 2024-05 | 99  | +2.95  | +0.33 | +15.26 | −56.44 |
| 2024-06 | 47  | +11.62 | +1.95 | +12.77 | −5.25  |

- **Broad, not one lucky month:** 11/13 months net-positive; longs positive in all
  but the 5-day partial first month (2023-06). The two negatives are 2023-06 (tiny
  n) and 2023-08 (−5.6).
- **But time-varying and regime-concentrated:** clearly strongest through the
  late-2023 → early-2024 bull run (Dec +39, Jan +66, Mar +47), weak/negative over
  summer 2023. The full-train +23.7 is partly carried by that window.
- **Per-month inference is low-powered** (n 30–134): only a few months are
  individually significant under HAC. The short book is especially noisy
  month-to-month (e.g. 2024-05 short −56).
- **Caveat for the real OOS:** the regime-concentration means the test period
  (Jun–Oct 2024), if a different regime, is a genuine test the in-sample number may
  not survive — exactly why it is being kept untouched.

### Market-neutral (abnormal return) — the edge is alpha, not drift

A single-BTC-asset directional trade can't be hedged against BTC over the same
window (that is identically its own return), so this uses an **event-study
abnormal return**: subtract the trend the price was already on, estimated
*causally* from a pre-cascade window. Per trade (`market_neutralize.py`, all from
L1 mids): `signal = direction·(mid[dec+5m]/mid[dec]−1)`; `normal = direction·`(mid
drift over the clean window [dec−65m, dec−5m] projected onto 5 min); `abnormal =
signal − normal`.

| component | mean bps |
|---|---|
| raw mid signal | +31.96 |
| normal (trend) removed | **−5.40** |
| **abnormal (alpha)** | **+37.36**  (t_IID +13.0, **t_NW +6.59**, auto L=6) |
| abnormal − 10 bps fee | **+27.36** |

The trend component is **negative**: sell cascades are the culmination of a
sell-off, so price drifts *down into* the event and the reversion trade bets
*against* the local trend. Removing that adverse trend makes the alpha **larger**,
not smaller (+31.96 → +37.36) — the opposite of the bull-drift worry. The edge is
fighting drift, not riding it. Both directions revert against trend (long +41.05,
short +26.80 abnormal bps), and **12/13 train months are net-positive on abnormal**
(only the 5-day partial 2023-06 negative). Sanity: raw mid signal +31.96 ≈ the
fill-based gross +33.67 above.

Caveat: the drift is estimated from a single 60-min pre-window ending 5 min before
decision; window-length sensitivity is untested (the adverse-trend sign is
mechanically expected, though, since cascades follow sell-offs).

### P&L decomposition — the edge is the signal, not execution

`pnl_decomposition.py --trades …reversion_long_horizons_trades.csv` splits the net
per trade into additive bps (costs shown positive-and-subtracted; a *negative*
spread term means the fill beat mid):

| horizon | n | gross_mid_to_mid | latency | spread_entry | spread_exit | fees | net |
|---|---|---|---|---|---|---|---|
| 1min  | 969 | 14.58 | 0.24 | −0.83 | −1.77 | 10 | 6.93 |
| **5min**  | 969 | **31.91** | 0.24 | −0.83 | −1.17 | 10 | **23.67** |
| 30min | 969 | 31.25 | 0.24 | −0.83 | −1.56 | 10 | 23.40 |
| 60min | 965 | 20.48 | 0.23 | −0.81 | −1.21 | 10 | 12.27 |

The 5-min net is essentially `gross_mid_to_mid (+31.91) − 10 bps fee`. Execution is
negligible: latency +0.24, and the spread terms are slightly *favorable*
(−2.0 bps combined). Three independent measures of the signal agree: gross-mid
**+31.91** ≈ market-neutral raw signal **+31.96** ≈ fill-based gross **+33.67**. The
mid genuinely retraces ~32 bps over 5 min — not a spread/book-walk artifact.
Reconciliation is exact (net_reconstructed 23.674 ≈ net_realized 23.670).

- **Fees (10 bps) are the dominant friction** — the obvious next lever is maker entry.
- **Caveat:** the favorable spread reflects the optimistic sweep-fill assumption. A
  conservative ~2 bps/side taker cost instead gives ~**+18 bps** net
  (31.9 − 0.24 − 4 − 10) — still strongly positive, but the honest floor.

### Next steps

1. ~~Market-neutralize~~ **done**: abnormal +37.36 bps, t_NW +6.59 — alpha fighting
   an adverse trend, not drift.
2. ~~P&L-decompose~~ **done** (above): the 5-min edge is `gross_mid_to_mid`
   (+31.9 bps), execution negligible. Fees are the binding friction.
3. **Maker entry (Step 2)** — the dominant 10 bps fee is now the clear target;
   maker-in/taker-out saves ~3 bps, maker/maker ~6 bps round-trip.
4. **True out-of-sample** on the held-out test period (2024-06-25 → 2024-10-14) —
   final-run only; keep it untouched until the above are exhausted.

---

## Finding 5 — MAE/MFE path study: early exits (take-profit / stop) HURT; holding is the edge

**Date:** 2026-06-28
**Artifact:** `mae_mfe_analysis.py` (writes `data/results/mae_mfe_8mo_trades.csv`).
**Scope:** **8-month dev window only** (2023-06-25 → 2024-02-24; book reads capped
at the dev boundary so the holdout stays untouched), ≥98th tail reversion, 5-min
trades, **n=541**. Per trade, replay the L1 mid path forward from decision and
measure the running signed return `r(t)=direction·(mid_t/mid_decision−1)` in bps;
mid-to-mid (so the net view subtracts the 10 bps fee).

### Baseline reproduces the edge on the dev window

Hold-to-5min: gross **+31.92** / net **+21.92** bps (long +38.1 / short +18.5) —
essentially identical to the full-train gross +31.91, confirming the 8-month dev
window faithfully carries the edge (it contains the strong Dec23–Feb24 stretch).

### Trades take huge two-sided excursions, then recover

| within 5 min | mean | median |
|---|---|---|
| MFE (best point) | +69.4 | +46.7 |
| MAE (worst point) | −76.6 | −33.9 |
| give-back (MFE − final) | +37.4 | — |
| time-to-MFE-peak | — | 166 s |

90% of trades reach +10 at some point, 69% reach +30, 47% reach +50. The reversion
plays out largely as a **recovery from intra-window drawdown** (mean trade goes
from −77 at its worst to +32 by the 5-min mark). Mean path: +20 @3min, +31.9 @5min,
flat to +33 @30min — **5 min is near-optimal as a fixed cap.**

### Take-profit hurts at every level (right-skewed, persistent winners)

| TP (bps) | %hit | net | vs base +21.92 |
|---|---|---|---|
| +20 | 79% | +1.74 | worse |
| +50 | 47% | +10.01 | worse |
| +100 | 17% | +14.80 | worse |

No take-profit beats hold-to-5min. Trades that reach +100 within 5 min tend to *end
even higher*, so any cap sacrifices the fat right tail that carries the edge.

### Stops are catastrophic

A −10 stop → net −6.64; even a −50 stop → net −5.27 (vs +21.92). Winners take real
heat (winner MAE mean −43 vs loser MAE −154), so any stop tight enough to catch
losers also cuts winners about to revert. Locking in a drawdown kills the recovery
that *is* the edge.

### Verdict

**The take-profit / stop family of early exits does not improve expected return —
it destroys it.** Hold-to-5min is near-optimal. This also explains why the
Improvement-1 dynamic `RetracementExit` underperformed the fixed exit: TP caps the
persistent winners, the stop locks in recoverable drawdowns. The path features that
separate winners from losers (MAE) are only known *after* entry, so they can't trim
either. **The lever with real headroom is entry-time selection (trimming), not
exiting** — pursue book-state / cascade-feature entry filters next. One narrow
untested variant: a *trailing* stop armed only after an extreme MFE (unlikely to
help given the persistent right tail).

---

## Finding 6 — Entry filters: the edge lives in violent cascades; trimming the calm ones ~doubles per-trade return

**Date:** 2026-06-28
**Artifact:** `entry_filter_analysis.py` (writes
`data/results/entry_filter_8mo_features.csv`).
**Scope:** **8-month dev window** (2023-06-25 → 2024-02-24; book reads only look
back from decision, holdout untouched), ≥98th tail reversion, 5-min trades,
**n=541**. For each trade, compute causal decision-time features and test which
predict the realized net bps (after fee), via Spearman rank-correlation + quintile
conditional means.

### Baseline & direction

Mean net **+24.15 bps**. Long **+30.1** (n=371) vs short **+11.1** (n=170) — longs
clearly stronger, shorts positive but weaker (consistent with F4).

### Feature → net return (Spearman r, t)

| feature | r | t | reading |
|---|---|---|---|
| trailing 30-min vol | **+0.38** | +9.5 | strongest — high-vol regime reverts hard |
| cascade displacement | **+0.36** | +9.0 | bigger dislocation → bigger reversion |
| pre-cascade trend (predrift) | −0.22 | −5.2 | steeper *capitulation* → stronger bounce |
| spread | +0.16 | +3.8 | positive but non-monotonic → not usable |
| size_ratio (volume / threshold) | −0.09 | −2.1 | extreme *volume* reverts *less* |
| book imbalance | +0.03 | ns | no signal |
| micro-price lean | 0.00 | ns | no signal |

- **Vol and displacement are the same underlying signal** (Spearman 0.78) — both
  proxy how violent/overshot the cascade was; that is the dominant predictor.
  Quintile highlights: vol Q5 **+113** vs bottom-4 ≈0; displacement Q5 **+96** vs
  bottom-2 ≈0; steepest-capitulation predrift Q1 **+85**.
- **Microstructure-at-decision (imbalance, micro-price) predicts nothing** — the
  *magnitude of dislocation* does, not the instantaneous book lean.
- The original "extreme = non-reverting" heuristic holds **only for volume**
  (size_ratio >2× threshold nets just +3.5), but **price displacement** runs the
  other way — large displacement is good, extreme volume is not.

### Trim simulation

| filter | kept | mean net | % of total P&L kept |
|---|---|---|---|
| baseline | 100% | +24.15 | 100% |
| **displacement ≥ median** | 50% | **+46.6** | **97%** |
| vol ≥ median | 50% | +44.7 | 93% |
| vol ≥ med AND disp ≥ med | 42% | +53.6 | 93% |
| vol top-20% | 20% | **+113** | 94% |

The bottom half of cascades by displacement contributes ~3% of total profit.
Dropping them **doubles** the per-trade net edge (+24 → +47) while keeping ~97% of
the money — same P&L, half the trades, half the capital-at-risk, far better margin
over the 10-bps fee. The top vol quintile alone (20% of trades) captures 94% of all
profit.

### Verdict / recommendation

The reversion edge is concentrated in the **violent, overshot cascades**; the calm,
small-displacement ≥98th events barely revert and only dilute the per-trade edge.
**Recommended filter:** trade ≥98th cascades only when decision-time **displacement
(or trailing 30-min vol) is above the median.** Economically grounded and robust
(median, not a tuned extreme).

**Caveats:** in-sample on dev; vol ≈ displacement (Spearman 0.78) so pick one —
displacement is the more direct "room to revert"; the trimmed book has **higher
per-trade variance** (these are the big-MFE/MAE trades) and leans harder on the
volatile regime (ties to F4's regime-concentration); a *live* filter needs a
trailing/rolling median, not the full-sample one. **Validate the threshold on the
holdout before trusting it.**

### Next steps

1. ~~Bake a causal (trailing-median) displacement filter and re-measure~~ **done →
   Finding 7** (validates, including within-train OOS).
2. Maker entry (Step 2) — orthogonal, on top of the trimmed book.
3. Final test-period OOS — reserved, final run only.

---

## Finding 7 — Causal trim filter holds up out-of-sample (within-train holdout)

**Date:** 2026-06-28
**Artifact:** `apply_trim_filter.py` (writes `data/results/trim_filter_trades.csv`).
**Scope:** full-train 5-min tail trades split into **dev** (2023-06-25 → 2024-02-24)
and the **within-train holdout** (2024-02-25 → 2024-06-24); the test period
(2024-06-25 →) is *not* touched. Trimming only drops events, so masking the existing
trades is exactly equivalent to baking the filter into the signal.

### The filter (causal, fixed from dev)

Keep a ≥98th cascade iff its decision-time **displacement** exceeds a **trailing
30-day median** of past tail-event displacements (≥15 prior events, else keep — a
causal cold-start). Adaptive and look-ahead-free; parameters fixed from F6, no
holdout tuning.

### Results

| cell | n | mean net bps | t_IID | t_NW |
|---|---|---|---|---|
| DEV all | 541 | +24.15 | +6.75 | +3.55 |
| DEV filtered (64% kept) | 347 | **+34.68** | +6.65 | +3.59 |
| HOLDOUT all | 428 | +23.06 | +5.25 | +2.76 |
| **HOLDOUT filtered (36% kept)** | 156 | **+51.01** | +5.54 | **+3.22** |

1. **Base edge holds OOS within train:** unfiltered holdout +23.06 (t_NW +2.76) ≈
   dev +24.15 — the 5-min tail reversion is stable across the train period, not a
   dev artifact.
2. **The filter generalizes — better OOS:** fixed from dev, it lifts the holdout
   mean +23 → **+51** and *raises* significance (t_NW 2.76 → 3.22) on 36% of trades;
   on dev it lifts +24 → +35 with t_NW essentially unchanged. Trimming preserves
   significance despite fewer, higher-variance trades.

The adaptive keep-rate (64% dev vs 36% holdout) is the trailing median tightening in
calmer stretches so only genuinely violent cascades trade — exactly the intended
behavior, and those revert hardest.

**In dollars** (net bps × $1 per $10k notional per round-trip): DEV all $24.15 →
filtered $34.68; HOLDOUT all $23.06 → **filtered $51.01 net per $10k per trade**.
(Per-trade on notional — not a return on fixed capital, since 5-min holds overlap.)

### Holdout dollar P&L and capital (at the backtest's $50k/trade)

Over the ~4-month holdout (2024-02-26 → 2024-06-24), accounting for overlap (peak
simultaneous positions sets the capital base):

| | trades | total net P&L | per-trade notional | peak concurrent | peak capital | return on peak cap |
|---|---|---|---|---|---|---|
| HOLDOUT all | 428 | $49,350 | $50k | 23 | $1,150,000 | +4.3% |
| **HOLDOUT filtered** | 156 | **$39,789** | $50k | 12 | **$600,000** | **+6.6%** |

The trim gives up ~20% of the gross dollars but on **half the capital**, so
return-on-capital rises +4.3% → **+6.6%** over ~4 months (~20% annualized, caveated).
"Initial notional" has two senses: per-position **$50k**, or the capital base to fund
the strategy ≈ **$600k** (peak-12 concurrent; average utilization is far lower, so
position limits could run it on less). $50k/trade is arbitrary — the scale-free
quantity is the per-trade edge (+51 bps); total $ scales ~linearly with size until
impact bites (negligible here, F1). Indicative, not hardened: within-train OOS,
optimistic fills, taker fees, regime-concentrated.

### How much to believe it

This is the project's first OOS validation and it passes — **but it is within-train
OOS, not the reserved test period.** The filtered holdout n=156 is getting small and
the trimmed book carries higher per-trade variance. The multiple-testing worry from
F6 is now substantially reduced: displacement was an a-priori-sensible feature *and*
it validates OOS without re-tuning.

### Next steps

1. Maker entry (Step 2) — kill the ~10 bps fee on top of the trimmed, OOS-validated
   book; would lift the filtered ~+35–51 toward ~+40–57. **→ Finding 8: tested and
   rejected — passive entry is a net loser here.**
2. Final **test-period OOS** (2024-06-25 → 2024-10-14) — single final run only, kept
   untouched until the strategy is frozen.

---

## Finding 8 — Maker entry doesn't help: the fee saving is outweighed by adverse selection (the long-standing "Step 2" is rejected)

**Date:** 2026-06-28
**Artifacts:** `backtester._maker_fill_from_agg_trades` (new passive-fill primitive)
and `backtest_reversion_maker.py` (writes `data/results/reversion_maker_trades.csv`).
**Scope:** the Finding-7 setup — full-train ≥98th tail, 5-min reversion, $50k,
causal displacement trim filter (fixed from dev) — split DEV (2023-06-25 →
2024-02-24) vs within-train HOLDOUT (2024-02-25 → 2024-06-24). Test period
untouched.

### Motivation

Every finding since F1 named the **10 bps round-trip taker fee** as the binding
friction, and F5 showed we *want* to hold (no hurry to fill) — the textbook setup
for passive entry. The standing "Step 2" (since F2's next-steps) was that maker
entry would lift the filtered net from ~+35–51 toward ~+40–57. This tests it.

### Method

`_maker_fill_from_agg_trades` is the mirror of the taker sweep: a resting limit at
the decision-time **touch** (bid for a long reversion, ask for a short) fills only
when an aggressor on the *opposite* side trades into it (`want_buyer_maker` flips to
`signed > 0`), gated by a price condition. **strict_cross** (used here) requires the
tape to trade *strictly through* the limit — a conservative queue assumption (the
level cleared, so a resting order at it filled). A passive order fills at its own
quoted price, so the fill VWAP is the limit, not the print. **No-fills are dropped**
(you posted, never filled → no trade). The driver simulates a taker entry and a
maker entry per event, exits both with a taker sweep at decision + 5min + latency,
and applies the F7 trim mask. Fills are fee-independent, so net is computed under
three fee modes off one sim. **The honest headline is `net_bps(all)`: mean over
*every attempted event*, crediting the non-filled maker events at 0 P&L** — so the
entry-fee saving and the forfeited winners sit on one ledger.

### Results (net bps; baseline reproduces F7 exactly)

| cell | taker/taker (baseline) | maker-in/taker-out (realistic) | maker/maker (optimistic bound) |
|---|---|---|---|
| DEV all (n=541) | **24.15** (t_NW 3.55) | 22.67 (3.43) | 25.44 (3.84) |
| DEV filtered (n=347) | **34.68** (3.59) | 33.40 (3.47) | 36.22 (3.76) |
| **HOLDOUT all** (n=428) | **23.06** (2.76) | 20.11 (2.80) | 22.92 (3.19) |
| **HOLDOUT filtered** (n=156) | **51.01** (3.22) | **45.58** (3.29) | 48.41 (3.50) |

(`maker/maker` is a fee-only bound: it credits a 2 bps maker *exit* fee without
modelling passive-exit fill risk — same taker-exit fills, 2 bps charged. Maker fill
rate 94–96% throughout.) Baseline taker/taker matches F7 to the decimal (DEV all
24.15 / filtered 34.68, HOLDOUT all 23.06 / filtered 51.01), confirming the event
set, displacement, trim mask, and P&L are apples-to-apples.

### Verdict: passive entry is a net loser here

**The realistic maker-in/taker-out underperforms the taker baseline in every cell** —
by 1.3 bps on dev up to **−5.4 bps on the OOS filtered holdout** (51.01 → 45.58).
Even the *optimistic* maker/maker fee-only bound only ties/slightly-beats taker on
dev and **still loses on the holdout** (51.01 → 48.41). Significance is preserved
(t_NW ≈ 3.2–3.5) — the point estimate just doesn't improve. **The 3 bps entry-fee
saving does not survive contact with the fill.**

### Mechanism — two costs, both intrinsic to a reversion signal

1. **Adverse selection on the fills.** Compare `net_bps(filled)`: HOLDOUT-all
   maker-in/taker-out fills net **21.36** vs taker **23.06** — *worse* despite a
   3 bps fee discount, i.e. the gross fill price is ~4–5 bps worse. A resting bid in
   a falling cascade is only hit when a taker sells *through* it; you fill precisely
   on the events that kept dropping, then recover from a deeper hole.
2. **Missed winners.** Fill rate is high (94–96%), but the 4–6% no-fills aren't
   random — they're the events where the bounce started immediately and price never
   traded back down to your bid (your *best* trades). On the same-universe `all`
   basis those misses count as zero, and forfeiting them costs more than the fee
   saving gains.

Both are baked into the signal: you are betting "price is about to revert," so a
passive order on the side you want is **by construction adversely selected against**.
Posting more aggressively (inside the touch) fills more but at worse prices; less
aggressively misses more winners — no placement escapes the trade-off.

### Implication

This **overturns the standing assumption** (carried since F1) that maker entry was
the highest-leverage next step. The fee is the biggest cost *line* but is not
capturable passively here — the cure costs more than the disease. The **frozen taker
candidate stands as the best configuration: trimmed 5-min tail reversion, +51 bps
net on the within-train holdout (t_NW 3.22), taker/taker fills.** There is no maker
improvement to stack before the final test.

**Caveats:** one queue model (strict-cross) and one placement (touch); inner/outer
limit prices and the non-strict (front-of-queue) assumption were not swept — but the
*optimistic* maker/maker bound already failing OOS means they would not rescue it.
Within-train holdout, optimistic taker exit sweep, regime-concentrated — unchanged
from F7.

### Next steps

1. ~~Maker entry~~ **done — rejected.** No fee-side improvement to pursue.
2. The remaining real lever is the **one-shot TRUE OOS on the test period**
   (2024-06-25 → 2024-10-14), to be run once the strategy is declared frozen — which,
   with maker ruled out, it effectively is (trimmed 5-min taker tail reversion).
3. Optional before freezing: parameter-robustness of the trim lookback (30d) and the
   market-neutral pre-window (60min) on dev only (F4/F7 flagged both as untested),
   to de-risk the single OOS shot. **→ Finding 9: done — all checks pass.**

---

## Finding 9 — Pre-OOS de-risking: the frozen candidate is lookback-robust, survives conservative fills, and is not impact-distorted at backtest sizes

**Date:** 2026-06-29
**Artifact:** `robustness_checks.py` (reuses `data/results/trim_filter_trades.csv` for
the precomputed causal displacement, `..._long_horizons_decomp_trades.csv` for the
per-trade bps split, and `reversion_decomposition_trades.csv` for the $50k-vs-$100k
fills). No new simulation; no book reads.
**Scope:** the F7 frozen candidate — trimmed 5-min ≥98th tail reversion — on
DEV (2023-06-25 → 2024-02-24) and the within-train HOLDOUT (2024-02-25 → 2024-06-24).
**Test period untouched.** Three checks flagged as the pre-OOS gaps (F4/F7).

### Check 1 — trim-filter robustness: the +51 holdout edge does not depend on the lookback

The causal displacement filter's trailing-median lookback `W` (F7 fixed at 30d) and
`min_history` (15) were never tested. Sweeping **W ∈ {5,10,15,30,45,60,90,120} days ×
min_history ∈ {10,15,25}**, the **holdout-filtered result is invariant**: +51.01 bps,
t_NW 3.22, n=156 (36% kept) in *every* setting — the holdout keep-set is element-wise
identical for all W ≥ 30d and gives the same mean/n below that. `min_history` only
nudges the dev cold-start (501 vs 504 of 969 total kept), never the holdout. The
displacement distribution of tail events is stationary enough that a 5-day and a
120-day trailing median rank the same events across the keep threshold. **The W=30d /
min_hist=15 choice does not drive the number — it is not a tuned setting.**

### Check 2 — pessimistic fill floor: survives a conservative taker spread

The headline net rests on the optimistic aggTrades sweep, whose spread terms came out
slightly *favorable* (F4: −2.0 bps combined). This check is an **arithmetic substitution
on the decomposition, not a pessimistic re-simulation**: take
`net_floor = gross_mid_to_mid − latency − 2·s − 10` and replace the favorable fills with
a flat taker spread `s` per side. It does **not** re-run fills crossing the full spread or
model partial/no-fills. The 2 bps/side base is the F4 assumption — *not* a measured
cascade-time half-spread — so it is also swept to 3 and 4 bps.

At s = 2 bps/side:

| cell | optimistic (realized) | floor (2 bps/side) |
|---|---|---|
| DEV all (n=541) | +24.15 (t_NW 3.55) | +17.88 (2.62) |
| DEV filtered (n=347) | +34.68 (3.59) | +28.43 (2.93) |
| HOLDOUT all (n=428) | +23.06 (2.76) | +17.42 (2.08) |
| **HOLDOUT filtered (n=156)** | **+51.01 (3.22)** | **+45.21 (2.86)** |

Sensitivity to the spread magnitude `s` (the assumed, untested parameter):

| s (bps/side) | HOLDOUT filtered | HOLDOUT all |
|---|---|---|
| 2 | +45.21 (t_NW 2.86) | +17.42 (2.08) |
| 3 | +43.21 (2.73) | +15.42 (1.84) |
| 4 | +41.21 (2.60) | +13.42 (1.60) |

**The strategy — the *filtered* book — stays strongly positive and significant through
s = 4 (+41 bps, t_NW 2.60).** The honest floor to carry into the OOS is **~+41 to +45 bps**
depending on the spread you believe. Note the **unfiltered** holdout is more fragile: it
loses significance by s = 3 (t_NW 1.84). That is fine — the strategy *is* the filtered
version — but it shows the trim is doing real work under conservative fills, not just
lifting the point estimate.

### Check 3 — capacity on violent events: doubling size barely moves the fill

F1 found book-walk negligible ($50k ≈ $100k) on the *full* population, but the trimmed
book trades the most violent cascades — when the L1 book is thinnest. Comparing the
entry+exit book-walk (`spread_entry + spread_exit`, bps; negative = filled inside mid)
at $50k vs $100k on the kept (violent) subset (2-min decomposition; entry is at
decision+latency, identical across horizons, so it carries to the 5-min tail):

| subset | $50k total | $100k total |
|---|---|---|
| all events | −2.42 | −2.61 |
| kept (violent) | −2.61 | −2.78 |

Doubling size on the violent subset moves total book-walk by **−0.17 bps** (slightly
*more* favorable — measurement noise either way). Negligible against the +41–51 bps edge.
But this is a **2× probe, not a capacity ceiling**: it shows $50k → $100k is free on
violent events, and says *nothing* about where impact turns material ($250k? $1M?). The
real capacity question — how large the strategy can scale — is untested; all that is
established is that the backtest's $50k–$100k sizing is not impact-distorted.

### What this does and doesn't establish (and a tension with F7)

The Check-1 invariance is genuinely surprising and deserves more than the "stationary
enough" hand-wave: a 5-day-reactive and a 120-day-smooth trailing median should *not*,
a priori, make identical keep decisions. The mechanism is that the trailing median
adapts to the **level** of recent displacements (so the keep-*rate* moves with regime:
64% on dev vs 36% on holdout, F7) but is insensitive to the **window length** (the
displacement level is similar whether averaged over 15 or 120 days, so the bar lands in
the same place). These are different axes, and the log should not conflate them.

This does, however, **temper F7's "adaptive tightening" story.** F7 framed the median as
actively tightening in calm stretches; the W-invariance shows that mechanism is weak —
if it tightened sharply, short windows would react faster than long ones and split the
events differently. They don't. The honest reading: the keep decision is dominated by
whether an event's displacement is large *relative to a fairly stable bar*, and the
bar's window length doesn't matter over 5–120 days. That is robustness, but it also
means the filter is closer to "trade the absolutely-violent cascades" than to a finely
regime-adaptive rule.

**Two parameters were NOT swept** (so "parameter-robust" is scoped to W and min_history,
not the whole filter): (i) the **threshold quantile** — the filter uses the *median* of
past displacements; 40th/60th-percentile variants are untested (F6 only looked at the
displacement quintiles of the realized edge, not at moving the keep bar); (ii) the
**fill-spread magnitude** `s` in Check 2 — addressed there by sweeping s ∈ {2,3,4}, but
the value is assumed, not measured from the cascade-time book.

### Verdict

On the axes tested, the frozen candidate is de-risked: the filtered holdout edge is
invariant to the trim lookback (+51 bps across W ∈ [5,120]d), survives a conservative
fill substitution (**floor ~+41 to +45 bps, t_NW 2.6–2.9** at s = 4 to 2 bps/side), and
is not impact-distorted at the backtest's $50k–$100k sizing. **Nothing here argues
against running the single test-period OOS.** The two caveats that matter are scope, not
red flags: the untested filter parameters above, and — the big one — **regime
concentration (F4), which none of these within-train checks can address. That is exactly
what the OOS tests**, and why the +41–45 floor (not +51) is the number to carry in.

**Caveats:** all within-train. The pessimistic floor is an arithmetic substitution that
still assumes the historical tape liquidity was available (no queue/impact model beyond
the −0.17 bps 2× probe); capacity-to-scale is untested. The threshold-quantile and
spread-magnitude parameters are not (or only partly) swept. Regime-concentration is
unaddressed by design.

### Next steps

1. **One-shot TRUE OOS on the test period** (2024-06-25 → 2024-10-14), strategy frozen:
   trimmed 5-min ≥98th taker tail reversion, displacement > causal trailing-median.
   Carry the floor (**+41 to +45 bps**, spread-dependent) as the conservative
   expectation, not +51; state the regime-dependence prior (F4) before running.
2. Optional, cheap: sweep the trim **threshold quantile** (40th/60th pct vs median) on
   dev to close the last untested filter parameter before freezing.
3. Optional, slow (book reads): the neutralization-window sweep (F4 60-min pre-window
   at 30/90/120) — least critical, re-confirms alpha-vs-drift only.

---

## Finding 10 — TRUE out-of-sample: the edge holds on the reserved test period (+48.85 bps net, t_NW 2.62)

**Date:** 2026-06-29
**Artifact:** `strategies/backtest_oos_test.py` (writes `data/results/oos_test_trades.csv`).
**Scope:** the reserved **test period 2024-06-25 → 2024-10-14**, untouched in F1–F9.
**The frozen strategy was run exactly once; nothing was re-tuned.** Data is loaded from
2024-04-15 so the trailing-7d event threshold and the trailing-30d trim median are fully
warmed for every test event (an early-test event's trim history legitimately includes
late-train tail events — the live/causal behaviour); only events with `decision_time` in
the test period are reported. One causal book pass per trade yields both the displacement
(for the trim) and the mid path (for the conservative floor).

### The strategy (frozen)

≥98th-pct cascade tail (trailing-7d threshold) → reversion, 5-min fixed hold, taker
fills → causal trim: keep iff decision-time displacement > trailing-30d median of past
tail-event displacements (min_history 15). Identical to F7/F9; zero new parameters.

### Result — it generalizes

**245** ≥98th tail events fired in the test period; the trim kept **105 (43%)**.

| cell | n | mean net bps | SE_NW | **t_NW** | t_IID |
|---|---|---|---|---|---|
| test all (unfiltered) | 245 | +15.12 | 10.23 | **1.48** | 2.75 |
| **test FILTERED (the strategy)** | 105 | **+48.85** | 18.67 | **+2.62** | 4.78 |
| gross filtered (before fees) | 105 | +58.85 | 18.67 | +3.15 | 5.76 |

**The filtered OOS net is +48.85 bps (t_NW +2.62), within noise of the within-train
holdout (+51.01 optimistic / +41–45 floor).** The edge did not decay out-of-sample. The
**unfiltered** book is +15.12 and **not** significant (t_NW 1.48) — exactly as F9
predicted, so the **trim is doing essential work**, not cosmetic lifting. NW SEs are
~1.8× the IID SEs (event clustering; L=4), and the result clears the HAC bar anyway.

### Both directions profit — not a regime artifact

| direction | n | net bps | SE_NW | t_NW |
|---|---|---|---|---|
| long | 64 | +58.12 | 27.71 | +2.10 |
| short | 41 | +34.38 | 8.50 | **+4.04** |

The short side is **clean and tightly significant this time** (t_NW 4.04) — strong
evidence against a long-drift/beta explanation, and a genuine test win given the test
window is a different (post-bull, choppier) regime than the Dec23–Mar24 stretch that
carried much of the in-sample edge (F4 prior). Longs are noisier (wide SE, big winners).

### Conservative fill floor (filtered; exact mids, F9 method)

| s (bps/side) | mean net bps | SE_NW | t_NW |
|---|---|---|---|
| 2 | +43.13 | 18.89 | +2.28 |
| 3 | +41.13 | 18.89 | +2.18 |
| 4 | +39.13 | 18.89 | +2.07 |

The edge **survives a conservative taker spread through s = 4 (+39 bps, t_NW 2.07)**.
Sanity: realized-fill gross +58.85 vs mid-path gross +56.10 (the ~2.75 bps gap is the
optimistic sweep's favorable fill, consistent with F4) — the fills are not wildly
optimistic.

### Dollar P&L, capital, ROI (filtered strategy, 112-day test span)

| | trades | total net | mean/trade | peak concurrent | peak capital | ROI | ~annualized |
|---|---|---|---|---|---|---|---|
| $50k/trade | 105 | **$25,647** | $244 | 18 | $900,000 | **+2.85%** | ~+9.3%/yr |
| $100k/trade | 105 | **$50,725** | $483 | 18 | $1,800,000 | +2.82% | ~+9.2%/yr |

$100k nets ~2× $50k at the same ROI — **capacity confirmed live** (impact negligible,
as F1/F9 predicted). Gross $30,897 at $50k, of which $5,250 is fees. The ROI is on the
*peak* capital base (18 simultaneous positions); average utilization is far lower (only
22 of 112 days are active), so the strategy is **capital-inefficient** — the scale-free
quantity is the per-trade edge (+49 bps), and ROI depends heavily on the capital
convention and position-limit policy.

### Risk / Sharpe (filtered, $50k)

- **Per-trade Sharpe +0.467** (mean/std of per-trade net bps) — the robust primitive.
- **Annualized Sharpe ≈ +4.46** (daily P&L on peak capital over all 112 calendar days,
  22 active, ×√365) — *flattered by the short, sparse sample*; treat as indicative.
- **Max drawdown −$6,571** on the equity curve (vs +$25,647 total) — shallow.
- **Hit rate 75%** (79/105 net-positive); best/worst trade +282 / −324 bps; median +42,
  mean +49.

### Verdict

**The liquidation-cascade reversion edge is real and out-of-sample.** Discovered as an
*inversion* of the original momentum hypothesis (F1), localized to the violent ≥98th
tail at a 5-min hold (F3–F4), shown to be alpha fighting an adverse trend (F4),
concentrated by a causal displacement trim (F6–F7), de-risked on parameters/fills/capacity
(F9), and with the fee-side maker improvement ruled out (F8) — it now **clears a single,
pre-registered, one-shot OOS at +48.85 bps net (t_NW 2.62), ~+39–43 under conservative
fills, profitable in both directions, in a different regime.**

**Caveats (the honest ceiling on the claim):**
- **One test period, one asset, n=105 filtered.** The NW SE is wide (18.7 bps); t_NW 2.62
  is ~1% significance but not overwhelming. This is *a* clean OOS, not a Sharpe-rich
  industrial strategy.
- The whole result rests on the trim (unfiltered is insignificant), so the effective
  sample is 105 violent cascades.
- **Optimistic exit fills**; the +39–43 conservative floor is the number to believe.
  Maker entry can't reclaim the fee (F8).
- **Capital-inefficient / bursty** (22 active days); the headline is the per-trade edge,
  not the ~9%/yr ROI, which is convention-dependent.
- **The test set is now spent.** Any further tuning would be in-sample on it; new
  validation requires new data (other COIN-M perps, or a later period).

### Next steps (post-validation)

1. The BTC edge is confirmed — the prerequisite for **cross-sectional generalization**:
   does the same ≥98th tail reversion exist on other COIN-M perps (ETH, …)? That is the
   next real out-of-sample, now worth opening (Track C).
2. Feature work (scale-normalized shock, cascade dynamics) is now unblocked, but must be
   validated on *new* data, never the spent test set.

---

## Finding 11 — Cross-asset OOS: the edge does NOT transfer to ETHUSD_PERP (filtered net −157.52 bps)

**Date:** 2026-07-01
**Artifact:** `strategies/backtest_oos_test_eth.py` (writes `data/results/oos_test_eth_trades.csv`).
Engine change enabling it: `_book_ticker_day_path` / `BookProvider` / the Model-C runner are
now symbol-aware (default `BTCUSD_PERP`, so every BTC run is byte-identical — verified: the
BTC OOS driver reproduces F10 exactly, 245 events → keep 105 → +48.85 / t_NW 2.62).
**Scope:** ETHUSD_PERP, the **same OOS window 2024-06-25 → 2024-10-14** as the BTC one-shot
(F10). The BTC strategy was applied **frozen — nothing re-tuned, no ETH-specific fitting**.
Only the asset bindings differ: ETH data paths, `$10`/contract (ETHUSD_PERP COIN-M face
value vs BTC `$100`; confirmed against Binance's COIN-M specs), and the bookTicker filename
symbol. Warmup load from 2024-04-15 (causal threshold + trim), same as BTC.

### The strategy (frozen, identical to F10)

≥98th-pct cascade tail (trailing-7d threshold) → reversion, 5-min fixed hold, taker fills →
causal trim: keep iff decision-time displacement > trailing-30d median of past tail-event
displacements (min_history 15). Zero new parameters.

### Result — it does not generalize

| set | n | gross bps | net bps | t_NW |
|---|---|---|---|---|
| ETH all | 201 | −56.84 | **−66.84** | −1.59 |
| ETH FILTERED (strat) | 75 | −147.52 | **−157.52** | −1.62 |
| *(BTC F10 filtered, same window)* | *105* | *+58.85* | *+48.85* | *+2.62* |

The point estimate is **strongly negative and flips the sign of the BTC result**. It is not
statistically significant as a *loser* (t_NW −1.62; the NW SE is enormous, 96.9 bps, driven
by a −1413 bps worst trade), but it **decisively fails to replicate** the BTC positive: none
of the BTC structure survives. Mid-path gross (−154.23) ≈ realized-fill gross (−147.52), so
the loss is the **signal**, not an execution/fill artifact.

### Why — a directional sell-off destroyed a long-heavy reversion book

- **The trim, which was essential and helpful on BTC, is actively HARMFUL on ETH**: all →
  filtered moves −66.84 → −157.52 (on BTC it *raised* the mean). The displacement filter is
  not a robust cross-asset selector.
- **The kept set collapses to all-long (75 long / 0 short).** Of 201 tail events, 183 are
  long-reversions (reversions of SELL-liquidation cascades = forced selling) and only 18 are
  shorts; longs carry far higher displacement (mean 87.8 vs 24.9 bps), so the "keep the most
  displaced" trim selects *only* longs.
- Jun–Oct 2024 was an **ETH down-trend**. The strategy loaded up on long "bounce" bets into
  falling ETH and got run over — both raw directions lose (long −71.9, short −15.0 net), longs
  far worse and dominant. $50k sizing: total net −$59k, ROI −5.63% / 112d, max DD −$79k,
  per-trade Sharpe −0.35, hit rate 52%, worst trade −1413 bps.

This is exactly the **regime-concentration risk** flagged in F4/F10: the BTC edge was
concentrated in a bull run; on ETH, the *same calendar window* was a sell-off, and a
reversion strategy that ends up structurally long has no protection against it.

### Verdict

**The liquidation-cascade reversion edge is BTC- and/or regime-specific — it does not survive
cross-asset transfer to ETH on this window.** The clean BTC OOS (F10) should now be read with
this ceiling: one asset, one favourable regime. The signal is *not* a general property of
COIN-M liquidation cascades as currently defined.

**Caveats / what this is and isn't:**
- One asset added, one window, n=75 filtered, t_NW not significant — this is a *failure to
  confirm*, not a proven negative edge. But the burden was on ETH to replicate, and it didn't.
- Frozen transfer is the strict, correct test; **do not re-tune on ETH to rescue it** — that
  would spend ETH the way the BTC test set is spent (F10).
- The all-long collapse suggests the strategy needs a **direction/regime guard** (or a
  trend-neutralized displacement trim) before any cross-asset claim — but that is *new
  research to be validated on new data*, not a fix to bolt on and re-score here.

### Next steps

1. The cross-asset generalization claim is **rejected as-is**. Any revival needs a
   regime/direction-aware reformulation (e.g. trend-conditioned trim, or forbidding a
   structurally net-directional book) — designed and validated on *fresh* data, never this
   ETH window (now informative) or the BTC test set (spent).
2. Optional context (cheap, nothing spent on ETH): run the ETH **train** period to see
   whether ETH ever showed the edge, or whether it's absent on ETH in all regimes.

## Finding 12 — Deployable concurrency caps (train only): the greedy first-come cap is the wrong tool

**Date:** 2026-07-03
**Question (from CONCURRENCY_ANALYSIS.md):** the frozen reversion strategy pyramids up to 18
one-directional positions per cascade; capping concurrency was recommended. Does a **causal,
deployable** concurrency cap improve the strategy on the training data — and which limit
(no-stack N=1, N≤3, N≤5, or uncapped) does best?

**Scope & discipline:** TRAIN ONLY; the reserved OOS test set is untouched. **Select on the dev
window** (2023-06-25 → 2024-02-24, 347 kept trades), **confirm on the within-train holdout**
(2024-02-25 → 2024-06-24, 156 kept — matches F7's holdout n); the holdout never informs the
choice. **No look-ahead:** the cap is the greedy **first-come** rule (open only if < N are
already open, else SKIP the arriving bet) — a pure causal filter on the already-realized frozen
$50k trades. The most-violent-per-cascade oracle (`dedupe_per_episode`) is deliberately **not**
used (it peeks at the whole cascade). Pre-registered selection metric: highest √365 Sharpe among
policies with t_NW ≥ 2, ties → smaller MtM drawdown.
**Artifacts:** `strategies/concurrency_train_caps.py`, `tests/test_concurrency_train_caps.py`
(17 tests), summary `data/results/concurrency_train_caps_summary.csv`. Reuses `cap_at_n` /
`subset_metrics` (`concurrency_analysis.py`), `mtm_max_drawdown` / `realized_close_dd`
(`backtest_oos_risk.py`), and `pnl_decomposition.add_decomposition`. Reviewed adversarially;
findings folded in.

### Results (net bps of $50k; DD in $; ret/DD = window net $ ÷ |MtM DD|)

**DEV (347 kept):**

| Policy | trades | net bps | t_NW | t_clust | Sharpe√365 | realized DD | **MtM DD** | peak cap | ret/\|MtM DD\| |
|---|---|---|---|---|---|---|---|---|---|
| uncapped (N=∞) | 347 | **34.68** | **3.59** | 3.02 | **3.26** | −4,461 | **−47,860** | $900k | **1.26** |
| cap N=5 | 268 | 17.27 | 2.05 | 1.83 | 1.71 | −5,167 | −23,325 | $250k | 0.99 |
| cap N=3 | 201 | 12.71 | 1.56 | 1.49 | 1.45 | −3,746 | −14,225 | $150k | 0.90 |
| cap N=1 (no concurrency) | 81 | 12.25 | 1.45 | 1.66 | 1.70 | −1,524 | **−4,823** | $50k | 1.03 |

**HOLDOUT (156 kept):**

| Policy | trades | net bps | t_NW | t_clust | Sharpe√365 | realized DD | **MtM DD** | peak cap | ret/\|MtM DD\| |
|---|---|---|---|---|---|---|---|---|---|
| uncapped (N=∞) | 156 | **51.01** | **3.22** | 2.77 | **4.21** | −3,805 | **−18,629** | $600k | **2.14** |
| cap N=5 | 119 | 31.44 | 2.05 | 1.71 | 2.63 | −2,754 | −12,994 | $250k | 1.44 |
| cap N=3 | 88 | 32.23 | 2.10 | 1.74 | 2.70 | −1,706 | −7,808 | $150k | 1.82 |
| cap N=1 (no concurrency) | 38 | 25.57 | 1.73 | 1.48 | 2.38 | −1,286 | −3,472 | $50k | 1.40 |

**Dev-selected policy: UNCAPPED** (highest √365 Sharpe among the two policies clearing t_NW ≥ 2:
uncapped 3.26 vs N=5 1.71). It **confirms on the holdout**: +51.01 bps, t_NW 3.22, t_clust 2.77,
Sharpe 4.21.

### What the cap actually does

1. **It cuts risk in ABSOLUTE terms — a lot.** MtM drawdown falls ~10× (dev −$47.9k → −$4.8k at
   N=1; holdout −$18.6k → −$3.5k) and peak capital 18× ($900k → $50k). A book with a fixed
   *dollar* drawdown budget strictly prefers the cap.
2. **But RETURN falls faster, so every risk-adjusted ratio favours uncapped** — on Sharpe (3.26
   vs ≤1.71) **and** on the MtM-aware ret/|MtM DD| Calmar (dev 1.26 vs ≤1.03; holdout 2.14 vs
   ≤1.82), on **both** windows. The Calmar cross-check matters because the √365 Sharpe books only
   realized closes and is blind to the very MtM pain the cap targets; here both metrics agree.
3. **Why the edge collapses: greedy first-come ANTI-SELECTS on edge.** The reversion edge builds
   as a cascade intensifies (F3/F6), so keeping the *earliest* event of each cascade keeps the
   low-edge one. Cost decomposition confirms it: mean gross-mid signal drops (dev 42.6 → 21.5
   bps) while fees (10 bps round-trip) and spread (~2 bps) are fixed, so caps shed *signal*, not
   cost.
4. **Drawdown is NOT driven by the top-P&L day on train** (mtm_dd_topday_share ≈ 0%): the P&L
   peaks on 2024-01-09 (dev) / 2024-03-05 (holdout) but the equity troughs elsewhere
   (2023-06-30 dev / 2024-02-28 holdout). Contrast OOS F10, where one crash day drove 86% of DD.
   Concentration shares are also **non-monotone** in N (N=1 sits below N=3), so "caps concentrate
   P&L" is not a clean effect.

### Verdict

**The greedy first-come concurrency cap is the wrong tool.** On train it trades away more edge
than drawdown and loses significance (N=3/N=1 fall below t_NW 2), so uncapped dominates it on
every risk-adjusted measure across dev and holdout. **This is not "no causal cap can win"** —
only that *this* causal family does, by construction, throw away the high-edge late-cascade
trades.

**Essential caveat that bounds the whole result:** a single train window **cannot realise the
rare one-directional cascade tail the cap exists to defend against** (the OOS 2024-08-05 pyramid
that was 86% of DD; the ETH F11 all-long run-over). "Uncapped wins on train" is the
*pick-up-pennies* reading — good realized Calmar in windows that did not contain the tail. So the
conclusion is scoped: **greedy first-come is rejected; concurrency risk is NOT dismissed.**

### Next steps

1. The deployable candidate remains a **causal displacement-gated entry** (enter only on the
   first event clearing a train-estimated displacement bar, then lock out for the hold) — it
   keeps the *violent* event without look-ahead, unlike first-come. Must be designed and
   validated on **fresh** data (the OOS test set is spent, F10).
2. Or a **direction/regime guard** so the book cannot become structurally one-way (the F11 ETH
   failure mode) — again, fresh data only.

## Finding 13 — Clock-reset bounded cap: the "extend the hold" idea barely fires (cascades are second-scale bursts) and does not beat the greedy cap

**Date:** 2026-07-04
**Question:** F12 compared only *pile on* (uncapped) vs *ignore the new signal* (greedy skip),
and greedy lost by anti-selecting on edge. Proposed third rule: hold a bounded book of at most N
**same-direction** units, but when a new qualifying tail signal arrives while open, **reset the
shared 5-min exit clock** (extend the hold) instead of piling on beyond N or ignoring it — build
up to N units, then extend-only; all units flatten together at (last signal + 5min + latency).
The hope: keep leverage bounded (the cap's risk win) but *use* the later cascade events to hold
through more of the reversion (F5: holding is the edge), recovering the edge greedy discarded.

**Scope & discipline:** TRAIN ONLY (OOS untouched); dev-select (2023-06-25→2024-02-24) +
holdout-confirm (2024-02-25→2024-06-24); runs built **within each window** so a holdout signal
can never extend a dev run. Fully **causal/deployable** — the exit fires 5min after the last
*observed* signal (defer the resting exit on each new arrival), no look-ahead. N=5 primary (F12's
significance-surviving cap), plus N=3 and single-unit extend-only N=1.
**Method:** clock-reset changes *exit times*, so exits are **re-simulated** on the aggTrades tape
with the same taker sweep (`_sweep_fill_from_agg_trades`) as F10/F12; the frozen `entry_*` fills
are reused verbatim (they don't depend on the exit clock). Each bounded book flattens in one
aggregated exit sweep.
**Artifacts:** `strategies/concurrency_clock_reset.py`, `tests/test_concurrency_clock_reset.py`
(12 tests). Reuses the F12 battery (`concurrency_train_caps.py`) + `cap_at_n` benchmark. Reviewed
adversarially; framing/inference caveats and the partial-fill test folded in.

### The mechanism killer: cascade events are second-scale bursts

Within a chained run, consecutive same-direction tail signals have a **median gap of 1.1 s**
(p90 2.2 s; **96% within 10 s**). So resetting a *5-minute* clock pushes it by ~1 s per event:
the hold barely extends — clock-reset **median hold 5.1 min**, p90 5.4–5.9, max 7.8–9.6 (vs the
5.0 min fixed baseline), despite ~3.1–3.2 resets/run and 74–84% multi-event runs. The premise
(later events *materially* extend the hold) does not operate on this market.

### Results (net bps of $50k; both DDs in $; ret/DD = window net ÷ |MtM DD|)

**DEV (347 kept):**

| Policy | trades | net bps | t_NW | t_clust | Sharpe√365 | realized DD | MtM DD | peak cap | ret/\|DD\| |
|---|---|---|---|---|---|---|---|---|---|
| uncapped | 347 | **34.68** | 3.59 | 3.02 | 3.26 | −4,461 | −47,860 | $900k | 1.26 |
| greedy cap N=5 | 268 | 17.27 | 2.05 | 1.83 | 1.71 | −5,167 | −23,325 | $250k | 0.99 |
| **clock-reset N=5** | 273 | 17.48 | 2.13 | 2.00 | 1.93 | −5,874 | −24,454 | $500k | 0.98 |
| clock-reset N=3 | 204 | 12.65 | 1.67 | 1.57 | 1.59 | −4,525 | −15,091 | $300k | 0.86 |
| clock-reset N=1 | 82 | 11.41 | 1.57 | 1.62 | 1.78 | −1,882 | −5,271 | $100k | 0.89 |

**HOLDOUT (156 kept):**

| Policy | trades | net bps | t_NW | t_clust | Sharpe√365 | MtM DD | peak cap | ret/\|DD\| |
|---|---|---|---|---|---|---|---|---|
| uncapped | 156 | **51.01** | 3.22 | 2.77 | 4.21 | −18,629 | $600k | 2.14 |
| greedy cap N=5 | 119 | **31.44** | 2.05 | 1.71 | 2.63 | −12,994 | $250k | 1.44 |
| **clock-reset N=5** | 119 | 25.13 | **1.52** | 1.26 | 1.89 | −13,183 | $250k | 1.13 |
| clock-reset N=3 | 88 | 27.25 | 1.67 | 1.40 | 2.07 | −8,270 | $150k | 1.45 |
| clock-reset N=1 | 38 | 23.67 | 1.58 | 1.37 | 2.01 | −3,084 | $50k | 1.46 |

**Dev-selected clock-reset = N=5** (max √365 Sharpe among t_NW≥2). It **fails to confirm on the
holdout**: 25.13 bps, **t_NW 1.52 (not significant)**, and it is **below greedy N=5's 31.44**.

### Read

- **Clock-reset ≈ greedy on dev, WORSE on holdout.** Because the clock barely extends, clock-reset
  collapses onto the greedy cap (dev 17.48 vs 17.27) — which F12 already rejected — and on the
  holdout it underperforms greedy (25.13 vs 31.44) and loses significance.
- **The dev "win" is a t_NW artifact.** N=5's dev t_NW 2.13 is not backed by the honest
  episode-clustered t (**t_clust 2.00 dev, 1.26 holdout**); a bounded book's units share one exit
  and are near-perfectly correlated, so t_NW overstates significance. No clock-reset variant is
  robustly significant even on dev.
- **The negative result is not a modelling artifact** (adversarially checked): (i) clock-reset
  caps N *per direction* (≤2N total; dev peak cap $500k vs greedy's $250k), so it runs with the
  **looser** capital budget and still loses — exculpatory, not a handicap; (ii) the synchronized
  aggregated book-flatten is a worst-case exit (spread_exit −1.8…−2.2 vs greedy −0.8…−1.3, ~1 bp)
  but far short of the ~6 bp holdout gap to greedy — the larger effect is lower per-unit gross-mid.
- **Nothing beats uncapped** on Sharpe or ret/|MtM DD|, on either window (as in F12).

### Verdict

**Clock-reset is rejected.** It barely differs from the greedy cap F12 already rejected, and
underperforms it out-of-sample-within-train, because the edge it targets — later cascade events
extending the hold — doesn't exist at scale here: **liquidation cascades fire within ~1 second**,
so a 5-min clock never meaningfully extends. Manipulating the *exit clock* is the wrong lever.

### Next steps

1. The remaining deployable lever is a **causal displacement-gated ENTRY** (enter only on the
   first event clearing a train-estimated displacement bar, then lock out — capturing the violent
   event *without* look-ahead), not exit-clock manipulation. Validate on **fresh** data.
2. The extend-the-hold idea would only matter for an instrument/signal whose qualifying events are
   spaced over **minutes**, not seconds — not BTCUSD_PERP liquidation cascades.
