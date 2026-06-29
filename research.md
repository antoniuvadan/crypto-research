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
   book; would lift the filtered ~+35–51 toward ~+40–57.
2. Final **test-period OOS** (2024-06-25 → 2024-10-14) — single final run only, kept
   untouched until the strategy is frozen.
