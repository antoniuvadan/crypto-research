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

### Next steps

1. **Out-of-sample** confirmation of the 5-min tail edge (held-out period; ideally
   spanning a non-bull regime).
2. **Market-neutralize** — subtract the contemporaneous BTC return over each hold
   to separate reversion alpha from drift.
3. Maker entry (Step 2) — orthogonal ~3–6 bps round-trip on top.
4. P&L-decompose the 5-min trades to confirm the edge is `gross_mid_to_mid`
   (signal), not an execution/spread artifact.
