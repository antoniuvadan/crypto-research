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
reconciles exactly (`net_reconstructed == net_realized` to 3 decimals). t-stats
are `net_sharpe × sqrt(969)`.

### Interpretation

- **`gross_mid_to_mid` flips sign almost exactly** vs Finding 1 (−3.23 → +3.24,
  …, −15.72 → +15.72). Clean confirmation that the mid retraces against the
  liquidation flow; reversion captures it. Edge grows monotonically with horizon
  and is consistent across both sizes — not a cherry-picked cell.
- **Fees unchanged at 10 bps round-trip** — still the largest single line and
  still what sinks the short horizons.
- **Reversion is net-positive only once the gross edge clears 10 bps**: marginal
  at 30s (+0.6 bps, t≈0.3 — indistinguishable from zero), meaningful at **1min
  (+6.9, t≈2.9)** and **2min (+7.9, t≈2.6)**.
- `latency` flips to a tiny cost (+0.24 bps); spread/execution terms are small
  (~1–2 bps) and don't change the conclusion.

### How much to believe it

Positive but hold loosely:
- **In-sample** (training window only) — no out-of-sample confirmation yet.
- **Optimistic fills** — sweep assumes the historical tape was available and the
  order adds no impact.
- **Low per-trade Sharpe (~0.09)**; the ~2.6–2.9 t-stats are in-sample and assume
  IID trades. "Promising," not "established."
- **Execution/spread terms cross two feeds** (aggTrades vs bookTicker) and are
  timing-sensitive around cascades — lean on `gross_mid_to_mid` (single-feed,
  robust), not the execution bps.

One-liner: the gross reversion edge is real and sizable at 1–2 min (~+15 bps),
and it's net-positive because that finally exceeds the 10 bps fee — but barely,
and only at the long end.

### Next steps

1. **Step 2 — maker entry (highest leverage).** The 10 bps taker fee eats ~2/3 of
   the gross edge. Maker is 2 bps vs 5 bps; even maker-in/taker-out saves ~3 bps
   round-trip, which would push 30s clearly positive and 1–2 min toward ~+10 bps.
2. Out-of-sample / robustness: confirm on a held-out period before believing the
   1–2 min edge.
3. Step 3 — event selection (liquidation magnitude / vol regime) to concentrate
   the edge.
