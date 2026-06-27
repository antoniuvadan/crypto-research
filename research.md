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
