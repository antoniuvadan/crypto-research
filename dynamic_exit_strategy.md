# Strategy: state-dependent (retracement) exit for the reversion trade

**Status:** implemented (Improvement 1 of the dynamic-strategy plan). Mechanism is
correct and wired through the `ExitPolicy` seam; parameter selection and
out-of-sample validation are open.

## Why

The reversion trade (Finding 2, `research.md`) closes on a **fixed clock** —
`decision_time + holding_period` — and the "grid of horizons" (`5s…2min`) is just
five static clocks tested independently. None of them reacts to whether the
reversion has actually happened. That is naive: the same liquidation cascade can
retrace in 3 seconds or grind for two minutes, and a fixed clock either leaves
money on the table or holds a losing position to the bell.

A state-dependent exit makes the holding period **endogenous to each event**: hold
while the reversion is still playing out, close when the L1 book says it is done
(target reached), abandoned (cascade continues), or stale (time cap).

Secondary motive: the F2 edge was *not* significant under Newey-West because
overlapping 1–2 min holds induce serial correlation (HAC SEs ~2× IID). Shorter,
adaptive holds reduce position overlap, which can tighten the error bars on the
same point edge.

## Mechanism

For a reversion trade opened after a cascade, define two L1 mid references:

- `mid_pre` — mid `pre_lookback` before the liquidation (default 5 s before, the
  start of the ±5 s signal window): the pre-cascade level.
- `mid_decision` — mid at `decision_time` (= liquidation_time + 5 s): the
  dislocated level the trade is entered against.

The **cascade displacement** is `favorable = mid_pre − mid_decision`. Reversion
expects the mid to travel from `mid_decision` back toward `mid_pre`, i.e. in the
`favorable` direction, which (for a genuine reversion event) equals the traded
`direction` (`+1` long after a sell cascade, `−1` short after a buy cascade).

Two book-relative levels are set once, at entry:

```
mid_target = mid_decision + take_profit_frac * favorable    # retracement target
mid_stop   = mid_decision − stop_frac        * favorable    # continuation stop
```

The policy then scans L1 updates forward from the entry fill to the time cap and
returns the **first** update where, with `s = direction`:

| Trigger | Condition | Meaning |
|---|---|---|
| Take-profit | `s · (mid_t − mid_target) ≥ 0` | mid retraced `take_profit_frac` of the displacement toward `mid_pre` |
| Stop | `s · (mid_t − mid_stop) ≤ 0` | mid extended `stop_frac` of the displacement past `mid_decision` (cascade continues) |
| Time cap | neither within `max_hold` | reversion never resolved; close at `max_hold` |

The time cap is **also the fallback** when a reference mid is missing or when the
observed dislocation is not in the expected reversion direction
(`s · favorable ≤ 0`) — then no meaningful levels exist and the trade is simply
held to the cap, reproducing the fixed-horizon behavior for that event.

### Timing / no look-ahead

The returned trigger is the **observation** time `t` — the moment the book first
prints a qualifying mid. The runner then adds the same `latency` (300 ms) used at
entry before the exit sweep begins (`exit_start = t + latency`), so the close is
acted on with realistic delay and the policy never uses information past `t`. The
entry sweep is capped at `decision_time + max_hold + latency` (the latest possible
exit), exactly as the fixed path capped it at the static exit.

## Parameters

| Param | Default | Role |
|---|---|---|
| `max_hold` | `2 min` | time cap and entry-sweep bound; the horizon where the F2 gross edge was largest |
| `take_profit_frac` | `0.5` | fraction of displacement to retrace before taking profit |
| `stop_frac` | `1.0` | fraction of displacement of adverse continuation before stopping out |
| `pre_lookback` | `5 s` | how far before the liquidation to sample `mid_pre` |

## Where it lives / how it plugs in

- `backtester.RetracementExit` — the policy (an `ExitPolicy`: exposes `max_hold`,
  implements `exit_trigger_time(ctx, book)`). Reads the L1 book through the lazy
  `BookProvider` seam; no new data plumbing.
- `backtester.ExitContext` — carries `decision_time`, `liquidation_time`,
  `direction`, `latency`, and the realized `entry` fill the policy needs.
- `backtest_momentum.run_liquidation_momentum_model_c_backtests` — already routes
  exits through the policy; pass `exit_policy=RetracementExit(...)` and
  `holding_periods=(max_hold,)`.
- `backtest_reversion_dynamic.py` — the driver over the training window
  (`signal_direction_sign=-1`), writing
  `data/results/reversion_dynamic_exit_{summary,trades}.csv`. The event set is
  identical to the fixed-exit baseline, so trades A/B one-to-one against
  `reversion_model_c_trades.csv`.

## Running

```bash
python backtest_reversion_dynamic.py
```

bookTicker is loaded lazily, one UTC day at a time (LRU-cached), so the run reads
only the days on which signal events fire.

## Preliminary result (in-sample slice — NOT the verdict)

On a 6-week in-sample slice (2023-06-25 → 2023-08-05, n = 86 events, $50k), the
mechanism behaves as designed — exit mix 42 take-profit / 27 stop / 17 time-cap /
0 fallback; realized holds spread from 1 s to 120 s (median ~14 s, mean ~39 s) vs
the flat 120 s of the fixed exit.

Net bps per trade on the slice (same 86 events throughout):

| Exit | net bps |
|---|---|
| fixed 30 s | −17.51 |
| fixed 1 min | −18.78 |
| **fixed 2 min** | **−4.94** |
| dynamic tp=0.5 stop=1.0 | −9.73 |
| dynamic tp=0.5 stop=2.0 | −9.96 |
| dynamic tp=1.0 stop=1.0 | −10.81 |
| dynamic tp=1.0 stop=2.0 | −10.57 |
| dynamic tp=1.5 stop=1.0 | −10.30 |
| dynamic tp=1.5 stop=2.0 | −9.99 |

**Read — two things, and a warning not to over-read either:**

1. **The dynamic exit lands *between* the short and long fixed horizons and does
   not beat fixed-2 min here.** It is also strikingly **insensitive** to the
   take-profit / stop fractions (every config is ~−10 bps). So on this slice the
   take-profit/stop levels are not where the action is — the dominant effect is
   simply that the adaptive exit shortens the average hold (median ~14 s), and on
   this window shorter holds are worse, mirroring the fixed-30 s/1 min rows being
   worse than 2 min.
2. **This slice is unrepresentative of the F2 result.** Fixed-2 min nets **−4.94
   bps here vs +7.90 bps over the full training window** — the early weeks are a
   losing window for the baseline itself. So these numbers characterize the
   mechanism's *behavior*, not its edge; nothing here argues the dynamic exit
   helps or hurts over the full study.

## Caveats

- **In-sample, single slice, small n.** These numbers characterize the mechanism,
  they do not establish an edge. The full training-window run and an out-of-sample
  check are required before believing any configuration.
- **Parameters are unfit.** Defaults are a starting point; `take_profit_frac`,
  `stop_frac`, and `max_hold` should be chosen on the training window and confirmed
  out-of-sample, mindful of in-sample overfitting on three free parameters.
- **Mid references are point lookups** (`as_of`), so a single noisy L1 print at the
  reference instants perturbs the displacement; a short median of nearby quotes
  would be more robust.
- **Still all-taker.** The exit does nothing about the binding ~10 bps round-trip
  fee — that is Improvement 3 (maker entry).

## Possible extensions

- **Flow-exhaustion exit:** close when the same-direction taker pressure that drove
  the cascade decays or flips, reusing the `_same_direction_aggregate_quantities`
  machinery forward in time (needs aggTrades threaded into the policy alongside the
  book).
- **Imbalance / micro-price triggers:** `MarketSnapshot.book_imbalance` and
  `micro_price` are already available on every book update for an exhaustion signal
  that does not depend on a `mid_pre` reference.
- **Trailing stop / target ratchet** once the trade is in profit.
