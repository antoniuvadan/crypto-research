# Sunday June 28, 2026

- [x] **QA: lookahead audit of the backtest harness — PASS, no lookahead in the
  decision path.** Traced every step:
  - Trailing threshold (`_trailing_quantiles`): slice `values[left:right]` excludes
    the current + all future events → each event compared only to strictly-past 7d. ✓
  - ±5s signal (`_same_direction_aggregate_quantities`): the ONLY use of post-event
    data ([liq−5s, liq+5s]), but gated — `decision_time = liq + seconds_after` (5s),
    so it fires only after the +5s window is fully observed. ✓
  - Entry `start = decision + latency`; sweep (`_sweep_fill_from_agg_trades`) uses
    `searchsorted(start, "left")` → trades at time ≥ start only, forward-only. ✓
  - Exit `start = trigger + latency`, forward sweep. `RetracementExit` refs are
    causal (`as_of(liq−pre)`, `as_of(decision)`) and the forward scan returns the
    FIRST target/stop hit → only uses book up to the act instant. ✓
  - `BookView.as_of` = `searchsorted("right")−1` (last quote ≤ t). Latency always
    ADDED, never subtracted. ✓
  - Post-hoc tools (`pnl_decomposition`, `market_neutralize`) measure completed
    trades (exit-time mids are what happened, not forecasts); market-neutral drift
    uses a strictly pre-event window. ✓
  - Non-lookahead caveats (separate dimensions): fill realism is optimistic (assumes
    historical tape liquidity available, zero impact); a trade at exactly
    `exit_start` could in principle be double-used by entry+exit sweeps, but entry
    breaks on `remaining<=0` long before, so it ~never fires.

- Built the dynamic-strategy plumbing: book state (`BookProvider`/`BookView`,
  L1 microstructure) is now feedable into the decision path, plus an `ExitPolicy`
  seam in the Model C runner (default `FixedHorizonExit` is byte-identical to the
  old fixed clock). See `dynamic_exit_strategy.md`.
- [x] **Improvement 1 — state-dependent exit** (`backtester.RetracementExit`,
  driver `backtest_reversion_dynamic.py`). Take-profit at a fraction of the cascade
  displacement, stop on continuation, time cap. Mechanism works; on a 6-week slice
  it is insensitive to TP/stop params (~−10 bps) and doesn't beat fixed-2min — but
  that slice is unrepresentative (fixed-2min −4.94 vs +7.90 full study). No
  full-study/OOS verdict yet.

- [x] **Event-selection sweep (entry side)** — `backtest_reversion_event_slices.py`;
  generalized the signal to a percentile **band** (`upper_percentile`). Tested the
  heuristic that the ≥98th tail is non-reverting and the inner region holds the
  edge. **Heuristic is inverted** (see `research.md` Finding 3): reversion strength
  grows *monotonically with cascade size* — at 2min gross runs +3.1 → +3.7 → +6.8
  → **+17.9** bps from [0.50,0.80] up to the ≥98th tail. Every inner band is
  decisively net-negative (n up to 12k, t_IID −15…−100); only the extreme tail
  clears the fee, and only at 1–2min. The original 98th threshold was already
  picking the right events. **Inner-region branch is dead.**

- [x] **Longer horizons on the tail** — `backtest_reversion_long_horizons.py`,
  grid [1m,5m,30m,60m] on the ≥98th events. **Breakthrough** (see `research.md`
  Finding 4): gross does NOT plateau at 2min — it nearly doubles by 5min (+16.9 →
  +33.7 bp), holds at 30min, fades by 60min. **5min nets +23.67 bp with t_NW=4.33**
  (auto L=6), and stays **t_NW=3.43 at L=100** — the first config to survive
  Newey-West (1–2min never did). Not just bull-market beta: at 5min BOTH directions
  profit (long +27.2, short +13.5 bp); a pure long-drift artifact would make shorts
  lose. Strongest result so far, but in-sample / optimistic fills / not yet
  market-neutralized.

- [x] **Sub-period stability of the 5min tail edge** (NOT true OOS — used train
  months, kept the test period untouched). Sliced the Finding-4 5min trades by
  calendar month (causal threshold ⇒ slice == standalone run). **11/13 train months
  net-positive**, longs +ve in all but the 5-day partial first month. But the edge
  is **time-varying / regime-concentrated**: strongest Dec23–Mar24 bull run (+39 to
  +66 bp), weak/negative over summer 2023; per-month n=30–134 so per-month t_NW is
  low-powered. Regime-concentration is a real risk for the real OOS (test = Jun–Oct
  2024, a possibly-different regime — that's why it stays untouched). See
  `research.md` Finding 4 → "Sub-period stability".

- [x] **Market-neutralized the 5min tail trades** (`market_neutralize.py`,
  event-study abnormal return: signal minus the causal pre-cascade trend over a
  60min window). **Edge is alpha, not drift — and the trend was AGAINST it.** The
  pre-event drift is negative (−5.4 bp in trade direction: cascades follow
  sell-offs, so price falls into the event and the reversion bets against the local
  downtrend), so removing it *raises* the alpha: raw mid signal +31.96 → **abnormal
  +37.36 bp** (t_IID +13.0, **t_NW +6.59**), net-tradeable +27.36. Both directions
  revert against trend (long +41.1, short +26.8); 12/13 train months +ve on
  abnormal. Decisively kills the bull-drift worry. See `research.md` Finding 4 →
  "Market-neutral".

- [x] **P&L-decomposed the 5min trades** (`pnl_decomposition.py`; extended its
  HOLDING_ORDER like significance.py). **Edge is the signal, not execution.** 5min
  net +23.67 ≈ gross_mid_to_mid (+31.91) − 10 bp fee; latency +0.24, spread terms
  slightly *favorable* (−2.0 bp combined). Three measures of the signal agree:
  gross-mid +31.91 ≈ market-neutral raw +31.96 ≈ fill-gross +33.67. Fees are the
  dominant friction. Caveat: favorable spread = optimistic sweep fills; a
  conservative ~2 bp/side taker cost gives ~+18 bp net (honest floor). See
  `research.md` Finding 4 → "P&L decomposition".

TODO (next):
- [ ] **Maker entry (Step 2)** — fees (10 bp) are now the clear binding friction;
      maker-in/taker-out ~3 bp, maker/maker ~6 bp round-trip on top of the +23.7 net.
- [ ] **TRUE OOS** on the test period (2024-06-25 → 2024-10-14) — *final run only*,
      keep untouched until the above are done.
- [ ] Improvement 3 / Step 2 — maker entry (kill the ~10 bp fee); orthogonal,
      lifts the 5min net toward ~+27 bp.
- [ ] Improvement 2 (book-aware entry gating/sizing) — inner-region *size* selection
      is ruled out (F3); any gating should key off spread/imbalance, not magnitude.
- [ ] Full-study + OOS run of the dynamic exit (Improvement 1) before any verdict.


# Saturday June 27, 2026

- Step 1 (P&L decomposition) done. Split Model C net P&L per trade into
  gross_mid_to_mid, latency, entry/exit spread, and fees (bps, sign-corrected);
  reconciles exactly to realized net. Tool: `pnl_decomposition.py`, outputs in
  `data/results/`. Full writeup: `research.md` Finding 1.
- Result: the momentum signal is **dead and inverted** at every horizon.
  `gross_mid_to_mid` is negative before any friction and worsens monotonically
  with horizon (−3.2 bp @5s → −15.7 bp @2min) — the signature of mean-reversion
  (price retraces against the liquidation-flow direction). Net loss is dominated
  by the 10 bp round-trip taker fee; spread ≈ 2 bp; latency negligible and
  slightly favorable (−0.24 bp); size impact negligible ($50k ≈ $100k).
- Implication: a naive reversion flip only clears friction at 1–2 min, and
  thinly. Needs maker entry (kill the 10 bp fee) and/or event selection.

- [x] Flipped to reversion (`backtest_reversion.py`, `signal_direction_sign=-1`;
  identical 969-event set, opposite trade direction). See `research.md` Finding 2.
  - `gross_mid_to_mid` flips sign almost exactly (+3.2 bp @5s → +15.7 bp @2min),
    confirming the retracement. Net positive (point estimate) at 1min (+6.9) and
    2min (+7.9) once it clears the 10 bp fee; still loses at 5s/10s.
  - **Newey-West kills the significance** (`significance.py`, HAC L=6): SEs ~2× the
    IID ones (event clustering + overlap). 1–2min t_NW ≈ 1.3–1.5 (was IID ~2.6–2.9)
    → NOT significant, p≈0.15. Momentum stays decisively negative (t_NW −3.7…−6.4).
  - Caveats: in-sample, optimistic fills, execution bps are cross-feed/timing-
    sensitive — trust `gross_mid_to_mid`, not spread terms. Don't trade as-is.

TODO (next):
- [ ] **Step 2 — maker entry** (highest leverage): 10 bp taker fee eats ~2/3 of the
      gross edge; maker-in/taker-out saves ~3 bp/round-trip → 30s positive, 1–2 min ~+10 bp.
- [ ] Out-of-sample / robustness check on the 1–2 min edge before believing it.
- [ ] Step 3 — filter events by liquidation magnitude / vol regime.


# Friday May 29, 2026

- Graphed price and liquidations around the liquidation with maximum volume in
the [-5s, 5s] window. Found that around this particular selloff / liquidation 
the price mean reverts at much higher frequencies (1-2min) because, at larger
time horizons, the price continues to move downward

TODO:
- [ ] cluster liquidations + aggTrades around liquidation clusters
- [ ] EDA: plot signed returns around liquidation events

- Plot realized vol at the following configs against liquidation events:
```    
    ("5s",   5,   0.1),   # 5s window,   100ms sampling
    ("30s",  30,  0.3),   # 30s window,  300ms sampling
    ("1min", 60,  0.5),   # 1min window, 500ms sampling
    ("5min", 300, 2.0),   # 5min window, 2s sampling
```
  - found what's expected -- higher volatility around liquidations. This was
  expected to be true by construction of what constitutes a liquidation event.


# Tuesday May 26, 2026

- About the research process for this problem:
  - Branches you should climb (because they directly inform the trunk):
    - What's the right event definition (size threshold, aggTrades-confirmed cascades, etc.)?
    - **What's the time horizon of reversion if it exists?**
    - **How does the signal vary by vol regime?**
    - **How does the signal vary by liquidation magnitude?**
    - How is realized vol related to liquidation frequency, and what does the lead-lag tell us about mechanism?
- Branches to note but not to climb (yet):
  - Is there alpha in cross-exchange basis dynamics during cascades?
  - Do similar dynamics exist on COIN-M altcoin perps?
  - What's the relationship between liquidations and spot ETF flow?

- found period of very high spread starting 2023-07-13 21:09:54.146 UTC

- use `accumulated_fill_quantity` in the liquidation snapshot data to get
  number of contracts actually filled
  - use `average_price` for the average underlying price of BTC

TODO:
- [x] realized vol
  - [ ] liquidation rate per volatility regime
    - vol regime = decile of volatility at the minute level
  - use volatility of log returns, not price; log returns are additive,
    more symmetric and closer to Gaussian, which is exactly where std dev is 
    more interpretable
  - on sampling frequency: using every sample from L1 data is too noisy. Compute
    mid and sample 1 per second
  - window:
    - 5s, 30s, 1min, 5min
      - 5s: is something unusual happening right now
      - 30s: is something unusual happening this minute
      - 1min and 5min get closer to volatility regime change
    - must match each volatility window to an adequate sampling frequency
      - 5s: 100ms
      - 30s: 300ms
      - 1min: 500ms
      - 5min: 2s
  - bipower variation?
  - could aim to answer "do liquidations lead or lag vol regime changes?"
  - compute volatility across 5s, 30s, 1min, 5min windows 
    **per liquidation event**
- [ ] aggregate taker flow vs. liquidation events visualization.
  - For a few visually striking days (highest liquidation count days), plot:
    - Per-minute net taker flow from aggTrades (signed by direction)
    - Per-minute total liquidation notional (signed by direction)
    - Mid-price
  - overlaid on the same time axis. Look at three or four such days manually.
    - Why: this is the qualitative check that aggTrades and liquidationSnapshot 
    tell consistent stories about cascades. If you see large negative taker flow
    lines up with clusters of long-liquidation events (sells) and corresponds 
    to sharp price drops, the data is internally consistent. 
    If they don't align, something's wrong with one of the datasets or your 
    understanding of them.
    - this is also where intuition lives. You'll start noticing patterns: 
      - do cascades tend to happen at certain times of day? 
      - do they cluster around macro news? 
      - do they show characteristic shapes (long buildup then sharp drop, or sudden drop?). 
      - rhis intuition guides everything downstream.


# Tuesday May 19, 2026

- Mondays show the smallest amount of liquidation events over the one-month
  2023-06-25 - 2023-07-25 period. Fridays show the largest.
  - Actionable?


# Monday May 11, 2026

- write down research project steps in exploration.ipynb
- lazy scan first 1 month of agg_trades and liquidation_snapshots
- compute agg_qty_5s_before_5s_after around liquidation events for the first
  month
- compute the percentile of agg_qty_5s_before_5s_after value for each row in
  the liquidation snapshot dataframe (actually a lazyframe) within the rolling
  window of last 7 days (TODO: change to 30 for real study)

- goal is to do exploratory analysis before doing the t-test across multiple
  time windows


# Tuesday May 12, 2026

- the distribution of agg_qty_5s_before_5s_after is very fat-tailed, as expected
- for the 2023-06-25 - 2023-07-25 period, the 95th percentile of the 
  agg_qty_5s_before_5s_after column is 126070.0, which is 
  $126070.0 \times \$100 = \$12,607,000$ of notional in perps (**Note:** changed
  to $50 notional as of 2026-04-14 at 06:30 (UTC) -- https://www.binance.com/en/square/post/04-09-2026-binance-to-adjust-minimum-notional-value-for-btc-perpetual-futures-310575614771425).
- above 95th percentile there are 16 BUY liquidations and 104 SELL liquidations
