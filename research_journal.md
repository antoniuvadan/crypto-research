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
    confirming the retracement. Net positive once it clears the 10 bp fee:
    marginal at 30s (+0.6, t≈0.3), meaningful at 1min (+6.9, t≈2.9) and 2min
    (+7.9, t≈2.6). Still loses at 5s/10s.
  - Caveats: in-sample, optimistic fills, low per-trade Sharpe (~0.09), execution
    bps are cross-feed/timing-sensitive — trust `gross_mid_to_mid`, not spread terms.

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
