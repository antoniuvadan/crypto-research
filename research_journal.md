# Tuesday May 26, 2026
- found period of very high spread starting 2023-07-13 21:09:54.146 UTC

- use `accumulated_fill_quantity` in the liquidation snapshot data to get
  number of contracts actually filled
  - use `average_price` for the average underlying price of BTC

TODO:
- [ ] realized vol for regime analysis
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
