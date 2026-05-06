# crypto-research

## Research question: Liquidation cascade reversal.

When a price move triggers mass liquidations on Binance, the forced market-sell (or market-buy) flow temporarily pushes price beyond fundamental value. Reversal may follow. Binance publishes a forceOrder websocket stream and historical liquidation data — can identify clusters of large liquidations in near-real-time. 

The hypothesis: when liquidation volume in a 5-minute window exceeds the Xth percentile of the trailing distribution, expect mean reversion over the next 30-120 minutes (for example). Note that this is structural — liquidations are forced trades, not informed trades, so the price impact is mechanical and should reverse. 

The well-trodden version of this idea exists ("buy the wicks") but the rigorous version (calibrating thresholds, modeling slippage on entry, conditional on volatility regime, capacity analysis) is genuinely under-studied.
