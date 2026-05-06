# crypto-research

## Research question: Liquidation cascade reversal

When a price move triggers mass liquidations on Binance, the forced market-sell
(or market-buy) flow temporarily pushes price beyond fundamental value. Reversal
may follow. Binance publishes a forceOrder websocket stream and historical
liquidation data — can identify clusters of large liquidations in
near-real-time. 

The hypothesis: when liquidation volume in a 5-minute window exceeds the Xth
percentile of the trailing distribution, expect mean reversion over the next
30-120 minutes (for example). Note that this is structural — liquidations are
forced trades, not informed trades, so the price impact is mechanical and should
reverse. 

The well-trodden version of this idea exists ("buy the wicks") but the rigorous
version (calibrating thresholds, modeling slippage on entry, conditional on
volatility regime, capacity analysis) is genuinely under-studied.

The research question is "do liquidation cascades exhibit mean reversion at
horizon H?". To answer that, we need (a) liquidation events, (b) price series at
horizon H, (c) some null/baseline distribution to compare against. All of these
are available in the public archive. The cleanest version of the answer is
"yes/no, with these conditional caveats" and it's defensible without ever
opening a book.

Where L2 would actually matter is in two specific parts of the project:

1. Execution modeling for the backtest entry. When you buy a wick, what slippage do
you face? At small size (say <$50k notional on BTCUSDT), bookTicker plus
aggTrades is enough — you can see top-of-book spread, you can see how aggTrades
cleared in the seconds after the liquidation, and you can estimate effective
slippage from "if I had market-bought $X at this moment, what would my average
fill have been" using the actual trade tape that followed. This is a recognized
methodology (it's how a lot of execution research is done when full book is
unavailable). At larger size ($1M+), you'd need full book depth to model
honestly. For a one-month research project trading at small notional, bookTicker
+ aggTrades is sufficient.  

1. Capacity analysis. "At what AUM does this strategy
stop working?" This is harder without full book. You can do a rough version
using aggTrades — measure typical traded volume in the windows you'd be active
in, and reason about what fraction of that volume your strategy would represent
at different AUMs. It's cruder than book-based capacity modeling but it's
defensible if you're explicit about the methodology and its limitations.
