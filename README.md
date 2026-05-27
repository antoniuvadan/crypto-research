# crypto-research

## Research question: Liquidation cascade reversal

The research question is "do liquidation cascades exhibit mean reversion at
horizon H?".

Additional directions:
1. Condition on volatility regime
2. Can volatility regime alone explain mean-reversion (if it exists). We want
to discern between high-vol mean reversion and liquidation cascades to be able
to tell what the source of alpha is if this is used as a downstream signal.

An important note about the data: "For each symbol，only the largest one liquidation order
within 1000ms will be pushed as the snapshot." This will systematically 
underestimate liquidation volume.

## Data
Limiting scope to COIN-M BTCUSD perps (USD-denominated, BTC-settled BTC perpetuals).

Data required:
- [x] liquidationSnapshot
- [x] aggTrades
  - gives directional aggressor flow -- if one order sweeps through multiple
    price levels, aggTrades reports this into one row
- [x] bookTicker

Study period:
Start: `2023-06-25 04:53:20.357000+00:00`
End:   `2024-10-14 05:27:11.079000+00:00`

Role of **aggTrades**: aggTrades provides more information in addition to
liquidation snapshots. Again, liquidation snapshots historical data only reveals
the largest liquidation over the span of a second. aggTrades can reveal larger
cascades as a result of an initial liquidation event. Isolated liquidations may
not reverse meaningfully; full cascades may.
