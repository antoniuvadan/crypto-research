# crypto-research

## Research question: Liquidation cascade reversal

The research question is "do liquidation cascades exhibit mean reversion at
horizon H?". To answer that, we need (a) liquidation events, (b) price series at
horizon H, (c) some null/baseline distribution to compare against. All of these
are available in the public archive. The cleanest version of the answer is
"yes/no, with these conditional caveats" and it's defensible without ever
opening a book.

An important note: "For each symbol，only the largest one liquidation order
within 1000ms will be pushed as the snapshot." This will systematically 
underestimate liquidation volume.

## Data
Limiting scope to COIN-M BTCUSD perps.

Required:
- [x] liquidationSnapshot
- [x] aggTrades
  - gives directional aggressor flow -- if one order sweeps through multiple
    price levels, aggTrades reports this into one row
- [x] bookTicker

Not for the main research question:
- [ ] open interest
   - this is the `metrics` dataset
   - freq: 5min
   - do i need? aggTrades contains a measure of volume

Study period:
Start: `2023-06-25 04:53:20.357000+00:00`
End:   `2024-10-14 05:27:11.079000+00:00`

Role of **aggTrades**: aggTrades provides more information in addition to
liquidation snapshots. Again, liquidation snapshots historical data only reveals
the largest liquidation over the span of a second. aggTrades can reveal larger
cascades as a result of an initial liquidation event. Isolated liquidations may
not reverse meaningfully; full cascades may.
