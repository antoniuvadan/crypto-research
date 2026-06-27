# crypto-research — CLAUDE.md

## Project overview

Research into whether **liquidation cascades on BTCUSD_PERP (Binance COIN-M perpetual)** exhibit momentum or mean-reversion at various horizons, and whether that signal is tradeable after fees.

The instrument is a USD-denominated, BTC-settled BTC perpetual futures contract. One contract = $100 notional (note: Binance changed minimum notional to $50 in April 2026, but the historical data and backtester use $100).

Study period: **2023-06-25 → 2024-10-14**.

## Research direction

Primary question: do large liquidations (top 2% by same-direction aggTrade flow in a trailing 7-day window) predict short-term **momentum** (not mean reversion)?

Open sub-questions (not yet climbed):
- How does the signal vary by volatility regime?
- How does the signal vary by liquidation magnitude?
- Is there alpha in cross-exchange basis dynamics during cascades?
- Do similar dynamics exist on COIN-M altcoin perps?

Key caveat in the raw data: Binance only pushes **the largest liquidation within each 1000ms window** per symbol, so raw `liquidationSnapshot` systematically underestimates liquidation volume.

## Key files

| File | Purpose |
|---|---|
| `backtester.py` | Reusable backtesting engine + library: event-driven `Backtester`, data structures (`MarketSnapshot`/`Order`/`Fill`/`PortfolioSnapshot`/`Strategy`), data loaders, fill simulation (`_sweep_fill_from_agg_trades`), and metrics. No strategy-specific logic. |
| `backtest_momentum.py` | The liquidation signal definition + momentum strategy (`LiquidationMomentumStrategy`, `_same_direction_aggregate_quantities`) and the run drivers (naive book-ticker + Model C sensitivity). Imports the engine from `backtester.py`. Has a `__main__` that runs the Model C grid. |
| `backtest_reversion.py` | Reversion variant — same signal/events, trade direction flipped (`signal_direction_sign=-1`). Reuses the Model C runner from `backtest_momentum.py`. |
| `pnl_decomposition.py` | Decomposes Model C net P&L per trade into gross-mid, latency, spread, and fees (bps). Takes `--trades`/`--label`; reads a trades CSV + bookTicker. |
| `downloader.py` | Downloads `liquidationSnapshot` zips from Binance Vision, extracts CSVs, adds `time_datetime`, saves as parquet |
| `research.md` | Dated, self-contained research findings (one section per finding) |
| `research_journal.md` | Dated research notes and open TODOs |
| `README.md` | High-level research question and data overview |

Jupyter notebooks (`.ipynb`) exist but **do not read them** — they are too token-heavy. All logic that matters lives in the `backtest*.py` files.

## Data directory layout

```
data/
  BTCUSD_PERP-liquidationSnapshot/   *.parquet, one per day
  BTCUSD_PERP-aggTrades/             *.parquet, one per day
  BTCUSD_PERP-bookTicker/            *.zip (raw Binance downloads)
  samples/                            small sample files for quick iteration
  results/                            derived backtest / analysis output CSVs
```

bookTicker data is still in `.zip` format. The other two datasets have been processed to parquet.

## Data schemas

All timestamps are UTC. Parquet files are loaded with polars.

### `liquidationSnapshot` (used columns)

| Column | Type | Notes |
|---|---|---|
| `time_datetime` | Datetime (UTC) | Derived from `time` (ms epoch); timestamp of the liquidation event |
| `side` | String | `"BUY"` or `"SELL"` — the side of the liquidation order |
| `original_quantity` | Float | Contracts originally submitted |
| `accumulated_fill_quantity` | Float | Contracts actually filled; use this for realized liquidation volume |
| `price` | Float | Order price |
| `average_price` | Float | Average fill price of the underlying BTC |
| `order_status` | String | e.g. `"FILLED"`, `"PARTIALLY_FILLED"` |

Direction convention: `SELL` liquidation = a long position being liquidated (forced seller); `BUY` liquidation = a short being liquidated (forced buyer).

### `aggTrades` (used columns)

| Column | Type | Notes |
|---|---|---|
| `transact_time_datetime` | Datetime (UTC) | Trade timestamp |
| `quantity` | Float | Contracts traded in this aggregated trade |
| `price` | Float | Trade price |
| `is_buyer_maker` | Bool | `True` = the buyer is the passive/maker side (i.e. a taker sell) |

`is_buyer_maker` convention: `True` → seller was the aggressor (taker sell); `False` → buyer was the aggressor (taker buy).

### `bookTicker` (used columns)

| Column | Type | Notes |
|---|---|---|
| `event_time_datetime` | Datetime (UTC) | Timestamp of the L1 update |
| `best_bid_price` | Float | |
| `best_bid_qty` | Float | Contracts at best bid |
| `best_ask_price` | Float | |
| `best_ask_qty` | Float | Contracts at best ask |

A derived `mid_price` column is added on load: `(best_bid_price + best_ask_price) / 2`.

## Exchange / contract details

- **Exchange**: Binance COIN-M futures
- **Contract**: BTCUSD_PERP (linear in USD terms, settled in BTC)
- **Contract notional**: $100 USD per contract
- **Taker fee**: 5 bps (0.05%) per leg — default for non-VIP
- **Maker fee**: 2 bps (0.02%) per leg
- Fees are charged on notional: `abs(contracts) * $100 * fee_rate`

## Strategy: Liquidation Momentum (`LiquidationMomentumStrategy`)

Signal logic (implemented in `backtest_momentum.py`):

1. For each liquidation event, compute `agg_qty_5s_before_5s_after`: same-direction aggTrade volume in the [-5s, +5s] window around the liquidation time.
2. Compare against a trailing 7-day 98th-percentile threshold.
3. If the event exceeds the threshold → signal fires `seconds_after` seconds after the liquidation time (i.e. once the full window is observable).
4. Trade **in the same direction** as the liquidating flow (momentum, not mean-reversion).
5. Close after a fixed holding period (tested: 5s, 10s, 30s, 1min, 2min).

### Execution models

| Model | Description |
|---|---|
| **Naive / book-ticker** | Uses live L1 ask/bid qty as trade size; no latency; optimistic |
| **Model C** | Fixed notional (`trade_notional_usd / $100` contracts); skips a 300ms latency window; sweeps through same-side `aggTrades` for realistic fill simulation |

## Running the backtest

```bash
python backtest_momentum.py    # momentum (trade with the flow)
python backtest_reversion.py   # reversion (trade against the flow)
```

`backtest_momentum.py` runs `run_liquidation_momentum_model_c_sensitivity()` over the full study period, across holding periods `[5s, 10s, 30s, 1min, 2min]` and trade sizes `[$50k, $100k]`. Outputs (written under `data/results/`):

- `data/results/liquidation_momentum_model_c_summary.csv` / `..._trades.csv`
- `backtest_reversion.py` → `data/results/reversion_model_c_summary.csv` / `..._trades.csv`

Then decompose the P&L:

```bash
python pnl_decomposition.py                                              # momentum
python pnl_decomposition.py --trades data/results/reversion_model_c_trades.csv --label reversion_decomposition
```

Progress is written to stderr; summary table is printed to stdout.

## Coding conventions

- **Polars** for all dataframes (not pandas, except in `downloader.py` for the parquet write step).
- **Lazy evaluation**: load data as `pl.LazyFrame` and `.collect()` only when needed.
- Data loaders (`load_book_ticker`, `load_liq_snap`, `load_agg_trades`) accept optional `start_date`/`end_date` for filtering.
- Strategy objects implement the `Strategy` protocol: `on_book(book: MarketSnapshot, portfolio: PortfolioSnapshot) -> Iterable[Order] | Order | None`.
- Progress output goes to **stderr**; results to **stdout**.
- No type: ignore, no suppressed warnings.
