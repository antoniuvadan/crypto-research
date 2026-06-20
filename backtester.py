from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol

import polars as pl


DEFAULT_BOOK_TICKER_PATH = Path("data/BTCUSD_PERP-bookTicker")
CONTRACT_NOTIONAL_USD = 100.0


@dataclass(frozen=True)
class MarketSnapshot:
    """One L1 book update."""

    time: datetime
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0


@dataclass(frozen=True)
class Order:
    """Market order in contracts. Positive quantity buys, negative quantity sells."""

    quantity: float


@dataclass(frozen=True)
class Fill:
    time: datetime
    quantity: float
    price: float
    fee: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    time: datetime
    position: float
    avg_entry_price: float | None
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    equity: float
    mid_price: float


class Strategy(Protocol):
    """Implement this protocol to backtest any signal or trading rule."""

    def on_book(self, book: MarketSnapshot, portfolio: PortfolioSnapshot) -> Iterable[Order] | Order | None:
        ...


def _date_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def load_book_ticker(
    book_ticker_path: Path = DEFAULT_BOOK_TICKER_PATH,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.LazyFrame:
    """Load L1 book ticker data using the same scan/filter/sort pattern as the notebook."""

    df = pl.scan_parquet(book_ticker_path / "*.parquet")

    if start_date is not None:
        df = df.filter(pl.col("event_time_datetime") >= _date_start(start_date))
    if end_date is not None:
        df = df.filter(pl.col("event_time_datetime") < _date_start(end_date) + timedelta(days=1))

    return (
        df.with_columns(
            ((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2.0).alias("mid_price")
        )
        .select(
            [
                "event_time_datetime",
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "mid_price",
            ]
        )
        .sort("event_time_datetime")
    )


class Backtester:
    def __init__(
        self,
        book: pl.LazyFrame | pl.DataFrame,
        strategy: Strategy,
        initial_cash: float = 0.0,
        initial_position: float = 0.0,
        initial_entry_price: float | None = None,
        fee_rate: float = 0.0,
        contract_notional_usd: float = CONTRACT_NOTIONAL_USD,
    ) -> None:
        self.book = book
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.initial_position = initial_position
        self.initial_entry_price = initial_entry_price
        self.fee_rate = fee_rate
        self.contract_notional_usd = contract_notional_usd

    def run(self) -> dict[str, pl.DataFrame]:
        book_df = self.book.collect() if isinstance(self.book, pl.LazyFrame) else self.book

        position = self.initial_position
        avg_entry_price = self.initial_entry_price
        realized_pnl = 0.0
        fees_paid = 0.0
        fills: list[Fill] = []
        portfolio_rows: list[PortfolioSnapshot] = []

        for row in book_df.iter_rows(named=True):
            book = MarketSnapshot(
                time=row["event_time_datetime"],
                bid_price=float(row["best_bid_price"]),
                bid_qty=float(row["best_bid_qty"]),
                ask_price=float(row["best_ask_price"]),
                ask_qty=float(row["best_ask_qty"]),
            )
            portfolio = PortfolioSnapshot(
                time=book.time,
                position=position,
                avg_entry_price=avg_entry_price,
                cash=self.initial_cash + realized_pnl - fees_paid,
                realized_pnl=realized_pnl,
                unrealized_pnl=self._unrealized_pnl(position, avg_entry_price, book.mid_price),
                fees_paid=fees_paid,
                equity=self._equity(realized_pnl, fees_paid, position, avg_entry_price, book.mid_price),
                mid_price=book.mid_price,
            )

            orders = self.strategy.on_book(book, portfolio)
            if orders is None:
                orders_iter: Iterable[Order] = []
            elif isinstance(orders, Order):
                orders_iter = [orders]
            else:
                orders_iter = orders

            for order in orders_iter:
                if order.quantity == 0:
                    continue

                fill_price = book.ask_price if order.quantity > 0 else book.bid_price
                fee = abs(order.quantity) * self.contract_notional_usd * self.fee_rate

                realized_delta, position, avg_entry_price = self._apply_fill(
                    position=position,
                    avg_entry_price=avg_entry_price,
                    fill_quantity=order.quantity,
                    fill_price=fill_price,
                )
                realized_pnl += realized_delta
                fees_paid += fee
                fills.append(Fill(book.time, order.quantity, fill_price, fee))

            portfolio_rows.append(
                PortfolioSnapshot(
                    time=book.time,
                    position=position,
                    avg_entry_price=avg_entry_price,
                    cash=self.initial_cash + realized_pnl - fees_paid,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=self._unrealized_pnl(position, avg_entry_price, book.mid_price),
                    fees_paid=fees_paid,
                    equity=self._equity(realized_pnl, fees_paid, position, avg_entry_price, book.mid_price),
                    mid_price=book.mid_price,
                )
            )

        return {
            "portfolio": pl.DataFrame(portfolio_rows),
            "fills": pl.DataFrame(fills),
        }

    def _apply_fill(
        self,
        position: float,
        avg_entry_price: float | None,
        fill_quantity: float,
        fill_price: float,
    ) -> tuple[float, float, float | None]:
        if position == 0 or position * fill_quantity > 0:
            new_position = position + fill_quantity
            new_avg_entry = self._weighted_entry(position, avg_entry_price, fill_quantity, fill_price)
            return 0.0, new_position, new_avg_entry

        closing_qty = min(abs(position), abs(fill_quantity))
        closed_position = closing_qty if position > 0 else -closing_qty
        realized_pnl = self._pnl_usd(closed_position, avg_entry_price, fill_price)
        new_position = position + fill_quantity

        if new_position == 0:
            return realized_pnl, 0.0, None
        if position * new_position > 0:
            return realized_pnl, new_position, avg_entry_price
        return realized_pnl, new_position, fill_price

    def _weighted_entry(
        self,
        position: float,
        avg_entry_price: float | None,
        fill_quantity: float,
        fill_price: float,
    ) -> float:
        if position == 0 or avg_entry_price is None:
            return fill_price
        return (
            abs(position) * avg_entry_price + abs(fill_quantity) * fill_price
        ) / (abs(position) + abs(fill_quantity))

    def _pnl_usd(self, position: float, entry_price: float | None, exit_price: float) -> float:
        if position == 0 or entry_price is None:
            return 0.0
        return position * self.contract_notional_usd * (exit_price / entry_price - 1.0)

    def _unrealized_pnl(
        self, position: float, avg_entry_price: float | None, mid_price: float
    ) -> float:
        return self._pnl_usd(position, avg_entry_price, mid_price)

    def _equity(
        self,
        realized_pnl: float,
        fees_paid: float,
        position: float,
        avg_entry_price: float | None,
        mid_price: float,
    ) -> float:
        return (
            self.initial_cash
            + realized_pnl
            + self._unrealized_pnl(position, avg_entry_price, mid_price)
            - fees_paid
        )


class TargetPositionStrategy:
    """Adapter for signal functions that return desired position in contracts."""

    def __init__(self, signal_fn) -> None:
        self.signal_fn = signal_fn

    def on_book(self, book: MarketSnapshot, portfolio: PortfolioSnapshot) -> Order | None:
        target_position = float(self.signal_fn(book, portfolio))
        delta = target_position - portfolio.position
        return Order(delta) if delta else None
