from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import polars as pl


DEFAULT_BOOK_TICKER_PATH = Path("data/BTCUSD_PERP-bookTicker")
DEFAULT_LIQ_SNAP_PATH = Path("data/BTCUSD_PERP-liquidationSnapshot")
DEFAULT_AGG_TRADES_PATH = Path("data/BTCUSD_PERP-aggTrades")
CONTRACT_NOTIONAL_USD = 100.0
MAKER_FEE_RATE = 0.0002
TAKER_FEE_RATE = 0.0005
ROUND_TRIP_TAKER_FEE_RATE = TAKER_FEE_RATE * 2.0
DEFAULT_LATENCY = timedelta(milliseconds=300)
DEFAULT_TRADE_NOTIONAL_USD = 50_000.0
DEFAULT_TRADE_NOTIONAL_USD_GRID = (50_000.0, 100_000.0)

LIQ_SNAP_COLS = [
    "time_datetime",
    "side",
    "original_quantity",
    "accumulated_fill_quantity",
    "price",
    "average_price",
    "order_status",
]

AGG_TRADES_COLS = [
    "transact_time_datetime",
    "quantity",
    "price",
    "is_buyer_maker",
]


class ProgressPrinter:
    """Minimal tqdm-like progress printer for CLI runs."""

    def __init__(
        self,
        label: str,
        total: int,
        update_every: int = 10_000,
        min_interval_s: float = 1.0,
    ) -> None:
        self.label = label
        self.total = total
        self.update_every = max(update_every, 1)
        self.min_interval_s = min_interval_s
        self.start_time = time.monotonic()
        self.last_update = 0
        self.last_render_time = self.start_time

    def update(self, completed: int, force: bool = False) -> None:
        now = time.monotonic()
        row_threshold_met = completed - self.last_update >= self.update_every
        time_threshold_met = now - self.last_render_time >= self.min_interval_s
        if not force and not row_threshold_met and not time_threshold_met:
            return

        self.last_update = completed
        self.last_render_time = now
        elapsed = now - self.start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        pct = completed / self.total if self.total else 1.0
        bar_width = 30
        filled = min(bar_width, int(bar_width * pct))
        bar = "#" * filled + "-" * (bar_width - filled)
        message = (
            f"\r{self.label} [{bar}] {completed:,}/{self.total:,} "
            f"({pct:6.2%}) {rate:,.0f} rows/s"
        )
        sys.stderr.write(message)
        sys.stderr.flush()

        if completed >= self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _progress_message(label: str | None, message: str) -> None:
    if label is None:
        return
    print(f"{label}: {message}", file=sys.stderr, flush=True)


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


def load_liq_snap(
    liq_snap_path: Path = DEFAULT_LIQ_SNAP_PATH,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.LazyFrame:
    """Load liquidation snapshots using the same columns and date filter as the notebook."""

    df = pl.scan_parquet(liq_snap_path / "*.parquet").select(LIQ_SNAP_COLS)

    if start_date is not None:
        df = df.filter(pl.col("time_datetime") >= _date_start(start_date))
    if end_date is not None:
        df = df.filter(pl.col("time_datetime") < _date_start(end_date) + timedelta(days=1))

    return df.unique(subset=["time_datetime"]).sort("time_datetime")


def load_agg_trades(
    agg_trades_path: Path = DEFAULT_AGG_TRADES_PATH,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.LazyFrame:
    """Load aggregate trades using the same columns and date filter as the notebook."""

    df = pl.scan_parquet(agg_trades_path / "*.parquet").select(AGG_TRADES_COLS)

    if start_date is not None:
        df = df.filter(pl.col("transact_time_datetime") >= _date_start(start_date))
    if end_date is not None:
        df = df.filter(
            pl.col("transact_time_datetime") < _date_start(end_date) + timedelta(days=1)
        )

    return df.sort("transact_time_datetime")


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

    def run(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, pl.DataFrame]:
        book_df = self.book.collect() if isinstance(self.book, pl.LazyFrame) else self.book
        total_rows = len(book_df)
        if progress_callback is not None:
            progress_callback(0, total_rows)

        position = self.initial_position
        avg_entry_price = self.initial_entry_price
        realized_pnl = 0.0
        fees_paid = 0.0
        fills: list[Fill] = []
        portfolio_rows: list[PortfolioSnapshot] = []

        for row_idx, row in enumerate(book_df.iter_rows(named=True), start=1):
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

            if progress_callback is not None:
                progress_callback(row_idx, total_rows)

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


def _trailing_quantiles(
    times_ns: np.ndarray,
    values: np.ndarray,
    window_ns: int,
    percentile: float,
) -> np.ndarray:
    thresholds = np.full(len(values), np.nan, dtype=float)
    left = 0

    for right, time_ns in enumerate(times_ns):
        while left < right and times_ns[left] < time_ns - window_ns:
            left += 1
        if left < right:
            thresholds[right] = float(np.quantile(values[left:right], percentile))

    return thresholds


def _portfolio_metrics(portfolio: pl.DataFrame, initial_cash: float) -> dict[str, float]:
    if portfolio.is_empty():
        return {
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "fees_paid": 0.0,
            "gross_return_volatility": float("nan"),
            "net_return_volatility": float("nan"),
            "gross_sharpe": float("nan"),
            "net_sharpe": float("nan"),
            "num_book_updates": 0.0,
        }

    net_equity = portfolio["equity"].to_numpy()
    fees_paid = portfolio["fees_paid"].to_numpy()
    gross_equity = net_equity + fees_paid
    gross_returns = np.diff(gross_equity) / initial_cash
    net_returns = np.diff(net_equity) / initial_cash

    return {
        "gross_pnl": float(gross_equity[-1] - initial_cash),
        "net_pnl": float(net_equity[-1] - initial_cash),
        "fees_paid": float(fees_paid[-1]),
        "gross_return_volatility": _return_volatility(gross_returns),
        "net_return_volatility": _return_volatility(net_returns),
        "gross_sharpe": _sharpe(gross_returns),
        "net_sharpe": _sharpe(net_returns),
        "num_book_updates": float(len(portfolio)),
    }


def _return_volatility(returns: np.ndarray) -> float:
    return float(np.std(returns, ddof=1)) if len(returns) > 1 else float("nan")


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return float("nan")
    return_vol = _return_volatility(returns)
    if return_vol == 0 or np.isnan(return_vol):
        return float("nan")
    return float(np.mean(returns) / return_vol)


def _format_timedelta(value: timedelta) -> str:
    total_seconds = int(value.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"


@dataclass(frozen=True)
class SweepExecution:
    requested_quantity: float
    filled_quantity: float
    avg_price: float | None
    start_time: datetime
    end_time: datetime | None
    is_complete: bool


def _delete_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _append_csv(df: pl.DataFrame, path: Path) -> None:
    if df.is_empty():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    include_header = not path.exists()
    with path.open("ab") as file:
        df.write_csv(file, include_header=include_header)


def _sweep_fill_from_agg_trades(
    signed_quantity: float,
    start_time: datetime,
    trade_times_ns: np.ndarray,
    trade_times: list[datetime],
    prices: np.ndarray,
    quantities: np.ndarray,
    is_buyer_maker: np.ndarray,
    max_end_time: datetime | None = None,
) -> SweepExecution:
    requested_abs = abs(signed_quantity)
    if requested_abs == 0:
        return SweepExecution(signed_quantity, 0.0, None, start_time, None, True)

    start_ns = _datetime_to_ns(start_time)
    max_end_ns = _datetime_to_ns(max_end_time) if max_end_time is not None else None
    start_idx = int(np.searchsorted(trade_times_ns, start_ns, side="left"))
    want_buyer_maker = signed_quantity < 0
    remaining = requested_abs
    filled_abs = 0.0
    notional_px_qty = 0.0
    end_time: datetime | None = None

    for idx in range(start_idx, len(trade_times_ns)):
        if max_end_ns is not None and trade_times_ns[idx] > max_end_ns:
            break
        if bool(is_buyer_maker[idx]) != want_buyer_maker:
            continue

        fill_abs = min(remaining, float(quantities[idx]))
        filled_abs += fill_abs
        remaining -= fill_abs
        notional_px_qty += fill_abs * float(prices[idx])
        end_time = trade_times[idx]

        if remaining <= 0:
            break

    signed_filled = filled_abs if signed_quantity > 0 else -filled_abs
    avg_price = notional_px_qty / filled_abs if filled_abs > 0 else None
    return SweepExecution(
        requested_quantity=signed_quantity,
        filled_quantity=signed_filled,
        avg_price=avg_price,
        start_time=start_time,
        end_time=end_time,
        is_complete=filled_abs >= requested_abs,
    )


def _linear_contract_pnl_usd(
    position: float,
    entry_price: float,
    exit_price: float,
    contract_notional_usd: float,
) -> float:
    return position * contract_notional_usd * (exit_price / entry_price - 1.0)


def _datetime_to_ns(value: datetime | None) -> int:
    if value is None:
        raise ValueError("datetime value cannot be None")
    return int(value.timestamp() * 1_000_000_000)
