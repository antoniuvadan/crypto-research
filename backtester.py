from __future__ import annotations

import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol

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
    micro_price: float | None = None

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def book_imbalance(self) -> float:
        """(bid_qty - ask_qty) / (bid_qty + ask_qty) in [-1, 1]; +ve = bid-heavy."""
        total = self.bid_qty + self.ask_qty
        return (self.bid_qty - self.ask_qty) / total if total > 0 else 0.0


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


def _maker_fill_from_agg_trades(
    signed_quantity: float,
    limit_price: float,
    start_time: datetime,
    trade_times_ns: np.ndarray,
    trade_times: list[datetime],
    prices: np.ndarray,
    quantities: np.ndarray,
    is_buyer_maker: np.ndarray,
    max_end_time: datetime | None = None,
    strict_cross: bool = True,
) -> SweepExecution:
    """Passive (maker) fill: a resting limit waits for the tape to come to it.

    A limit at `limit_price` fills only when an aggressor on the *opposite* side
    trades into it within [start_time, max_end_time]:
      * maker BUY  (signed_quantity > 0): rest a bid; fills on a taker SELL
        (is_buyer_maker == True) printing at price <= limit_price.
      * maker SELL (signed_quantity < 0): rest an ask; fills on a taker BUY
        (is_buyer_maker == False) printing at price >= limit_price.

    This is the mirror image of `_sweep_fill_from_agg_trades`, which crosses the
    spread and matches the *same*-side aggressor (`want_buyer_maker = signed < 0`);
    here `want_buyer_maker = signed > 0`.

    `strict_cross` requires the print to trade *strictly through* the limit -- a
    conservative queue assumption: the level was cleared, so anything resting at it
    (including us) must have filled. With it False, a print exactly at the limit
    also counts (optimistic front-of-queue). A passive order fills at its own
    quoted price, so `avg_price` is `limit_price` regardless of how far the
    aggressor walked past it.

    If nothing reaches the limit in the window, `filled_quantity` is 0 -- the order
    never filled and the trade does not happen (a missed trade, not a free entry).
    """
    requested_abs = abs(signed_quantity)
    if requested_abs == 0:
        return SweepExecution(signed_quantity, 0.0, None, start_time, None, True)

    start_ns = _datetime_to_ns(start_time)
    max_end_ns = _datetime_to_ns(max_end_time) if max_end_time is not None else None
    start_idx = int(np.searchsorted(trade_times_ns, start_ns, side="left"))
    want_buyer_maker = signed_quantity > 0  # a bid fills against taker SELLs
    is_buy = signed_quantity > 0
    remaining = requested_abs
    filled_abs = 0.0
    end_time: datetime | None = None

    for idx in range(start_idx, len(trade_times_ns)):
        if max_end_ns is not None and trade_times_ns[idx] > max_end_ns:
            break
        if bool(is_buyer_maker[idx]) != want_buyer_maker:
            continue
        price = float(prices[idx])
        if is_buy:
            crossed = price < limit_price if strict_cross else price <= limit_price
        else:
            crossed = price > limit_price if strict_cross else price >= limit_price
        if not crossed:
            continue

        fill_abs = min(remaining, float(quantities[idx]))
        filled_abs += fill_abs
        remaining -= fill_abs
        end_time = trade_times[idx]
        if remaining <= 0:
            break

    signed_filled = filled_abs if signed_quantity > 0 else -filled_abs
    avg_price = limit_price if filled_abs > 0 else None
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


# ---------------------------------------------------------------------------
# Book-state access (L1 microstructure) and the exit-policy seam.
#
# These exist so decision logic (e.g. a state-dependent exit) can read the L1
# book/flow at and after a trade, instead of relying on a fixed clock. The Model
# C runner is wired to drive exits through an ExitPolicy; the default
# FixedHorizonExit reproduces the current decision_time + holding_period behavior
# exactly, so nothing changes until a dynamic policy is supplied.
# ---------------------------------------------------------------------------

BOOK_TICKER_DAY_COLS = [
    "event_time_datetime",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "mid_price",
    "micro_price",
]


def _book_ticker_day_path(
    book_ticker_dir: Path, d: date, symbol: str = "BTCUSD_PERP"
) -> Path:
    return book_ticker_dir / f"{symbol}-bookTicker-{d.isoformat()}.parquet"


@dataclass(frozen=True)
class BookView:
    """Indexed, in-memory L1 book over a time range; queryable by timestamp.

    Backed by parallel numpy arrays (plus the original datetimes for cheap
    snapshot construction) so point and window lookups are binary searches, the
    same access pattern as _sweep_fill_from_agg_trades.
    """

    times_ns: np.ndarray
    times: list[datetime]
    bid_price: np.ndarray
    bid_qty: np.ndarray
    ask_price: np.ndarray
    ask_qty: np.ndarray
    mid_price: np.ndarray
    micro_price: np.ndarray

    @classmethod
    def from_frame(cls, df: pl.DataFrame) -> "BookView":
        df = df.sort("event_time_datetime")
        return cls(
            times_ns=df["event_time_datetime"].cast(pl.Int64).to_numpy(),
            times=df["event_time_datetime"].to_list(),
            bid_price=df["best_bid_price"].to_numpy(),
            bid_qty=df["best_bid_qty"].to_numpy(),
            ask_price=df["best_ask_price"].to_numpy(),
            ask_qty=df["best_ask_qty"].to_numpy(),
            mid_price=df["mid_price"].to_numpy(),
            micro_price=df["micro_price"].to_numpy(),
        )

    @classmethod
    def empty(cls) -> "BookView":
        f = np.empty(0, dtype=float)
        return cls(np.empty(0, dtype=np.int64), [], f, f.copy(), f.copy(), f.copy(), f.copy(), f.copy())

    def __len__(self) -> int:
        return len(self.times_ns)

    def _snapshot(self, idx: int) -> MarketSnapshot:
        return MarketSnapshot(
            time=self.times[idx],
            bid_price=float(self.bid_price[idx]),
            bid_qty=float(self.bid_qty[idx]),
            ask_price=float(self.ask_price[idx]),
            ask_qty=float(self.ask_qty[idx]),
            micro_price=float(self.micro_price[idx]),
        )

    def as_of(self, t: datetime) -> MarketSnapshot | None:
        """Last L1 update at or before t (backward as-of). None if t precedes all."""
        if len(self) == 0:
            return None
        idx = int(np.searchsorted(self.times_ns, _datetime_to_ns(t), side="right")) - 1
        return self._snapshot(idx) if idx >= 0 else None

    def slice_window(self, start: datetime, end: datetime) -> "BookView":
        """All updates with start <= time <= end (a cheap array-slice view)."""
        if len(self) == 0:
            return self
        lo = int(np.searchsorted(self.times_ns, _datetime_to_ns(start), side="left"))
        hi = int(np.searchsorted(self.times_ns, _datetime_to_ns(end), side="right"))
        return BookView(
            times_ns=self.times_ns[lo:hi],
            times=self.times[lo:hi],
            bid_price=self.bid_price[lo:hi],
            bid_qty=self.bid_qty[lo:hi],
            ask_price=self.ask_price[lo:hi],
            ask_qty=self.ask_qty[lo:hi],
            mid_price=self.mid_price[lo:hi],
            micro_price=self.micro_price[lo:hi],
        )

    def snapshots(self) -> Iterator[MarketSnapshot]:
        """Iterate the updates in time order (for a forward scan of the book)."""
        for idx in range(len(self)):
            yield self._snapshot(idx)

    @classmethod
    def concat(cls, views: list["BookView"]) -> "BookView":
        views = [v for v in views if len(v) > 0]
        if not views:
            return cls.empty()
        if len(views) == 1:
            return views[0]
        return cls(
            times_ns=np.concatenate([v.times_ns for v in views]),
            times=[t for v in views for t in v.times],
            bid_price=np.concatenate([v.bid_price for v in views]),
            bid_qty=np.concatenate([v.bid_qty for v in views]),
            ask_price=np.concatenate([v.ask_price for v in views]),
            ask_qty=np.concatenate([v.ask_qty for v in views]),
            mid_price=np.concatenate([v.mid_price for v in views]),
            micro_price=np.concatenate([v.micro_price for v in views]),
        )


class BookProvider:
    """Lazy, day-batched access to L1 bookTicker, cached by calendar day.

    bookTicker is one parquet per UTC day (~4.6M rows each), far too large to
    hold the full study period in memory, so days are loaded on first query and
    an LRU of the most recently used days is kept. Construction does no I/O — a
    runner can always build one and pay nothing if the active ExitPolicy never
    queries the book.
    """

    def __init__(
        self,
        book_ticker_dir: Path = DEFAULT_BOOK_TICKER_PATH,
        cache_days: int = 3,
        symbol: str = "BTCUSD_PERP",
    ) -> None:
        self.book_ticker_dir = book_ticker_dir
        self.cache_days = max(cache_days, 1)
        self.symbol = symbol
        self._cache: "OrderedDict[date, BookView]" = OrderedDict()

    def _day_view(self, d: date) -> BookView:
        cached = self._cache.get(d)
        if cached is not None:
            self._cache.move_to_end(d)
            return cached
        path = _book_ticker_day_path(self.book_ticker_dir, d, self.symbol)
        view = (
            BookView.from_frame(pl.read_parquet(path, columns=BOOK_TICKER_DAY_COLS))
            if path.exists()
            else BookView.empty()
        )
        self._cache[d] = view
        while len(self._cache) > self.cache_days:
            self._cache.popitem(last=False)
        return view

    def as_of(self, t: datetime) -> MarketSnapshot | None:
        return self._day_view(t.date()).as_of(t)

    def window(self, start: datetime, end: datetime) -> BookView:
        """L1 updates with start <= time <= end, spanning day files as needed."""
        if end < start:
            return BookView.empty()
        days: list[date] = []
        d = start.date()
        last = end.date()
        while d <= last:
            days.append(d)
            d += timedelta(days=1)
        return BookView.concat([self._day_view(day).slice_window(start, end) for day in days])


@dataclass(frozen=True)
class ExitContext:
    """State handed to an ExitPolicy after entry, to decide when to close.

    decision_time / direction / latency describe the trade; entry is the realized
    entry fill (avg_price, end_time, filled_quantity, ...). liquidation_time is
    the underlying event time (decision_time minus the signal's seconds_after),
    which dynamic policies use to anchor a pre-cascade reference. holding_period
    is the cell's configured horizon — fixed policies use it directly, dynamic
    policies may treat it as a default or ignore it in favor of their max_hold cap.
    """

    decision_time: datetime
    liquidation_time: datetime
    direction: int  # traded direction: +1 long, -1 short
    holding_period: timedelta
    latency: timedelta
    entry: SweepExecution


class ExitPolicy(Protocol):
    """Decides when an open position is closed.

    max_hold bounds how long a position may stay open; the runner uses it to cap
    the entry sweep and (for dynamic policies) to bound the forward book scan.
    exit_trigger_time returns the time the close decision is made — the runner
    adds latency before the exit sweep begins, mirroring entry. Dynamic policies
    read forward microstructure via the BookProvider; fixed ones ignore it.
    """

    @property
    def max_hold(self) -> timedelta: ...

    def exit_trigger_time(self, ctx: ExitContext, book: BookProvider | None) -> datetime: ...


@dataclass(frozen=True)
class FixedHorizonExit:
    """Close at decision_time + holding_period (the current static behavior)."""

    holding_period: timedelta

    @property
    def max_hold(self) -> timedelta:
        return self.holding_period

    def exit_trigger_time(self, ctx: ExitContext, book: BookProvider | None = None) -> datetime:
        return ctx.decision_time + self.holding_period


@dataclass(frozen=True)
class RetracementExit:
    """State-dependent reversion exit driven by the L1 mid path.

    A liquidation cascade dislocates the mid from `mid_pre` (sampled pre_lookback
    before the liquidation) to `mid_decision` (at decision_time). A reversion
    trade is opened expecting the mid to retrace from mid_decision back toward
    mid_pre. This policy closes the position at the first L1 update that shows:

      * take-profit -- the mid retraces `take_profit_frac` of the displacement
        (mid_decision -> mid_pre) in the favorable direction;
      * stop -- the mid extends `stop_frac` of the displacement *beyond*
        mid_decision (the cascade keeps running against the trade);
      * time cap -- neither fires within `max_hold`. The time cap is also the
        fallback when a reference mid is missing or the observed dislocation is
        not in the expected reversion direction (so no sensible levels exist).

    The returned trigger is the *observation* time; the runner adds latency before
    the exit sweep, mirroring entry. Reads only the L1 book via the BookProvider
    seam -- no look-ahead beyond the book updates up to each candidate exit.
    """

    max_hold: timedelta = timedelta(minutes=2)
    take_profit_frac: float = 0.5
    stop_frac: float = 1.0
    pre_lookback: timedelta = timedelta(seconds=5)

    def exit_trigger_time(self, ctx: ExitContext, book: BookProvider | None) -> datetime:
        time_cap = ctx.decision_time + self.max_hold
        if book is None:
            return time_cap

        pre = book.as_of(ctx.liquidation_time - self.pre_lookback)
        at_decision = book.as_of(ctx.decision_time)
        if pre is None or at_decision is None:
            return time_cap

        mid_pre = pre.mid_price
        mid_decision = at_decision.mid_price
        favorable = mid_pre - mid_decision  # reversion points back toward mid_pre
        direction = ctx.direction  # +1 long, -1 short

        # The dislocation must sit in the expected reversion direction (a long
        # reversion needs mid_pre above mid_decision, and vice versa) for the
        # retracement/stop levels to be meaningful; otherwise hold to the cap.
        if direction * favorable <= 0:
            return time_cap

        mid_target = mid_decision + self.take_profit_frac * favorable
        mid_stop = mid_decision - self.stop_frac * favorable

        scan_start = ctx.entry.end_time or (ctx.decision_time + ctx.latency)
        window = book.window(scan_start, time_cap)
        for mid_t, t in zip(window.mid_price, window.times):
            if direction * (mid_t - mid_target) >= 0:  # take-profit reached
                return t
            if direction * (mid_t - mid_stop) <= 0:  # stop hit
                return t
        return time_cap
