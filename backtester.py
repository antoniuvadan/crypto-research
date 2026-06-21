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


@dataclass(frozen=True)
class MomentumSignalEvent:
    liquidation_time: datetime
    decision_time: datetime
    side: str
    direction: int
    aggregate_quantity: float
    trailing_threshold: float


@dataclass
class _ScheduledClose:
    close_time: datetime
    quantity: float


class LiquidationMomentumStrategy:
    """
    Trade in the same direction as unusually large liquidation-driven flow.

    A liquidation event is evaluated only at time_datetime + seconds_after, so
    the +/- window used to compute aggregate trade quantity is fully observed.
    """

    def __init__(
        self,
        liq_snap: pl.LazyFrame | pl.DataFrame,
        agg_trades: pl.LazyFrame | pl.DataFrame | None = None,
        holding_period: timedelta = timedelta(seconds=5),
        percentile: float = 0.98,
        trailing_window: timedelta = timedelta(days=7),
        seconds_before: int = 5,
        seconds_after: int = 5,
        size_fraction: float = 0.5,
        aggregate_quantity_col: str = "agg_qty_5s_before_5s_after",
        progress_label: str | None = None,
    ) -> None:
        self.holding_period = holding_period
        self.size_fraction = size_fraction # TODO: stop using size fraction and use a more realistic entry price model
        self.events = self._build_signal_events(
            liq_snap=liq_snap,
            agg_trades=agg_trades,
            percentile=percentile,
            trailing_window=trailing_window,
            seconds_before=seconds_before,
            seconds_after=seconds_after,
            aggregate_quantity_col=aggregate_quantity_col,
            progress_label=progress_label,
        )
        self._next_event_idx = 0
        self._scheduled_closes: list[_ScheduledClose] = []

    def on_book(self, book: MarketSnapshot, portfolio: PortfolioSnapshot) -> list[Order]:
        orders: list[Order] = []

        remaining_closes: list[_ScheduledClose] = []
        for scheduled_close in self._scheduled_closes:
            if scheduled_close.close_time <= book.time:
                orders.append(Order(-scheduled_close.quantity))
            else:
                remaining_closes.append(scheduled_close)
        self._scheduled_closes = remaining_closes

        while self._next_event_idx < len(self.events):
            event = self.events[self._next_event_idx]
            if event.decision_time > book.time:
                break

            self._next_event_idx += 1
            if event.direction > 0:
                quantity = book.ask_qty * self.size_fraction
            else:
                quantity = -book.bid_qty * self.size_fraction

            if quantity == 0:
                continue

            orders.append(Order(quantity))
            self._scheduled_closes.append(
                _ScheduledClose(
                    close_time=event.decision_time + self.holding_period,
                    quantity=quantity,
                )
            )

        self._scheduled_closes.sort(key=lambda close: close.close_time)
        return orders

    @staticmethod
    def _build_signal_events(
        liq_snap: pl.LazyFrame | pl.DataFrame,
        agg_trades: pl.LazyFrame | pl.DataFrame | None,
        percentile: float,
        trailing_window: timedelta,
        seconds_before: int,
        seconds_after: int,
        aggregate_quantity_col: str,
        progress_label: str | None = None,
    ) -> list[MomentumSignalEvent]:
        _progress_message(progress_label, "collecting liquidation snapshots")
        liq_df = liq_snap.collect() if isinstance(liq_snap, pl.LazyFrame) else liq_snap
        liq_df = liq_df.sort("time_datetime")
        _progress_message(progress_label, f"loaded {len(liq_df):,} liquidation snapshots")

        if aggregate_quantity_col in liq_df.columns:
            _progress_message(progress_label, f"using existing {aggregate_quantity_col}")
            agg_qty = liq_df[aggregate_quantity_col].to_numpy()
        else:
            if agg_trades is None:
                raise ValueError(
                    f"{aggregate_quantity_col!r} is missing, so agg_trades is required."
                )
            _progress_message(progress_label, "collecting aggregate trades")
            trades_df = agg_trades.collect() if isinstance(agg_trades, pl.LazyFrame) else agg_trades
            _progress_message(progress_label, f"loaded {len(trades_df):,} aggregate trades")
            _progress_message(progress_label, "computing same-direction +/-5s aggregate quantities")
            agg_qty = _same_direction_aggregate_quantities(
                liq_df=liq_df,
                trades_df=trades_df,
                seconds_before=seconds_before,
                seconds_after=seconds_after,
            )

        _progress_message(progress_label, "computing trailing 7d percentile thresholds")
        liq_times = liq_df["time_datetime"].to_list()
        liq_times_ns = liq_df["time_datetime"].cast(pl.Int64).to_numpy()
        sides = liq_df["side"].to_list()
        thresholds = _trailing_quantiles(
            times_ns=liq_times_ns,
            values=agg_qty,
            window_ns=int(trailing_window.total_seconds() * 1e9),
            percentile=percentile,
        )

        _progress_message(progress_label, "building signal events")
        events: list[MomentumSignalEvent] = []
        for liq_time, side, qty, threshold in zip(liq_times, sides, agg_qty, thresholds):
            if np.isnan(threshold) or qty <= threshold:
                continue

            direction = 1 if side == "BUY" else -1
            events.append(
                MomentumSignalEvent(
                    liquidation_time=liq_time,
                    decision_time=liq_time + timedelta(seconds=seconds_after),
                    side=side,
                    direction=direction,
                    aggregate_quantity=float(qty),
                    trailing_threshold=float(threshold),
                )
            )

        _progress_message(progress_label, f"built {len(events):,} signal events")
        return events


def _same_direction_aggregate_quantities(
    liq_df: pl.DataFrame,
    trades_df: pl.DataFrame,
    seconds_before: int = 5,
    seconds_after: int = 5,
) -> np.ndarray:
    trades_df = trades_df.sort("transact_time_datetime")
    trade_times_ns = trades_df["transact_time_datetime"].cast(pl.Int64).to_numpy()
    quantities = trades_df["quantity"].to_numpy()
    is_buyer_maker = trades_df["is_buyer_maker"].to_numpy()

    # SELL liquidation means taker sells, so buyer is maker. BUY is the opposite.
    cum_qty_buyer_maker = np.concatenate(
        [[0.0], np.cumsum(np.where(is_buyer_maker, quantities, 0.0))]
    )
    cum_qty_seller_maker = np.concatenate(
        [[0.0], np.cumsum(np.where(~is_buyer_maker, quantities, 0.0))]
    )

    liq_times_ns = liq_df["time_datetime"].cast(pl.Int64).to_numpy()
    sides = liq_df["side"].to_list()
    lefts = np.searchsorted(
        trade_times_ns,
        liq_times_ns - int(seconds_before * 1e9),
        side="left",
    )
    rights = np.searchsorted(
        trade_times_ns,
        liq_times_ns + int(seconds_after * 1e9),
        side="right",
    )

    agg_qty = np.empty(len(liq_df), dtype=float)
    for i, side in enumerate(sides):
        cum_qty = cum_qty_buyer_maker if side == "SELL" else cum_qty_seller_maker
        agg_qty[i] = cum_qty[rights[i]] - cum_qty[lefts[i]]

    return agg_qty


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


def run_liquidation_momentum_backtests(
    book: pl.LazyFrame | pl.DataFrame,
    liq_snap: pl.LazyFrame | pl.DataFrame,
    agg_trades: pl.LazyFrame | pl.DataFrame | None = None,
    holding_periods: tuple[timedelta, ...] = (
        timedelta(seconds=5),
        timedelta(seconds=10),
        timedelta(seconds=30),
        timedelta(minutes=1),
        timedelta(minutes=2),
    ),
    initial_cash: float = 1_000_000.0,
    fee_rate: float = TAKER_FEE_RATE,
    contract_notional_usd: float = CONTRACT_NOTIONAL_USD,
    show_progress: bool = False,
    progress_update_every: int = 10_000,
) -> dict[str, pl.DataFrame | dict[str, dict[str, pl.DataFrame]]]:
    """
    Run one independent momentum strategy per holding period.

    Entries and exits are modeled as taker orders by default. Fees are charged
    on notional: abs(contracts) * contract_notional_usd * fee_rate.

    Returns a summary DataFrame plus per-horizon portfolio/fill/event details.
    """

    summary_rows = []
    details: dict[str, dict[str, pl.DataFrame]] = {}
    total_periods = len(holding_periods)
    if show_progress:
        print("Collecting book ticker once before horizon loop", file=sys.stderr, flush=True)
    book_df = book.collect() if isinstance(book, pl.LazyFrame) else book
    if show_progress:
        print(f"Collected {len(book_df):,} book ticker rows", file=sys.stderr, flush=True)

    for period_idx, holding_period in enumerate(holding_periods, start=1):
        label = _format_timedelta(holding_period)
        if show_progress:
            print(
                f"Starting holding period {period_idx}/{total_periods}: {label}",
                file=sys.stderr,
                flush=True,
            )

        strategy = LiquidationMomentumStrategy(
            liq_snap=liq_snap,
            agg_trades=agg_trades,
            holding_period=holding_period,
            progress_label=f"{period_idx}/{total_periods} {label}" if show_progress else None,
        )
        progress_printer = (
            ProgressPrinter(
                label=f"{period_idx}/{total_periods} {label}",
                total=0,
                update_every=progress_update_every,
            )
            if show_progress
            else None
        )

        def progress_callback(completed: int, total: int) -> None:
            if progress_printer is None:
                return
            if progress_printer.total == 0:
                progress_printer.total = total
            progress_printer.update(completed, force=completed == 0 or completed == total)

        if show_progress:
            print(
                f"{period_idx}/{total_periods} {label}: running row-by-row backtest",
                file=sys.stderr,
                flush=True,
            )

        results = Backtester(
            book=book_df,
            strategy=strategy,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            contract_notional_usd=contract_notional_usd,
        ).run(progress_callback=progress_callback if show_progress else None)

        portfolio = results["portfolio"]
        metrics = _portfolio_metrics(portfolio, initial_cash)
        summary_rows.append({"holding_period": label, **metrics})
        details[label] = {
            "portfolio": portfolio,
            "fills": results["fills"],
            "events": pl.DataFrame(strategy.events),
        }

        if show_progress:
            print(
                f"Completed holding period {period_idx}/{total_periods}: {label}",
                file=sys.stderr,
                flush=True,
            )

    return {
        "summary": pl.DataFrame(summary_rows),
        "details": details,
    }


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


if __name__ == "__main__":
    train_start_date = date(2023, 6, 25)
    train_end_date = date(2024, 6, 24)

    holding_periods = (
        timedelta(seconds=5),
        timedelta(seconds=10),
        timedelta(seconds=30),
        timedelta(minutes=1),
        timedelta(minutes=2),
    )
    initial_cash = 1_000_000.0
    fee_rate = TAKER_FEE_RATE
    contract_notional_usd = 100.0

    # Positions are tracked in contracts. A $1,000,000 notional position is 10,000 contracts.
    # BTCUSD_PERP is modeled with non-VIP COIN-M taker fees: 5 bps per market-order leg.
    df_book_ticker = load_book_ticker(
        start_date=train_start_date,
        end_date=train_end_date,
    )
    df_liq_snap = load_liq_snap(
        start_date=train_start_date,
        end_date=train_end_date,
    )
    df_agg_trades = load_agg_trades(
        start_date=train_start_date,
        end_date=train_end_date,
    )

    results = run_liquidation_momentum_backtests(
        book=df_book_ticker,
        liq_snap=df_liq_snap,
        agg_trades=df_agg_trades,
        holding_periods=holding_periods,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        contract_notional_usd=contract_notional_usd,
        show_progress=True,
    )
    print(results["summary"])