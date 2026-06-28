"""
Liquidation-momentum strategy and its backtest runners.

The reusable backtesting engine, data loaders, fill simulation, and metrics
live in backtester.py; this module holds the liquidation signal definition,
the momentum strategy, and the runners (naive book-ticker + Model C).

Run the Model C sensitivity grid over the training window:
    python backtest_momentum.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from backtester import (
    CONTRACT_NOTIONAL_USD,
    DEFAULT_AGG_TRADES_PATH,
    DEFAULT_BOOK_TICKER_PATH,
    DEFAULT_LATENCY,
    DEFAULT_LIQ_SNAP_PATH,
    DEFAULT_TRADE_NOTIONAL_USD,
    DEFAULT_TRADE_NOTIONAL_USD_GRID,
    TAKER_FEE_RATE,
    Backtester,
    BookProvider,
    ExitContext,
    ExitPolicy,
    FixedHorizonExit,
    MarketSnapshot,
    Order,
    PortfolioSnapshot,
    ProgressPrinter,
    _append_csv,
    _delete_if_exists,
    _format_timedelta,
    _linear_contract_pnl_usd,
    _portfolio_metrics,
    _progress_message,
    _return_volatility,
    _sharpe,
    _sweep_fill_from_agg_trades,
    _trailing_quantiles,
    load_agg_trades,
    load_liq_snap,
)


DEFAULT_MODEL_C_SUMMARY_PATH = Path("data/results/liquidation_momentum_model_c_summary.csv")
DEFAULT_MODEL_C_TRADES_PATH = Path("data/results/liquidation_momentum_model_c_trades.csv")


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
        self.size_fraction = size_fraction
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
                    f"{aggregate_quantity_col!r} is missing, agg_trades is required."
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


def run_liquidation_momentum_model_c_backtests(
    liq_snap: pl.LazyFrame | pl.DataFrame,
    agg_trades: pl.LazyFrame | pl.DataFrame,
    holding_periods: tuple[timedelta, ...] = (
        timedelta(seconds=5),
        timedelta(seconds=10),
        timedelta(seconds=30),
        timedelta(minutes=1),
        timedelta(minutes=2),
    ),
    initial_cash: float = 1_000_000.0,
    trade_notional_usd: float = DEFAULT_TRADE_NOTIONAL_USD,
    latency: timedelta = DEFAULT_LATENCY,
    fee_rate: float = TAKER_FEE_RATE,
    contract_notional_usd: float = CONTRACT_NOTIONAL_USD,
    summary_csv_path: Path | None = DEFAULT_MODEL_C_SUMMARY_PATH,
    trades_csv_path: Path | None = DEFAULT_MODEL_C_TRADES_PATH,
    signal_direction_sign: int = 1,
    exit_policy: ExitPolicy | None = None,
    book_ticker_dir: Path | None = DEFAULT_BOOK_TICKER_PATH,
    show_progress: bool = False,
) -> dict[str, pl.DataFrame]:
    """
    Model C execution: latency, then sweep through same-side aggTrades for entry and exit.

    signal_direction_sign controls trade direction relative to the liquidating flow:
    +1 = momentum (trade with the flow, the original signal), -1 = reversion
    (trade against the flow). The recorded `side`/`direction` reflect the executed
    trade, so fills, fees, and P&L are all simulated on the correct side.

    Trade size is explicit notional. Contracts = trade_notional_usd / contract_notional_usd.

    Exits are routed through an ExitPolicy. When exit_policy is None each holding
    period uses FixedHorizonExit(holding_period), reproducing the static
    decision_time + holding_period + latency exit exactly. A dynamic policy reads
    forward L1 microstructure via a BookProvider over book_ticker_dir; that
    provider is lazy, so the default fixed path loads no bookTicker data.
    """

    label = "model-c"
    _progress_message(label if show_progress else None, "collecting aggregate trades")
    trades_df = agg_trades.collect() if isinstance(agg_trades, pl.LazyFrame) else agg_trades
    trades_df = trades_df.sort("transact_time_datetime")
    _progress_message(label if show_progress else None, f"loaded {len(trades_df):,} trades")

    events = LiquidationMomentumStrategy._build_signal_events(
        liq_snap=liq_snap,
        agg_trades=trades_df,
        percentile=0.98,
        trailing_window=timedelta(days=7),
        seconds_before=5,
        seconds_after=5,
        aggregate_quantity_col="agg_qty_5s_before_5s_after",
        progress_label=label if show_progress else None,
    )

    trade_times_ns = trades_df["transact_time_datetime"].cast(pl.Int64).to_numpy()
    trade_times = trades_df["transact_time_datetime"].to_list()
    prices = trades_df["price"].to_numpy()
    quantities = trades_df["quantity"].to_numpy()
    is_buyer_maker = trades_df["is_buyer_maker"].to_numpy()

    target_contracts = trade_notional_usd / contract_notional_usd
    summary_rows: list[dict[str, float | str]] = []
    trade_rows: list[dict[str, float | str | bool | None]] = []
    total_periods = len(holding_periods)

    # Lazy: loads no bookTicker until a dynamic ExitPolicy actually queries it.
    book_provider = BookProvider(book_ticker_dir) if book_ticker_dir is not None else None

    for period_idx, holding_period in enumerate(holding_periods, start=1):
        # A caller-supplied policy is used as-is across horizons; otherwise the
        # static per-horizon exit reproduces the original fixed-clock behavior.
        policy = exit_policy if exit_policy is not None else FixedHorizonExit(holding_period)
        holding_label = _format_timedelta(holding_period)
        progress = (
            ProgressPrinter(
                label=f"{period_idx}/{total_periods} {holding_label} model-c events",
                total=len(events),
                update_every=500,
            )
            if show_progress
            else None
        )
        _progress_message(
            label if show_progress else None,
            f"running holding period {period_idx}/{total_periods}: {holding_label}",
        )

        pnl_values: list[float] = []
        gross_pnl_values: list[float] = []
        gross_return_values: list[float] = []
        net_return_values: list[float] = []
        fees_paid = 0.0
        completed_round_trips = 0

        for event_idx, event in enumerate(events, start=1):
            traded_direction = signal_direction_sign * event.direction
            signed_entry_qty = target_contracts * traded_direction
            entry_start = event.decision_time + latency
            # Cap the entry sweep at the latest possible exit (decision + max_hold
            # + latency). For FixedHorizonExit this equals the static exit_start.
            entry_cap = event.decision_time + policy.max_hold + latency
            entry = _sweep_fill_from_agg_trades(
                signed_quantity=signed_entry_qty,
                start_time=entry_start,
                trade_times_ns=trade_times_ns,
                trade_times=trade_times,
                prices=prices,
                quantities=quantities,
                is_buyer_maker=is_buyer_maker,
                max_end_time=entry_cap,
            )

            if entry.filled_quantity == 0 or entry.avg_price is None:
                if progress is not None:
                    progress.update(event_idx, force=event_idx == len(events))
                continue

            # Policy chooses the close-decision time; latency then delays the exit
            # sweep, mirroring entry. Fixed policy => decision + holding + latency.
            exit_context = ExitContext(
                decision_time=event.decision_time,
                liquidation_time=event.liquidation_time,
                direction=traded_direction,
                holding_period=holding_period,
                latency=latency,
                entry=entry,
            )
            exit_start = policy.exit_trigger_time(exit_context, book_provider) + latency

            exit = _sweep_fill_from_agg_trades(
                signed_quantity=-entry.filled_quantity,
                start_time=exit_start,
                trade_times_ns=trade_times_ns,
                trade_times=trade_times,
                prices=prices,
                quantities=quantities,
                is_buyer_maker=is_buyer_maker,
            )

            if exit.filled_quantity == 0 or exit.avg_price is None:
                if progress is not None:
                    progress.update(event_idx, force=event_idx == len(events))
                continue

            closed_quantity = min(abs(entry.filled_quantity), abs(exit.filled_quantity))
            signed_closed_quantity = closed_quantity if entry.filled_quantity > 0 else -closed_quantity
            gross_pnl = _linear_contract_pnl_usd(
                position=signed_closed_quantity,
                entry_price=entry.avg_price,
                exit_price=exit.avg_price,
                contract_notional_usd=contract_notional_usd,
            )
            entry_fee = abs(entry.filled_quantity) * contract_notional_usd * fee_rate
            exit_fee = abs(exit.filled_quantity) * contract_notional_usd * fee_rate
            trade_fees = entry_fee + exit_fee
            net_pnl = gross_pnl - trade_fees

            completed_round_trips += 1
            fees_paid += trade_fees
            gross_pnl_values.append(gross_pnl)
            pnl_values.append(net_pnl)
            gross_return_values.append(gross_pnl / initial_cash)
            net_return_values.append(net_pnl / initial_cash)

            trade_rows.append(
                {
                    "holding_period": holding_label,
                    "liquidation_time": event.liquidation_time.isoformat(),
                    "decision_time": event.decision_time.isoformat(),
                    "side": "BUY" if traded_direction > 0 else "SELL",
                    "direction": traded_direction,
                    "trade_notional_usd": trade_notional_usd,
                    "target_contracts": target_contracts,
                    "entry_start_time": entry.start_time.isoformat(),
                    "entry_end_time": entry.end_time.isoformat() if entry.end_time else None,
                    "entry_quantity": entry.filled_quantity,
                    "entry_vwap": entry.avg_price,
                    "entry_complete": entry.is_complete,
                    "exit_start_time": exit.start_time.isoformat(),
                    "exit_end_time": exit.end_time.isoformat() if exit.end_time else None,
                    "exit_quantity": exit.filled_quantity,
                    "exit_vwap": exit.avg_price,
                    "exit_complete": exit.is_complete,
                    "gross_pnl": gross_pnl,
                    "fees_paid": trade_fees,
                    "net_pnl": net_pnl,
                    "aggregate_quantity": event.aggregate_quantity,
                    "trailing_threshold": event.trailing_threshold,
                }
            )

            if progress is not None:
                progress.update(event_idx, force=event_idx == len(events))

        gross_returns = np.array(gross_return_values)
        net_returns = np.array(net_return_values)

        summary_rows.append(
            {
                "holding_period": holding_label,
                "execution_model": "aggTrades_sweep_latency",
                "latency_ms": latency.total_seconds() * 1_000,
                "trade_notional_usd": trade_notional_usd,
                "target_contracts": target_contracts,
                "num_signal_events": float(len(events)),
                "num_completed_round_trips": float(completed_round_trips),
                "gross_pnl": float(sum(gross_pnl_values)),
                "net_pnl": float(sum(pnl_values)),
                "fees_paid": fees_paid,
                "gross_return_volatility": _return_volatility(gross_returns),
                "net_return_volatility": _return_volatility(net_returns),
                "gross_sharpe": _sharpe(gross_returns),
                "net_sharpe": _sharpe(net_returns),
            }
        )

    summary = pl.DataFrame(summary_rows)
    trades = pl.DataFrame(trade_rows)

    if summary_csv_path is not None:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary.write_csv(summary_csv_path)
    if trades_csv_path is not None:
        trades_csv_path.parent.mkdir(parents=True, exist_ok=True)
        trades.write_csv(trades_csv_path)

    return {
        "summary": summary,
        "trades": trades,
    }


def run_liquidation_momentum_model_c_sensitivity(
    liq_snap_path: Path = DEFAULT_LIQ_SNAP_PATH,
    agg_trades_path: Path = DEFAULT_AGG_TRADES_PATH,
    start_date: date = date(2023, 6, 25),
    end_date: date = date(2024, 6, 24),
    holding_periods: tuple[timedelta, ...] = (
        timedelta(seconds=5),
        timedelta(seconds=10),
        timedelta(seconds=30),
        timedelta(minutes=1),
        timedelta(minutes=2),
    ),
    trade_notional_usd_grid: tuple[float, ...] = DEFAULT_TRADE_NOTIONAL_USD_GRID,
    initial_cash: float = 1_000_000.0,
    latency: timedelta = DEFAULT_LATENCY,
    fee_rate: float = TAKER_FEE_RATE,
    contract_notional_usd: float = CONTRACT_NOTIONAL_USD,
    summary_csv_path: Path = DEFAULT_MODEL_C_SUMMARY_PATH,
    trades_csv_path: Path = DEFAULT_MODEL_C_TRADES_PATH,
    show_progress: bool = True,
) -> dict[str, pl.DataFrame]:
    tasks = [
        {
            "liq_snap_path": liq_snap_path,
            "agg_trades_path": agg_trades_path,
            "start_date": start_date,
            "end_date": end_date,
            "holding_period": holding_period,
            "trade_notional_usd": trade_notional_usd,
            "initial_cash": initial_cash,
            "latency": latency,
            "fee_rate": fee_rate,
            "contract_notional_usd": contract_notional_usd,
        }
        for trade_notional_usd in trade_notional_usd_grid
        for holding_period in holding_periods
    ]

    if show_progress:
        print(
            f"Running {len(tasks)} Model C backtests sequentially across "
            f"{len(holding_periods)} holding periods and {len(trade_notional_usd_grid)} trade sizes",
            file=sys.stderr,
            flush=True,
        )

    summary_frames: list[pl.DataFrame] = []
    trade_frames: list[pl.DataFrame] = []
    _delete_if_exists(summary_csv_path)
    _delete_if_exists(trades_csv_path)
    progress = (
        ProgressPrinter("model-c grid", total=len(tasks), update_every=1)
        if show_progress
        else None
    )

    for completed, task in enumerate(tasks, start=1):
        cell_label = (
            f"{_format_timedelta(task['holding_period'])}, "
            f"${task['trade_notional_usd']:,.0f}"
        )
        if show_progress:
            print(
                f"Starting grid cell {completed}/{len(tasks)}: "
                f"{cell_label}",
                file=sys.stderr,
                flush=True,
            )

        result = _run_model_c_grid_cell(task, progress_label=cell_label if show_progress else None)
        summary_frames.append(result["summary"])
        trade_frames.append(result["trades"])
        _append_csv(result["summary"], summary_csv_path)
        _append_csv(result["trades"], trades_csv_path)
        if progress is not None:
            progress.update(completed, force=completed == len(tasks))

    summary = pl.concat(summary_frames, how="vertical") if summary_frames else pl.DataFrame()
    trades = pl.concat(trade_frames, how="vertical") if trade_frames else pl.DataFrame()

    if not summary.is_empty():
        summary = summary.sort(["trade_notional_usd", "holding_period"])
    if not trades.is_empty():
        trades = trades.sort(["trade_notional_usd", "holding_period", "decision_time"])

    return {
        "summary": summary,
        "trades": trades,
    }


def _run_model_c_grid_cell(
    task: dict,
    progress_label: str | None = None,
) -> dict[str, pl.DataFrame]:
    holding_period = task["holding_period"]
    trade_notional_usd = task["trade_notional_usd"]

    _progress_message(progress_label, "before reading liquidation snapshots")
    liq_snap = load_liq_snap(
        liq_snap_path=task["liq_snap_path"],
        start_date=task["start_date"],
        end_date=task["end_date"],
    ).collect()
    _progress_message(progress_label, f"after reading {len(liq_snap):,} liquidation snapshots")

    _progress_message(progress_label, "before reading aggregate trades")
    agg_trades = load_agg_trades(
        agg_trades_path=task["agg_trades_path"],
        start_date=task["start_date"],
        end_date=task["end_date"],
    ).collect()
    _progress_message(progress_label, f"after reading {len(agg_trades):,} aggregate trades")
    _progress_message(progress_label, "starting actual Model C backtest")

    return run_liquidation_momentum_model_c_backtests(
        liq_snap=liq_snap,
        agg_trades=agg_trades,
        holding_periods=(holding_period,),
        initial_cash=task["initial_cash"],
        trade_notional_usd=trade_notional_usd,
        latency=task["latency"],
        fee_rate=task["fee_rate"],
        contract_notional_usd=task["contract_notional_usd"],
        summary_csv_path=None,
        trades_csv_path=None,
        show_progress=progress_label is not None,
    )


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
    contract_notional_usd = CONTRACT_NOTIONAL_USD
    latency = DEFAULT_LATENCY
    trade_notional_usd_grid = DEFAULT_TRADE_NOTIONAL_USD_GRID

    # Positions are tracked in contracts. For BTCUSD_PERP, contracts = dollar notional / 100.
    # Model C execution skips a latency window, then sweeps through actual aggTrades.
    # BTCUSD_PERP is modeled with non-VIP COIN-M taker fees: 5 bps per market-order leg.
    results = run_liquidation_momentum_model_c_sensitivity(
        liq_snap_path=DEFAULT_LIQ_SNAP_PATH,
        agg_trades_path=DEFAULT_AGG_TRADES_PATH,
        start_date=train_start_date,
        end_date=train_end_date,
        holding_periods=holding_periods,
        trade_notional_usd_grid=trade_notional_usd_grid,
        initial_cash=initial_cash,
        latency=latency,
        fee_rate=fee_rate,
        contract_notional_usd=contract_notional_usd,
        summary_csv_path=DEFAULT_MODEL_C_SUMMARY_PATH,
        trades_csv_path=DEFAULT_MODEL_C_TRADES_PATH,
    )
    print(results["summary"])
    print(f"Wrote summary CSV to {DEFAULT_MODEL_C_SUMMARY_PATH}")
    print(f"Wrote trades CSV to {DEFAULT_MODEL_C_TRADES_PATH}")
