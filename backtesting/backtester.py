from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from core.mt5_client import MT5Client
from core.strategy import Direction, Signal, Strategy


@dataclass
class SimulatedTrade:
    """A simulated trade from backtesting."""
    strategy: str
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str  # "tp", "sl", "trailing", "end"


class Backtester:
    """Backtesting engine for strategy evaluation."""

    def __init__(self, mt5_client: MT5Client | None = None, point_value: float = 10.0) -> None:
        self._client = mt5_client
        self._point_value = point_value

    def load_data(
        self, symbol: str, timeframe: str, count: int,
    ) -> pd.DataFrame:
        """Load historical data via MT5 client."""
        if self._client is None:
            raise RuntimeError("MT5 client not set")
        return self._client.get_rates(symbol, timeframe, count)

    def run(self, strategy: Strategy, data: pd.DataFrame) -> list[SimulatedTrade]:
        """Run backtest simulation.

        Args:
            strategy: Strategy to test.
            data: OHLCV DataFrame.

        Returns:
            List of simulated trades.
        """
        trades: list[SimulatedTrade] = []
        open_trade: dict[str, Any] | None = None
        min_bars = 30  # Minimum bars before generating signals

        logger.info(f"Backtesting {strategy.name} on {len(data)} bars")

        for i in range(min_bars, len(data)):
            window = data.iloc[:i + 1].copy()
            current = data.iloc[i]
            high = current["high"]
            low = current["low"]
            current_time = current.get("datetime", datetime.now())

            # Check open trade for SL/TP
            if open_trade is not None:
                exit_price = None
                exit_reason = ""

                if open_trade["direction"] == "BUY":
                    if low <= open_trade["sl"]:
                        exit_price = open_trade["sl"]
                        exit_reason = "sl"
                    elif high >= open_trade["tp"]:
                        exit_price = open_trade["tp"]
                        exit_reason = "tp"
                    # Trailing stop
                    elif open_trade.get("trailing_trigger") and high >= open_trade["entry_price"] + open_trade["trailing_trigger"]:
                        new_sl = high - open_trade["trailing_distance"]
                        if new_sl > open_trade["sl"]:
                            open_trade["sl"] = new_sl
                elif open_trade["direction"] == "SELL":
                    if high >= open_trade["sl"]:
                        exit_price = open_trade["sl"]
                        exit_reason = "sl"
                    elif low <= open_trade["tp"]:
                        exit_price = open_trade["tp"]
                        exit_reason = "tp"
                    elif open_trade.get("trailing_trigger") and low <= open_trade["entry_price"] - open_trade["trailing_trigger"]:
                        new_sl = low + open_trade["trailing_distance"]
                        if new_sl < open_trade["sl"]:
                            open_trade["sl"] = new_sl

                if exit_price is not None:
                    pnl = (exit_price - open_trade["entry_price"]) if open_trade["direction"] == "BUY" else (open_trade["entry_price"] - exit_price)
                    trades.append(SimulatedTrade(
                        strategy=strategy.name,
                        direction=open_trade["direction"],
                        entry_price=open_trade["entry_price"],
                        exit_price=exit_price,
                        sl=open_trade["original_sl"],
                        tp=open_trade["tp"],
                        pnl=pnl,
                        entry_time=open_trade["entry_time"],
                        exit_time=current_time,
                        exit_reason=exit_reason,
                    ))
                    open_trade = None

            # Generate new signal if no open trade
            if open_trade is None:
                signal = strategy.analyze(window)
                if signal is not None:
                    open_trade = {
                        "direction": signal.direction.value,
                        "entry_price": signal.entry_price,
                        "sl": signal.sl,
                        "original_sl": signal.sl,
                        "tp": signal.tp,
                        "entry_time": signal.timestamp,
                        "trailing_trigger": signal.trailing_stop_trigger,
                        "trailing_distance": signal.trailing_stop_distance or 0,
                    }

        # Close any remaining open trade at last price
        if open_trade is not None:
            last = data.iloc[-1]
            exit_price = last["close"]
            pnl = (exit_price - open_trade["entry_price"]) if open_trade["direction"] == "BUY" else (open_trade["entry_price"] - exit_price)
            trades.append(SimulatedTrade(
                strategy=strategy.name,
                direction=open_trade["direction"],
                entry_price=open_trade["entry_price"],
                exit_price=exit_price,
                sl=open_trade["original_sl"],
                tp=open_trade["tp"],
                pnl=pnl,
                entry_time=open_trade["entry_time"],
                exit_time=last.get("datetime", datetime.now()),
                exit_reason="end",
            ))

        logger.info(f"Backtest complete: {len(trades)} trades")
        return trades
