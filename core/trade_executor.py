from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from core.mt5_client import MT5Client
from core.risk_manager import RiskManager
from utils.db import TradeDB


class TradeExecutor:
    """Orchestrates trade execution with risk validation and logging."""

    def __init__(
        self,
        mt5_client: MT5Client,
        risk_manager: RiskManager,
        db: TradeDB,
        dry_run: bool = False,
        point_value: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize trade executor.

        Args:
            mt5_client: MT5 client for order execution.
            risk_manager: Risk manager for trade validation.
            db: Database for trade logging.
            dry_run: If True, log signals but don't execute orders.
            point_value: Value per point per lot (for position sizing).
            max_retries: Maximum retry attempts for failed orders.
        """
        self._client = mt5_client
        self._risk = risk_manager
        self._db = db
        self._dry_run = dry_run
        self._point_value = point_value
        self._max_retries = max_retries

    def execute_signal(
        self,
        signal: Any,
        account_info: dict[str, Any],
        open_positions: list[dict[str, Any]],
        daily_trades_count: int,
        daily_pnl: float,
        initial_balance: float | None = None,
    ) -> dict[str, Any] | None:
        """Execute a trading signal after risk validation.

        Args:
            signal: Trading signal with direction, entry_price, sl, tp, strategy_name.
            account_info: Current account information dict.
            open_positions: List of currently open positions.
            daily_trades_count: Number of trades executed today.
            daily_pnl: Today's realized PnL.
            initial_balance: Starting balance for drawdown calc.

        Returns:
            Order result dict if executed, None if blocked or dry-run.
        """
        now = datetime.now(timezone.utc)

        # Risk validation
        allowed, reason = self._risk.can_trade(
            account_info=account_info,
            open_positions=open_positions,
            daily_trades_count=daily_trades_count,
            daily_pnl=daily_pnl,
            current_time=now,
            initial_balance=initial_balance,
        )

        if not allowed:
            logger.warning(f"Trade blocked by risk manager: {reason}")
            return None

        # Calculate position size
        sl_points = abs(signal.entry_price - signal.sl)
        lot_size = self._risk.calculate_position_size(
            account_balance=account_info.get("balance", 0.0),
            sl_points=sl_points,
            point_value=self._point_value,
        )

        if lot_size <= 0:
            logger.warning("Position size calculation returned 0, skipping trade")
            return None

        logger.info(
            f"Signal: {signal.direction} | Entry: {signal.entry_price:.2f} | "
            f"SL: {signal.sl:.2f} | TP: {signal.tp:.2f} | Size: {lot_size} lots | "
            f"Strategy: {signal.strategy_name}"
        )

        # Dry-run mode
        if self._dry_run:
            logger.info("[DRY-RUN] Order would be placed but dry-run mode is active")
            trade_record = {
                "strategy": signal.strategy_name,
                "symbol": "XAUUSD",
                "direction": signal.direction.value,
                "entry_price": signal.entry_price,
                "exit_price": None,
                "sl": signal.sl,
                "tp": signal.tp,
                "lot_size": lot_size,
                "pnl": 0.0,
                "opened_at": now.isoformat(),
                "closed_at": None,
                "status": "dry-run",
            }
            self._db.insert_trade(trade_record)
            return {
                "ticket": 0,
                "price": signal.entry_price,
                "volume": lot_size,
                "comment": "dry-run",
            }

        # Execute order with retry
        order_result = self._send_order_with_retry(
            symbol="XAUUSD",
            order_type=str(signal.direction),
            volume=lot_size,
            price=signal.entry_price,
            sl=signal.sl,
            tp=signal.tp,
            comment=f"{signal.strategy_name}",
        )

        if order_result is None:
            logger.error("Order execution failed after all retries")
            return None

        # Verify SL placement
        if not self._verify_sl_placement(order_result["ticket"], signal.sl):
            logger.error(
                f"SL verification failed for ticket #{order_result['ticket']}, closing position"
            )
            self._client.close_position(order_result["ticket"])
            return None

        # Log to database
        trade_record = {
            "strategy": signal.strategy_name,
            "symbol": "XAUUSD",
            "direction": signal.direction.value,
            "entry_price": order_result["price"],
            "exit_price": None,
            "sl": signal.sl,
            "tp": signal.tp,
            "lot_size": order_result["volume"],
            "pnl": 0.0,
            "opened_at": now.isoformat(),
            "closed_at": None,
            "status": "open",
        }
        self._db.insert_trade(trade_record)
        logger.info(f"Order executed: ticket #{order_result['ticket']}")

        return order_result

    def _send_order_with_retry(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str,
    ) -> dict[str, Any] | None:
        """Send order with retry logic."""
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._client.send_order(
                    symbol=symbol,
                    order_type=order_type,
                    volume=volume,
                    price=price,
                    sl=sl,
                    tp=tp,
                    comment=comment,
                )
                return result
            except Exception as e:
                logger.warning(
                    f"Order attempt {attempt}/{self._max_retries} failed: {e}"
                )
                if attempt < self._max_retries:
                    time.sleep(1.0 * attempt)
        return None

    def _verify_sl_placement(
        self, ticket: int, expected_sl: float, tolerance: float = 0.5
    ) -> bool:
        """Verify that SL was placed correctly on the position."""
        try:
            positions = self._client.get_positions("XAUUSD")
            for pos in positions:
                if pos["ticket"] == ticket:
                    actual_sl = pos.get("sl", 0.0)
                    if abs(actual_sl - expected_sl) <= tolerance:
                        return True
                    logger.warning(
                        f"SL mismatch: expected {expected_sl:.2f}, got {actual_sl:.2f}"
                    )
                    return False
            logger.warning(f"Position #{ticket} not found for SL verification")
            return False
        except Exception as e:
            logger.error(f"SL verification error: {e}")
            return False

    def manage_trailing_stops(
        self,
        positions: list[dict[str, Any]],
        signals: dict[int, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Update trailing stops for open positions.

        Args:
            positions: List of open position dicts.
            signals: Optional mapping of ticket -> original Signal for trailing params.

        Returns:
            List of modified position dicts.
        """
        modified = []
        for pos in positions:
            ticket = pos["ticket"]
            current_profit = pos.get("profit", 0.0)

            # Simple trailing stop: if in profit, trail the SL
            if current_profit > 0:
                try:
                    tick = self._client.get_tick(pos["symbol"])
                    current_price = (
                        tick["bid"] if pos["type"] == "BUY" else tick["ask"]
                    )
                    entry_price = pos["price_open"]
                    current_sl = pos.get("sl", 0.0)

                    if pos["type"] == "BUY":
                        new_sl = current_price - abs(entry_price - current_sl)
                        if new_sl > current_sl:
                            logger.info(
                                f"Trailing SL for #{ticket}: {current_sl:.2f} -> {new_sl:.2f}"
                            )
                            modified.append({**pos, "sl": new_sl})
                    elif pos["type"] == "SELL":
                        new_sl = current_price + abs(current_sl - entry_price)
                        if new_sl < current_sl:
                            logger.info(
                                f"Trailing SL for #{ticket}: {current_sl:.2f} -> {new_sl:.2f}"
                            )
                            modified.append({**pos, "sl": new_sl})
                except Exception as e:
                    logger.error(f"Trailing stop error for #{ticket}: {e}")

        return modified

    def close_all_positions(self) -> list[dict[str, Any]]:
        """Close all open positions. Returns list of close results."""
        results = []
        try:
            positions = self._client.get_positions()
            for pos in positions:
                try:
                    success = self._client.close_position(pos["ticket"])
                    results.append({"ticket": pos["ticket"], "closed": success})
                    if success:
                        logger.info(f"Closed position #{pos['ticket']}")
                    else:
                        logger.error(f"Failed to close position #{pos['ticket']}")
                except Exception as e:
                    logger.error(f"Error closing position #{pos['ticket']}: {e}")
                    results.append(
                        {"ticket": pos["ticket"], "closed": False, "error": str(e)}
                    )
        except Exception as e:
            logger.error(f"Error getting positions for close_all: {e}")
        return results
