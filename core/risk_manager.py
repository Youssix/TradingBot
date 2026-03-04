from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class RiskConfig:
    """Risk management configuration."""
    risk_pct: float = 1.0
    max_open_trades: int = 2
    max_daily_trades: int = 6
    max_daily_drawdown_pct: float = 3.0
    max_total_drawdown_pct: float = 10.0
    friday_cutoff_hour: int = 20
    news_hours: list[tuple[int, int]] | None = None


class RiskManager:
    """Manages trading risk controls and position sizing."""

    def __init__(self, config: RiskConfig) -> None:
        self._config = config

    def calculate_position_size(
        self, account_balance: float, sl_points: float, point_value: float,
        risk_pct: float | None = None,
    ) -> float:
        """Calculate position size based on account risk percentage.

        Args:
            account_balance: Current account balance.
            sl_points: Stop loss distance in points.
            point_value: Value per point per lot.
            risk_pct: Override risk percentage (uses config default if None).

        Returns:
            Position size in lots, rounded to 2 decimal places.
        """
        risk = risk_pct if risk_pct is not None else self._config.risk_pct
        if sl_points <= 0 or point_value <= 0:
            logger.warning(f"Invalid SL points ({sl_points}) or point value ({point_value})")
            return 0.0
        risk_amount = account_balance * (risk / 100.0)
        lots = risk_amount / (sl_points * point_value)
        return round(max(lots, 0.01), 2)

    def check_max_open_trades(self, current_open: int) -> bool:
        """Check if we can open more trades. Returns True if allowed."""
        allowed = current_open < self._config.max_open_trades
        if not allowed:
            logger.info(f"Max open trades reached: {current_open}/{self._config.max_open_trades}")
        return allowed

    def check_max_daily_trades(self, daily_count: int) -> bool:
        """Check if daily trade limit allows more trades. Returns True if allowed."""
        allowed = daily_count < self._config.max_daily_trades
        if not allowed:
            logger.info(f"Max daily trades reached: {daily_count}/{self._config.max_daily_trades}")
        return allowed

    def check_daily_drawdown(self, daily_pnl: float, account_balance: float) -> bool:
        """Check if daily drawdown is within limits. Returns True if allowed."""
        if account_balance <= 0:
            return False
        dd_pct = abs(min(daily_pnl, 0.0)) / account_balance * 100
        allowed = dd_pct < self._config.max_daily_drawdown_pct
        if not allowed:
            logger.warning(f"Daily drawdown limit hit: {dd_pct:.2f}% >= {self._config.max_daily_drawdown_pct}%")
        return allowed

    def check_total_drawdown(self, total_pnl: float, initial_balance: float) -> bool:
        """Check if total drawdown is within limits. Returns True if allowed."""
        if initial_balance <= 0:
            return False
        dd_pct = abs(min(total_pnl, 0.0)) / initial_balance * 100
        allowed = dd_pct < self._config.max_total_drawdown_pct
        if not allowed:
            logger.warning(f"Total drawdown limit hit: {dd_pct:.2f}% >= {self._config.max_total_drawdown_pct}%")
        return allowed

    def is_friday_cutoff(self, current_time: datetime) -> bool:
        """Check if it's past Friday cutoff time. Returns True if trading should stop."""
        if current_time.weekday() != 4:  # Not Friday
            return False
        utc_time = current_time.astimezone(timezone.utc) if current_time.tzinfo else current_time
        is_cutoff = utc_time.hour >= self._config.friday_cutoff_hour
        if is_cutoff:
            logger.info(f"Friday cutoff active (after {self._config.friday_cutoff_hour}:00 UTC)")
        return is_cutoff

    def is_news_hour(self, current_time: datetime) -> bool:
        """Check if current time falls within news hours. Returns True if in news hour."""
        news_hours = self._config.news_hours or []
        utc_time = current_time.astimezone(timezone.utc) if current_time.tzinfo else current_time
        current_hour = utc_time.hour
        for start_hour, end_hour in news_hours:
            if start_hour <= current_hour < end_hour:
                logger.info(f"News hour active: {start_hour}:00-{end_hour}:00 UTC")
                return True
        return False

    def can_trade(
        self,
        account_info: dict[str, Any],
        open_positions: list[dict[str, Any]],
        daily_trades_count: int,
        daily_pnl: float,
        current_time: datetime,
        initial_balance: float | None = None,
    ) -> tuple[bool, str]:
        """Run all risk checks. Returns (allowed, reason)."""
        balance = account_info.get("balance", 0.0)
        init_balance = initial_balance or balance
        total_pnl = account_info.get("profit", 0.0)

        if self.is_friday_cutoff(current_time):
            return False, "Friday cutoff"
        if self.is_news_hour(current_time):
            return False, "News hour"
        if not self.check_max_open_trades(len(open_positions)):
            return False, f"Max open trades ({self._config.max_open_trades})"
        if not self.check_max_daily_trades(daily_trades_count):
            return False, f"Max daily trades ({self._config.max_daily_trades})"
        if not self.check_daily_drawdown(daily_pnl, balance):
            return False, f"Daily drawdown limit ({self._config.max_daily_drawdown_pct}%)"
        if not self.check_total_drawdown(total_pnl, init_balance):
            return False, f"Total drawdown limit ({self._config.max_total_drawdown_pct}%)"

        return True, "OK"
