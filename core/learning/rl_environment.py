"""Trading environment for RL agent: continuous action space [-1, +1]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


# Keep Action enum for backward compat with legacy DQN code paths
class Action:
    """Legacy action constants (kept for adapter compatibility)."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradeResult:
    """Result of a completed RL trade."""
    action: str  # "buy" or "sell"
    entry_price: float
    exit_price: float
    pnl: float
    reward: float
    hold_bars: int
    sl_price: float = 0.0
    tp_price: float = 0.0
    exit_reason: str = "signal"


class TradingEnvironment:
    """RL trading environment — continuous action space.

    Action: float in [-1, +1]
      - sign = direction (positive = buy, negative = sell)
      - magnitude = conviction / position size factor
      - |action| > open_threshold to open a position
      - opposing action beyond close_threshold to close

    Reward design: reward = PnL, nothing else.
    Rewards:
      - Trade close: pnl_pct x reward_scale (pure PnL, high scale)
      - TP hit: extra 20% of PnL reward
      - Trailing exit: extra 10% of PnL reward
      - HOLD while in profit: tiny positive
      - HOLD while flat: hold_penalty
      - Quick-close penalty: -0.05
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
        penalty_scale: float = 1.5,
        hold_penalty: float = -0.01,
        min_hold_bars: int = 1,
        max_hold_bars: int = 15,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
        trailing_atr_mult: float = 1.0,
        spread: float = 0.30,
        open_threshold: float = 0.3,
        close_threshold: float = 0.1,
    ) -> None:
        self._reward_scale = reward_scale
        self._penalty_scale = penalty_scale
        self._hold_penalty = hold_penalty
        self._min_hold_bars = min_hold_bars
        self._max_hold_bars = max_hold_bars
        self._sl_atr_mult = sl_atr_mult
        self._tp_atr_mult = tp_atr_mult
        self._trailing_atr_mult = trailing_atr_mult
        self._spread = spread
        self._open_threshold = open_threshold
        self._close_threshold = close_threshold

        # Current state
        self._position: str | None = None  # None = flat, "buy"/"sell" = in position
        self._entry_price: float = 0.0
        self._hold_bars: int = 0
        self._sl_price: float = 0.0
        self._tp_price: float = 0.0
        self._sl_trailed: bool = False
        self._total_reward: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0

    @property
    def in_position(self) -> bool:
        return self._position is not None

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def win_rate(self) -> float:
        return self._win_count / self._trade_count if self._trade_count > 0 else 0.0

    def reset(self) -> None:
        """Reset environment state."""
        self._position = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._sl_price = 0.0
        self._tp_price = 0.0
        self._sl_trailed = False

    def step(
        self,
        action: float,
        current_price: float,
        features: list[float],
        atr: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
    ) -> tuple[float, TradeResult | None]:
        """Execute action and return (reward, trade_result_or_None).

        Args:
            action: float in [-1, +1]. Sign = direction, magnitude = conviction.
            current_price: Current bar close price.
            features: State feature vector.
            atr: Current ATR for SL/TP computation.
            high: Current bar high (for SL/TP checks).
            low: Current bar low (for SL/TP checks).
        """
        action = float(np.clip(action, -1.0, 1.0))
        magnitude = abs(action)
        reward = 0.0
        trade_result = None

        if not self.in_position:
            # Not in a position -- can open one if magnitude > threshold
            if magnitude > self._open_threshold:
                direction = "buy" if action > 0 else "sell"
                self._position = direction
                self._entry_price = current_price
                self._hold_bars = 0

                # Compute SL/TP from ATR
                if atr > 0:
                    if direction == "buy":
                        self._sl_price = current_price - atr * self._sl_atr_mult
                        self._tp_price = current_price + atr * self._tp_atr_mult
                    else:
                        self._sl_price = current_price + atr * self._sl_atr_mult
                        self._tp_price = current_price - atr * self._tp_atr_mult
                else:
                    self._sl_price = 0.0
                    self._tp_price = 0.0
                self._sl_trailed = False

                reward = 0.0  # no reward for opening
            else:
                reward = self._hold_penalty  # mild nudge to not sit flat forever
        else:
            self._hold_bars += 1

            # --- Check SL/TP/trailing before signal-based close ---
            exit_reason = ""
            exit_price = current_price

            # Stop-loss check
            if self._sl_price > 0:
                if self._position == "buy" and low > 0 and low <= self._sl_price:
                    exit_reason = "trailing" if self._sl_trailed else "sl"
                    exit_price = self._sl_price
                elif self._position == "sell" and high > 0 and high >= self._sl_price:
                    exit_reason = "trailing" if self._sl_trailed else "sl"
                    exit_price = self._sl_price

            # Take-profit check
            if not exit_reason and self._tp_price > 0:
                if self._position == "buy" and high > 0 and high >= self._tp_price:
                    exit_reason = "tp"
                    exit_price = self._tp_price
                elif self._position == "sell" and low > 0 and low <= self._tp_price:
                    exit_reason = "tp"
                    exit_price = self._tp_price

            # Trailing stop update
            if not exit_reason and atr > 0 and self._trailing_atr_mult > 0 and self._sl_price > 0:
                if self._position == "buy":
                    unrealized = current_price - self._entry_price
                    if unrealized > self._trailing_atr_mult * atr:
                        new_sl = current_price - self._trailing_atr_mult * atr
                        if new_sl > self._sl_price:
                            self._sl_price = new_sl
                            self._sl_trailed = True
                else:
                    unrealized = self._entry_price - current_price
                    if unrealized > self._trailing_atr_mult * atr:
                        new_sl = current_price + self._trailing_atr_mult * atr
                        if new_sl < self._sl_price:
                            self._sl_price = new_sl
                            self._sl_trailed = True

            # Signal-based close (if no SL/TP/trailing triggered)
            if not exit_reason:
                # Close if opposing action exceeds close threshold
                if self._position == "buy" and action < -self._close_threshold:
                    exit_reason = "signal"
                    exit_price = current_price
                elif self._position == "sell" and action > self._close_threshold:
                    exit_reason = "signal"
                    exit_price = current_price
                elif magnitude <= self._close_threshold and self._hold_bars >= self._max_hold_bars:
                    # Low conviction hold + max bars => force close
                    exit_reason = "max_hold"
                    exit_price = current_price

            if exit_reason:
                trade_result = self._close_position(exit_price, exit_reason)
                reward = trade_result.reward
            else:
                # Still holding — reward based on unrealized P&L direction
                if self._position == "buy":
                    unrealized_pct = (current_price - self._entry_price) / self._entry_price
                else:
                    unrealized_pct = (self._entry_price - current_price) / self._entry_price

                if unrealized_pct > 0:
                    reward = unrealized_pct * 10
                else:
                    reward = 0.0

        self._total_reward += reward
        return reward, trade_result

    def _close_position(
        self, exit_price: float, exit_reason: str = "signal"
    ) -> TradeResult:
        """Close the current position and compute reward."""
        assert self._position is not None

        if self._position == "buy":
            pnl = exit_price - self._entry_price
        else:
            pnl = self._entry_price - exit_price

        pnl_pct = pnl / self._entry_price if self._entry_price > 0 else 0.0

        reward = pnl_pct * self._reward_scale * 1000

        if pnl_pct > 0:
            self._win_count += 1

        # Exit-reason multipliers
        if exit_reason == "tp" and pnl_pct > 0:
            reward *= 1.2
        elif exit_reason == "trailing" and pnl_pct > 0:
            reward *= 1.1

        # Quick-close penalty
        if self._hold_bars < self._min_hold_bars:
            reward -= 0.05

        self._trade_count += 1
        result = TradeResult(
            action=self._position,
            entry_price=self._entry_price,
            exit_price=exit_price,
            pnl=round(pnl, 5),
            reward=round(reward, 4),
            hold_bars=self._hold_bars,
            sl_price=round(self._sl_price, 5),
            tp_price=round(self._tp_price, 5),
            exit_reason=exit_reason,
        )

        # Reset position
        self._position = None
        self._entry_price = 0.0
        self._hold_bars = 0
        self._sl_price = 0.0
        self._tp_price = 0.0
        self._sl_trailed = False

        return result

    def get_state_dict(self) -> dict[str, Any]:
        """Serialize current environment state."""
        return {
            "in_position": self.in_position,
            "position_type": self._position,
            "entry_price": self._entry_price,
            "hold_bars": self._hold_bars,
            "sl_price": self._sl_price,
            "tp_price": self._tp_price,
            "total_reward": round(self._total_reward, 2),
            "trade_count": self._trade_count,
            "win_rate": round(self.win_rate, 4),
        }
