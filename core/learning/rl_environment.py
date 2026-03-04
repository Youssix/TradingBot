"""Trading environment for RL agent: state → action → reward."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from loguru import logger


class Action(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradeResult:
    """Result of a completed RL trade."""
    action: Action
    entry_price: float
    exit_price: float
    pnl: float
    reward: float
    hold_bars: int


class TradingEnvironment:
    """RL trading environment.

    State: feature vector from FeatureEngine (23 dims)
    Actions: HOLD (0), BUY (1), SELL (2)
    Rewards:
      - Winning trade: +PnL * reward_scale
      - Losing trade: -|PnL| * penalty_scale (asymmetric to discourage losses)
      - HOLD: small negative to avoid excessive inaction
      - Quick close penalty: extra penalty for trades closed in < min_hold bars
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
        penalty_scale: float = 1.5,
        hold_penalty: float = -0.01,
        min_hold_bars: int = 1,
        max_hold_bars: int = 15,
    ) -> None:
        self._reward_scale = reward_scale
        self._penalty_scale = penalty_scale
        self._hold_penalty = hold_penalty
        self._min_hold_bars = min_hold_bars
        self._max_hold_bars = max_hold_bars

        # Current state
        self._position: Action | None = None  # None = flat, BUY/SELL = in position
        self._entry_price: float = 0.0
        self._hold_bars: int = 0
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

    def step(
        self,
        action: int,
        current_price: float,
        features: list[float],
    ) -> tuple[float, TradeResult | None]:
        """Execute action and return (reward, trade_result_or_None).

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            current_price: Current close price
            features: Current feature vector (for logging)

        Returns:
            Tuple of (reward, trade_result if a trade was closed else None)
        """
        action_enum = Action(action)
        reward = 0.0
        trade_result = None

        if not self.in_position:
            # Not in a position — can open one
            if action_enum in (Action.BUY, Action.SELL):
                self._position = action_enum
                self._entry_price = current_price
                self._hold_bars = 0
                reward = 0.0  # no reward for opening
            else:
                reward = self._hold_penalty  # small penalty for doing nothing
        else:
            self._hold_bars += 1

            # Check if we should close (opposite signal or max hold)
            should_close = False
            if action_enum == Action.HOLD:
                if self._hold_bars >= self._max_hold_bars:
                    should_close = True  # force close after max hold
            elif action_enum != self._position:
                should_close = True  # opposite signal or HOLD closes position

            if should_close:
                trade_result = self._close_position(current_price)
                reward = trade_result.reward
            else:
                # Still holding — give tiny reward proportional to unrealized PnL
                if self._position == Action.BUY:
                    unrealized = (current_price - self._entry_price) / self._entry_price
                else:
                    unrealized = (self._entry_price - current_price) / self._entry_price
                reward = unrealized * 0.1  # tiny signal

        self._total_reward += reward
        return reward, trade_result

    def _close_position(self, exit_price: float) -> TradeResult:
        """Close the current position and compute reward."""
        assert self._position is not None

        if self._position == Action.BUY:
            pnl = exit_price - self._entry_price
        else:
            pnl = self._entry_price - exit_price

        # Normalize PnL by entry price for consistent reward scale
        pnl_pct = pnl / self._entry_price if self._entry_price > 0 else 0.0

        # Asymmetric reward
        if pnl_pct > 0:
            reward = pnl_pct * self._reward_scale * 100  # scale up for learning
            self._win_count += 1
        else:
            reward = pnl_pct * self._penalty_scale * 100

        # Quick-close penalty
        if self._hold_bars < self._min_hold_bars:
            reward -= 0.1

        self._trade_count += 1
        result = TradeResult(
            action=self._position,
            entry_price=self._entry_price,
            exit_price=exit_price,
            pnl=round(pnl, 5),
            reward=round(reward, 4),
            hold_bars=self._hold_bars,
        )

        # Reset position
        self._position = None
        self._entry_price = 0.0
        self._hold_bars = 0

        return result

    def get_state_dict(self) -> dict[str, Any]:
        """Serialize current environment state."""
        return {
            "in_position": self.in_position,
            "position_type": self._position.name if self._position else None,
            "entry_price": self._entry_price,
            "hold_bars": self._hold_bars,
            "total_reward": round(self._total_reward, 2),
            "trade_count": self._trade_count,
            "win_rate": round(self.win_rate, 4),
        }
