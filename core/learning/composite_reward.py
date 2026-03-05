"""Composite reward wrapper: adds drawdown/Sortino/consistency/transaction cost penalties."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from core.learning.rl_environment import TradingEnvironment, TradeResult


class CompositeRewardWrapper:
    """Wraps TradingEnvironment to adjust rewards on trade close.

    Adjustments applied to the base reward:
      - Drawdown penalty: penalizes when drawdown exceeds threshold
      - Sortino penalty: penalizes high downside deviation
      - Consistency bonus: rewards stable positive returns
      - Transaction cost: penalizes position changes (open/close)
    """

    def __init__(
        self,
        env: TradingEnvironment,
        drawdown_weight: float = 0.5,
        drawdown_threshold: float = 0.02,
        sortino_weight: float = 0.3,
        sortino_window: int = 20,
        consistency_weight: float = 0.05,
        transaction_weight: float = 0.1,
        transaction_fee_rate: float = 0.001,
    ) -> None:
        self._env = env
        self._drawdown_weight = drawdown_weight
        self._drawdown_threshold = drawdown_threshold
        self._sortino_weight = sortino_weight
        self._consistency_weight = consistency_weight
        self._transaction_weight = transaction_weight
        self._transaction_fee_rate = transaction_fee_rate

        self._returns: deque[float] = deque(maxlen=sortino_window)
        self._peak_equity: float = 0.0
        self._cumulative_pnl: float = 0.0
        self._last_action: float = 0.0

    # Delegate all environment properties
    @property
    def in_position(self) -> bool:
        return self._env.in_position

    @property
    def total_reward(self) -> float:
        return self._env.total_reward

    @property
    def trade_count(self) -> int:
        return self._env.trade_count

    @property
    def win_rate(self) -> float:
        return self._env.win_rate

    def reset(self) -> None:
        self._env.reset()
        self._returns.clear()
        self._peak_equity = 0.0
        self._cumulative_pnl = 0.0
        self._last_action = 0.0

    def get_state_dict(self) -> dict[str, Any]:
        return self._env.get_state_dict()

    def step(
        self,
        action: float,
        current_price: float,
        features: list[float],
        atr: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
    ) -> tuple[float, TradeResult | None]:
        """Step the environment and adjust reward on trade close."""
        reward, trade_result = self._env.step(
            action, current_price, features, atr=atr, high=high, low=low,
        )

        if trade_result is not None:
            # Track returns for Sortino calculation
            self._returns.append(trade_result.pnl)
            self._cumulative_pnl += trade_result.pnl

            # Update peak equity
            if self._cumulative_pnl > self._peak_equity:
                self._peak_equity = self._cumulative_pnl

            # Compute adjustments
            adjusted = reward
            base_abs = abs(reward) if abs(reward) > 0.01 else 1.0

            # 1. Drawdown penalty
            drawdown = self._peak_equity - self._cumulative_pnl
            dd_pct = drawdown / max(self._peak_equity, 1e-6) if self._peak_equity > 0 else 0.0
            if dd_pct > self._drawdown_threshold:
                adjusted -= self._drawdown_weight * (dd_pct - self._drawdown_threshold) * base_abs

            # 2. Sortino penalty (downside deviation)
            if len(self._returns) >= 5:
                returns_arr = np.array(self._returns)
                negative_returns = returns_arr[returns_arr < 0]
                if len(negative_returns) > 0:
                    downside_std = float(np.std(negative_returns))
                    adjusted -= self._sortino_weight * downside_std

            # 3. Consistency bonus
            if len(self._returns) >= 5:
                returns_arr = np.array(self._returns)
                mean_ret = float(returns_arr.mean())
                std_ret = float(returns_arr.std())
                if mean_ret > 0 and std_ret > 0:
                    adjusted += self._consistency_weight * (mean_ret / std_ret)

            # 4. Transaction cost penalty
            if self._transaction_weight > 0:
                # Penalize the round-trip (open + close)
                position_change = abs(action - self._last_action)
                transaction_cost = position_change * self._transaction_fee_rate * current_price
                adjusted -= self._transaction_weight * transaction_cost

            # Update the trade result with adjusted reward
            trade_result = TradeResult(
                action=trade_result.action,
                entry_price=trade_result.entry_price,
                exit_price=trade_result.exit_price,
                pnl=trade_result.pnl,
                reward=round(adjusted, 4),
                hold_bars=trade_result.hold_bars,
                sl_price=trade_result.sl_price,
                tp_price=trade_result.tp_price,
                exit_reason=trade_result.exit_reason,
            )
            reward = adjusted

        self._last_action = action
        return reward, trade_result
