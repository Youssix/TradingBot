"""RL backtester: train the DQN agent on historical candles bar-by-bar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from core.learning.feature_engine import FeatureEngine
from core.learning.regime_detector import RegimeDetector
from core.learning.replay_buffer import Transition
from core.learning.rl_agent import RLAgent
from core.learning.rl_environment import Action, TradingEnvironment, TradeResult


@dataclass
class RLBacktestConfig:
    """Configuration for an RL backtesting run."""

    timeframe: str = "H1"
    count: int = 2000
    epochs: int = 3
    train_every: int = 5


@dataclass
class RLTradeRecord:
    """A single trade from the backtest run."""

    epoch: int
    action: str
    entry_price: float
    exit_price: float
    pnl: float
    reward: float
    hold_bars: int
    bar_index: int


@dataclass
class RLBacktestResult:
    """Complete results of an RL backtesting run."""

    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_reward: float = 0.0
    avg_reward_per_trade: float = 0.0
    max_drawdown: float = 0.0
    final_epsilon: float = 1.0
    episodes: int = 0
    epochs_completed: int = 0
    equity_curve: list[dict[str, float]] = field(default_factory=list)
    trades: list[RLTradeRecord] = field(default_factory=list)
    epoch_stats: list[dict[str, Any]] = field(default_factory=list)
    claude_review: dict[str, Any] | None = None


class RLBacktester:
    """Run the RL agent through historical candles for offline training.

    Reuses existing components: FeatureEngine, RLAgent, TradingEnvironment,
    RegimeDetector. Supports multiple epochs over the same data so the agent
    can learn progressively.
    """

    def __init__(
        self,
        config: RLBacktestConfig | None = None,
        agent: RLAgent | None = None,
        env: TradingEnvironment | None = None,
        feature_engine: FeatureEngine | None = None,
        regime_detector: RegimeDetector | None = None,
    ) -> None:
        self._config = config or RLBacktestConfig()
        tf = self._config.timeframe

        self._agent = agent or RLAgent()
        self._env = env or TradingEnvironment()
        self._feature_engine = feature_engine or FeatureEngine(primary_tf=tf)
        self._regime_detector = regime_detector or RegimeDetector(primary_tf=tf)

    def run(
        self,
        data: pd.DataFrame,
        htf_data: dict[str, pd.DataFrame] | None = None,
    ) -> RLBacktestResult:
        """Run the RL agent through historical data for multiple epochs.

        Args:
            data: Primary timeframe OHLCV DataFrame with columns
                  [datetime, open, high, low, close, volume].
            htf_data: Optional higher-timeframe DataFrames for context features.

        Returns:
            RLBacktestResult with trades, equity curve, and statistics.
        """
        cfg = self._config
        min_bars = self._min_bars_needed()
        if len(data) < min_bars + 10:
            logger.warning(
                f"Not enough data: {len(data)} bars, need at least {min_bars + 10}"
            )
            return RLBacktestResult()

        all_trades: list[RLTradeRecord] = []
        equity_curve: list[dict[str, float]] = []
        epoch_stats: list[dict[str, Any]] = []
        cumulative_pnl = 0.0
        global_step = 0

        for epoch in range(cfg.epochs):
            self._env.reset()
            epoch_trades: list[RLTradeRecord] = []
            epoch_reward = 0.0
            train_losses: list[float] = []

            for i in range(min_bars, len(data)):
                window = data.iloc[: i + 1]

                features = self._feature_engine.extract(window, htf_data)
                state = features["vector"]

                # Update regime
                self._regime_detector.update(features)

                # Agent selects action
                action, q_values = self._agent.select_action(state)

                # Environment step
                reward, trade_result = self._env.step(
                    action, features["close"], state
                )

                # Build next state (peek at current bar as proxy)
                next_state = state  # next bar not yet available; self-transition
                done = trade_result is not None

                # Store transition
                self._agent.store_transition(
                    Transition(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                    )
                )

                # Train periodically
                global_step += 1
                if global_step % cfg.train_every == 0:
                    loss = self._agent.train_step()
                    if loss is not None:
                        train_losses.append(loss)

                epoch_reward += reward

                # Record completed trade
                if trade_result is not None:
                    self._agent.end_episode(trade_result.reward)

                    record = RLTradeRecord(
                        epoch=epoch,
                        action=Action(trade_result.action).name,
                        entry_price=trade_result.entry_price,
                        exit_price=trade_result.exit_price,
                        pnl=trade_result.pnl,
                        reward=trade_result.reward,
                        hold_bars=trade_result.hold_bars,
                        bar_index=i,
                    )
                    epoch_trades.append(record)
                    all_trades.append(record)

                    cumulative_pnl += trade_result.pnl
                    equity_curve.append(
                        {
                            "trade_index": len(all_trades),
                            "cumulative_pnl": round(cumulative_pnl, 5),
                        }
                    )

            # End-of-epoch stats
            epoch_wins = sum(1 for t in epoch_trades if t.pnl > 0)
            epoch_wr = (
                epoch_wins / len(epoch_trades) if epoch_trades else 0.0
            )
            avg_loss_val = (
                np.mean(train_losses).item() if train_losses else 0.0
            )

            stats = {
                "epoch": epoch,
                "trades": len(epoch_trades),
                "wins": epoch_wins,
                "win_rate": round(epoch_wr, 4),
                "total_reward": round(epoch_reward, 4),
                "avg_train_loss": round(avg_loss_val, 6),
                "epsilon": round(self._agent.epsilon, 4),
            }
            epoch_stats.append(stats)
            logger.info(
                f"Epoch {epoch}: {len(epoch_trades)} trades, "
                f"WR={epoch_wr:.1%}, reward={epoch_reward:.2f}, "
                f"eps={self._agent.epsilon:.4f}"
            )

        # Compute aggregate metrics
        total_trades = len(all_trades)
        wins = sum(1 for t in all_trades if t.pnl > 0)
        win_rate = wins / total_trades if total_trades else 0.0
        total_pnl = sum(t.pnl for t in all_trades)
        total_reward = sum(t.reward for t in all_trades)
        avg_reward = total_reward / total_trades if total_trades else 0.0

        max_dd = self._compute_max_drawdown(equity_curve)

        return RLBacktestResult(
            total_trades=total_trades,
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 5),
            total_reward=round(total_reward, 4),
            avg_reward_per_trade=round(avg_reward, 4),
            max_drawdown=round(max_dd, 5),
            final_epsilon=round(self._agent.epsilon, 4),
            episodes=self._agent.episode,
            epochs_completed=cfg.epochs,
            equity_curve=equity_curve,
            trades=all_trades,
            epoch_stats=epoch_stats,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _min_bars_needed(self) -> int:
        """Minimum bars before feature extraction works."""
        return max(
            self._feature_engine._ema_slow,
            self._feature_engine._bb_period,
            self._feature_engine._adx_period,
        ) + 5

    @staticmethod
    def _compute_max_drawdown(
        equity_curve: list[dict[str, float]],
    ) -> float:
        """Compute maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0
        peak = 0.0
        max_dd = 0.0
        for pt in equity_curve:
            pnl = pt["cumulative_pnl"]
            if pnl > peak:
                peak = pnl
            dd = peak - pnl
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def agent(self) -> RLAgent:
        return self._agent
