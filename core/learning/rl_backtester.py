"""RL backtester: train any BaseAgent on historical candles bar-by-bar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from config import STRATEGY_PROFILES
from core.learning.base_agent import BaseAgent
from core.learning.feature_engine import FeatureEngine
from core.learning.regime_detector import RegimeDetector
from core.learning.replay_buffer import Transition
from core.learning.rl_environment import TradingEnvironment, TradeResult


@dataclass
class RLBacktestConfig:
    """Configuration for an RL backtesting run."""

    timeframe: str = "H1"
    count: int = 2000
    epochs: int = 3
    train_every: int = 5
    profile: str = "medium"


@dataclass
class RLTradeRecord:
    """A single trade from the backtest run."""

    epoch: int
    action: str  # "buy" or "sell" (continuous action direction)
    entry_price: float
    exit_price: float
    pnl: float
    reward: float
    hold_bars: int
    bar_index: int
    exit_reason: str = "signal"
    sl_price: float = 0.0
    tp_price: float = 0.0


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
    profile: str = "medium"
    equity_curve: list[dict[str, float]] = field(default_factory=list)
    trades: list[RLTradeRecord] = field(default_factory=list)
    epoch_stats: list[dict[str, Any]] = field(default_factory=list)
    claude_review: dict[str, Any] | None = None


class RLBacktester:
    """Run any BaseAgent through historical candles for offline training.

    Reuses existing components: FeatureEngine, BaseAgent, TradingEnvironment,
    RegimeDetector. Supports multiple epochs over the same data so the agent
    can learn progressively.
    """

    def __init__(
        self,
        config: RLBacktestConfig | None = None,
        agent: BaseAgent | None = None,
        env: TradingEnvironment | None = None,
        feature_engine: FeatureEngine | None = None,
        regime_detector: RegimeDetector | None = None,
    ) -> None:
        self._config = config or RLBacktestConfig()
        tf = self._config.timeframe

        # Look up strategy profile
        profile_name = self._config.profile
        profile = STRATEGY_PROFILES.get(profile_name)

        # Configure agent from profile or use factory
        if agent is not None:
            self._agent = agent
        else:
            self._agent = self._create_default_agent(profile)

        # Configure environment from profile
        if env is not None:
            self._env = env
        elif profile is not None:
            self._env = TradingEnvironment(
                reward_scale=profile.reward_scale,
                penalty_scale=profile.penalty_scale,
                max_hold_bars=profile.max_hold_bars,
                sl_atr_mult=profile.sl_atr_mult,
                tp_atr_mult=profile.tp_atr_mult,
                trailing_atr_mult=profile.trailing_atr_mult,
            )
        else:
            self._env = TradingEnvironment()

        self._feature_engine = feature_engine or FeatureEngine(primary_tf=tf)
        self._regime_detector = regime_detector or RegimeDetector(primary_tf=tf)

    def _create_default_agent(self, profile=None) -> BaseAgent:
        """Create a default agent using the factory.

        Overrides initial_random_steps to 0 so training starts immediately
        during backtesting (backtests are too short for the default 10k warmup).
        """
        try:
            from dataclasses import replace
            from config import AppConfig, SACConfig, DDPGConfig
            from core.learning.agent_factory import create_agent

            cfg = AppConfig()
            # Zero out initial_random_steps for backtesting
            if hasattr(cfg, "sac"):
                cfg = replace(cfg, sac=replace(cfg.sac, initial_random_steps=0))
            if hasattr(cfg, "ddpg"):
                cfg = replace(cfg, ddpg=replace(cfg.ddpg, initial_random_steps=0))
            return create_agent(cfg)
        except Exception:
            # Fallback to DQN if factory fails
            from core.learning.rl_agent import RLAgent
            if profile is not None:
                return RLAgent(
                    epsilon_start=profile.epsilon_start,
                    epsilon_decay=profile.epsilon_decay,
                )
            return RLAgent()

    def run(
        self,
        data: pd.DataFrame,
        htf_data: dict[str, pd.DataFrame] | None = None,
        on_epoch: Callable[[dict[str, Any]], None] | None = None,
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> RLBacktestResult:
        """Run the RL agent through historical data for multiple epochs."""
        cfg = self._config
        min_bars = self._min_bars_needed()
        if len(data) < min_bars + 10:
            logger.warning(
                f"Not enough data: {len(data)} bars, need at least {min_bars + 10}"
            )
            return RLBacktestResult(profile=cfg.profile)

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

            total_bars = len(data) - 1 - min_bars
            for i in range(min_bars, len(data) - 1):
                if on_progress and (i - min_bars) % 200 == 0:
                    on_progress(epoch, i - min_bars, total_bars)

                window = data.iloc[: i + 1]
                features = self._feature_engine.extract(window, htf_data)
                state = features["vector"]

                self._regime_detector.update(features)

                # Agent selects continuous action
                action, info = self._agent.select_action(state)

                bar_high = float(data.iloc[i]["high"])
                bar_low = float(data.iloc[i]["low"])
                bar_atr = float(features.get("atr", 0.0))

                # Environment step with float action
                reward, trade_result = self._env.step(
                    action, features["close"], state,
                    atr=bar_atr, high=bar_high, low=bar_low,
                )

                # Build next state
                next_window = data.iloc[: i + 2]
                next_features = self._feature_engine.extract(next_window, htf_data)
                next_state = next_features["vector"]

                done = trade_result is not None

                # Store transition with float action
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
                        action=trade_result.action,  # already "buy" or "sell" string
                        entry_price=trade_result.entry_price,
                        exit_price=trade_result.exit_price,
                        pnl=trade_result.pnl,
                        reward=trade_result.reward,
                        hold_bars=trade_result.hold_bars,
                        bar_index=i,
                        exit_reason=trade_result.exit_reason,
                        sl_price=trade_result.sl_price,
                        tp_price=trade_result.tp_price,
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
            stats["cumulative_pnl"] = round(cumulative_pnl, 5)
            epoch_stats.append(stats)

            if on_epoch:
                on_epoch(stats)

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
            profile=cfg.profile,
            equity_curve=equity_curve,
            trades=all_trades,
            epoch_stats=epoch_stats,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _min_bars_needed(self) -> int:
        return max(
            self._feature_engine._ema_slow,
            self._feature_engine._bb_period,
            self._feature_engine._adx_period,
        ) + 5

    @staticmethod
    def _compute_max_drawdown(
        equity_curve: list[dict[str, float]],
    ) -> float:
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
    def agent(self) -> BaseAgent:
        return self._agent
