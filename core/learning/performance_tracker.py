"""Rolling strategy performance stats and reward tracking."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class StrategyStats:
    """Rolling performance stats for a single strategy."""
    name: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    recent_pnls: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 2),
        }


class PerformanceTracker:
    """Track rolling performance per strategy over a sliding window.

    Used for:
    - Dynamic weight adjustment in ensemble
    - Optimization triggers (when performance degrades)
    - RL reward curve tracking
    """

    def __init__(self, window_size: int = 50) -> None:
        self._window = window_size
        self._trades: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._all_trades: dict[str, list[float]] = defaultdict(list)
        self._rl_rewards: list[float] = []

    def record_trade(self, strategy_name: str, pnl: float) -> None:
        """Record a completed trade PnL for a strategy."""
        self._trades[strategy_name].append(pnl)
        self._all_trades[strategy_name].append(pnl)

    def record_rl_reward(self, reward: float) -> None:
        """Record an RL step reward."""
        self._rl_rewards.append(reward)

    def get_stats(self, strategy_name: str) -> StrategyStats:
        """Get rolling stats for a strategy."""
        pnls = list(self._trades.get(strategy_name, []))
        if not pnls:
            return StrategyStats(name=strategy_name)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0

        return StrategyStats(
            name=strategy_name,
            trades=len(pnls),
            wins=len(wins),
            losses=len(losses),
            total_pnl=sum(pnls),
            avg_pnl=sum(pnls) / len(pnls),
            win_rate=len(wins) / len(pnls) if pnls else 0.0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 999.0,
            recent_pnls=pnls[-10:],
        )

    def get_all_stats(self) -> list[StrategyStats]:
        """Get stats for all tracked strategies."""
        return [self.get_stats(name) for name in self._trades]

    def get_rl_reward_curve(self) -> list[float]:
        """Return RL cumulative reward curve."""
        if not self._rl_rewards:
            return []
        cumulative = []
        total = 0.0
        for r in self._rl_rewards:
            total += r
            cumulative.append(round(total, 2))
        return cumulative

    def should_optimize(self, strategy_name: str, min_trades: int = 20, min_win_rate: float = 0.4) -> bool:
        """Check if a strategy's performance has degraded enough to trigger optimization."""
        stats = self.get_stats(strategy_name)
        if stats.trades < min_trades:
            return False
        return stats.win_rate < min_win_rate

    def get_weight_scores(self) -> dict[str, float]:
        """Calculate performance-based weight scores for ensemble.

        Higher score = better recent performance.
        Returns values in [0.1, 1.0] range.
        """
        all_stats = self.get_all_stats()
        if not all_stats:
            return {}

        scores: dict[str, float] = {}
        for s in all_stats:
            if s.trades < 5:
                scores[s.name] = 0.5  # neutral for new strategies
            else:
                # Combine win rate and profit factor
                wr_score = s.win_rate
                pf_score = min(s.profit_factor / 3.0, 1.0)  # cap at PF=3
                scores[s.name] = max(0.1, (wr_score * 0.6 + pf_score * 0.4))

        return scores
