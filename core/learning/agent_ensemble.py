"""Multi-agent ensemble with configurable action selection strategies.

Strategies:
  - "weighted_average" (DEFAULT): softmax over rolling Sharpe, blend all actions
  - "best_sharpe": pick agent with highest rolling Sharpe ratio
  - "majority_vote": discretize actions into sell/hold/buy zones, take majority
"""

from __future__ import annotations

import io
from collections import deque
from typing import Any

import numpy as np
from loguru import logger

from core.learning.base_agent import BaseAgent
from core.learning.replay_buffer import Transition

CHECKPOINT_VERSION = 2


class AgentEnsemble(BaseAgent):
    """Ensemble of BaseAgent instances with multiple selection strategies.

    All agents receive every transition and train simultaneously.
    Tracks per-agent rolling Sharpe, max drawdown, and win rate.
    """

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        strategy: str = "weighted_average",
        sharpe_window: int = 100,
        eval_interval: int = 20,
    ) -> None:
        self._agents = agents
        self._strategy = strategy
        self._sharpe_window = sharpe_window
        self._eval_interval = eval_interval

        # Per-agent tracking
        self._reward_histories: dict[str, deque[float]] = {
            name: deque(maxlen=sharpe_window) for name in agents
        }
        self._cumulative_rewards: dict[str, float] = {name: 0.0 for name in agents}
        self._peak_rewards: dict[str, float] = {name: 0.0 for name in agents}
        self._max_drawdowns: dict[str, float] = {name: 0.0 for name in agents}
        self._win_counts: dict[str, int] = {name: 0 for name in agents}
        self._trade_counts: dict[str, int] = {name: 0 for name in agents}

        # Active agent (for best_sharpe strategy)
        self._active_name = next(iter(agents))
        self._episode_count = 0
        self._total_reward = 0.0
        self._training = True

        # Cache per-agent Sharpe ratios
        self._agent_sharpes: dict[str, float] = {name: 0.0 for name in agents}

    @property
    def active_agent_name(self) -> str:
        return self._active_name

    @property
    def active_agent(self) -> BaseAgent:
        return self._agents[self._active_name]

    @property
    def strategy(self) -> str:
        return self._strategy

    # -- BaseAgent interface --

    @property
    def epsilon(self) -> float:
        return self.active_agent.epsilon

    @property
    def episode(self) -> int:
        return self._episode_count

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool) -> None:
        self._training = val
        for agent in self._agents.values():
            agent.training = val

    @property
    def buffer_size(self) -> int:
        return self.active_agent.buffer_size

    def select_action(self, state: list[float]) -> tuple[float, dict]:
        """Select action based on current strategy."""
        if self._strategy == "weighted_average":
            return self._weighted_average_action(state)
        elif self._strategy == "majority_vote":
            return self._majority_vote_action(state)
        else:  # "best_sharpe"
            action, info = self.active_agent.select_action(state)
            info["active_agent"] = self._active_name
            info["strategy"] = "best_sharpe"
            return action, info

    def _weighted_average_action(self, state: list[float]) -> tuple[float, dict]:
        """Softmax over rolling Sharpe ratios, blend all agent actions."""
        actions = {}
        infos = {}
        for name, agent in self._agents.items():
            action, info = agent.select_action(state)
            actions[name] = action
            infos[name] = info

        sharpes = np.array([self._agent_sharpes.get(name, 0.0) for name in self._agents])

        # Softmax with temperature
        temperature = 1.0
        sharpes_shifted = sharpes - sharpes.max()  # numerical stability
        exp_sharpes = np.exp(sharpes_shifted / temperature)
        weights = exp_sharpes / exp_sharpes.sum()

        # Weighted blend
        action_values = np.array([actions[name] for name in self._agents])
        blended_action = float(np.clip(np.dot(weights, action_values), -1.0, 1.0))

        # Build info
        weight_dict = {name: round(float(w), 4) for name, w in zip(self._agents, weights)}
        active_name = max(weight_dict, key=weight_dict.get)

        return blended_action, {
            "strategy": "weighted_average",
            "weights": weight_dict,
            "per_agent_actions": {name: round(a, 4) for name, a in actions.items()},
            "active_agent": active_name,
            "q_value": infos.get(active_name, {}).get("q_value", 0.0),
            "mean": infos.get(active_name, {}).get("mean", 0.0),
        }

    def _majority_vote_action(self, state: list[float]) -> tuple[float, dict]:
        """Discretize actions into sell/hold/buy zones, take majority."""
        votes = {"sell": 0.0, "hold": 0.0, "buy": 0.0}
        actions = {}
        infos = {}
        threshold = 0.3

        for name, agent in self._agents.items():
            action, info = agent.select_action(state)
            actions[name] = action
            infos[name] = info

            sharpe = max(self._agent_sharpes.get(name, 0.0), 0.01)
            if action > threshold:
                votes["buy"] += sharpe
            elif action < -threshold:
                votes["sell"] += sharpe
            else:
                votes["hold"] += sharpe

        # Winner takes all
        winner = max(votes, key=votes.get)
        if winner == "buy":
            # Average of buy-voting agents
            buy_actions = [a for n, a in actions.items() if a > threshold]
            blended = float(np.mean(buy_actions)) if buy_actions else 0.8
        elif winner == "sell":
            sell_actions = [a for n, a in actions.items() if a < -threshold]
            blended = float(np.mean(sell_actions)) if sell_actions else -0.8
        else:
            blended = 0.0

        active_name = max(self._agent_sharpes, key=self._agent_sharpes.get, default=self._active_name)

        return float(np.clip(blended, -1.0, 1.0)), {
            "strategy": "majority_vote",
            "votes": {k: round(v, 4) for k, v in votes.items()},
            "winner": winner,
            "per_agent_actions": {name: round(a, 4) for name, a in actions.items()},
            "active_agent": active_name,
        }

    def store_transition(self, transition: Transition) -> None:
        """All agents receive every transition."""
        for agent in self._agents.values():
            agent.store_transition(transition)

    def train_step(self) -> float | None:
        """Train all agents, return active agent's loss."""
        losses = {}
        for name, agent in self._agents.items():
            loss = agent.train_step()
            if loss is not None:
                losses[name] = loss

        if losses:
            loss_str = " ".join(f"{n}={l:.4f}" for n, l in losses.items())
            sharpe_str = " ".join(f"{n}={s:.3f}" for n, s in self._agent_sharpes.items())
            dd_str = " ".join(f"{n}={d:.4f}" for n, d in self._max_drawdowns.items())
            wr_str = " ".join(
                f"{n}={self._win_counts[n] / max(self._trade_counts[n], 1):.2%}"
                for n in self._agents
            )
            logger.debug(
                f"Ensemble train | strategy={self._strategy} active={self._active_name} "
                f"losses=[{loss_str}] sharpes=[{sharpe_str}] "
                f"max_dd=[{dd_str}] win_rate=[{wr_str}]"
            )

        return losses.get(self._active_name)

    def end_episode(self, episode_reward: float) -> None:
        """End episode for all agents, update stats, and potentially re-evaluate."""
        self._episode_count += 1
        self._total_reward += episode_reward

        for name, agent in self._agents.items():
            agent.end_episode(episode_reward)
            self._reward_histories[name].append(episode_reward)

            # Update per-agent stats
            self._cumulative_rewards[name] += episode_reward
            if self._cumulative_rewards[name] > self._peak_rewards[name]:
                self._peak_rewards[name] = self._cumulative_rewards[name]
            drawdown = self._peak_rewards[name] - self._cumulative_rewards[name]
            if drawdown > self._max_drawdowns[name]:
                self._max_drawdowns[name] = drawdown

            self._trade_counts[name] += 1
            if episode_reward > 0:
                self._win_counts[name] += 1

        # Update Sharpe ratios
        self._update_sharpes()

        # Re-evaluate active agent periodically (for best_sharpe strategy)
        if self._episode_count % self._eval_interval == 0:
            self._select_best_agent()

    def _update_sharpes(self) -> None:
        """Recompute rolling Sharpe ratios for all agents."""
        for name, history in self._reward_histories.items():
            if len(history) >= 5:
                rewards = np.array(history)
                std = rewards.std()
                self._agent_sharpes[name] = float(rewards.mean() / std if std > 0 else 0.0)
            else:
                self._agent_sharpes[name] = 0.0

    def _select_best_agent(self) -> None:
        """Select agent with highest rolling Sharpe ratio."""
        best_name = self._active_name
        best_sharpe = -float("inf")

        for name, sharpe in self._agent_sharpes.items():
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_name = name

        if best_name != self._active_name:
            logger.info(
                f"Ensemble: switching active agent {self._active_name} → {best_name} "
                f"(Sharpe={best_sharpe:.3f})"
            )
            self._active_name = best_name

    def get_agent_stats(self) -> dict[str, dict[str, Any]]:
        """Get detailed per-agent performance metrics."""
        stats = {}
        for name in self._agents:
            win_rate = (
                self._win_counts[name] / self._trade_counts[name]
                if self._trade_counts[name] > 0 else 0.0
            )
            stats[name] = {
                "sharpe": round(self._agent_sharpes.get(name, 0.0), 4),
                "cumulative_reward": round(self._cumulative_rewards[name], 4),
                "max_drawdown": round(self._max_drawdowns[name], 4),
                "win_rate": round(win_rate, 4),
                "trade_count": self._trade_counts[name],
                "buffer_size": self._agents[name].buffer_size,
            }
        return stats

    def save_checkpoint(self) -> bytes | None:
        """Save all sub-agents + ensemble metadata."""
        import pickle
        data = {
            "version": CHECKPOINT_VERSION,
            "active_name": self._active_name,
            "episode_count": self._episode_count,
            "total_reward": self._total_reward,
            "strategy": self._strategy,
            "reward_histories": {
                name: list(h) for name, h in self._reward_histories.items()
            },
            "cumulative_rewards": dict(self._cumulative_rewards),
            "peak_rewards": dict(self._peak_rewards),
            "max_drawdowns": dict(self._max_drawdowns),
            "win_counts": dict(self._win_counts),
            "trade_counts": dict(self._trade_counts),
            "agent_checkpoints": {},
        }
        for name, agent in self._agents.items():
            blob = agent.save_checkpoint()
            if blob is not None:
                data["agent_checkpoints"][name] = blob

        return pickle.dumps(data)

    def load_checkpoint(self, data: bytes) -> None:
        """Load all sub-agents + ensemble metadata."""
        import pickle
        loaded = pickle.loads(data)
        self._active_name = loaded.get("active_name", self._active_name)
        self._episode_count = loaded.get("episode_count", 0)
        self._total_reward = loaded.get("total_reward", 0.0)
        self._strategy = loaded.get("strategy", self._strategy)

        for name, history_list in loaded.get("reward_histories", {}).items():
            if name in self._reward_histories:
                self._reward_histories[name] = deque(history_list, maxlen=self._sharpe_window)

        for key in ("cumulative_rewards", "peak_rewards", "max_drawdowns", "win_counts", "trade_counts"):
            src = loaded.get(key, {})
            dest = getattr(self, f"_{key}")
            for name, val in src.items():
                if name in dest:
                    dest[name] = val

        for name, blob in loaded.get("agent_checkpoints", {}).items():
            if name in self._agents:
                self._agents[name].load_checkpoint(blob)

        self._update_sharpes()
        logger.info(
            f"Loaded ensemble checkpoint v{loaded.get('version', 0)}: "
            f"active={self._active_name}, episodes={self._episode_count}"
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "episode": self._episode_count,
            "epsilon": round(self.epsilon, 4),
            "total_reward": round(self._total_reward, 2),
            "buffer_size": self.buffer_size,
            "training": self._training,
            "torch_available": True,
            "agent_type": "ensemble",
            "active_agent": self._active_name,
            "strategy": self._strategy,
            "agent_sharpes": {n: round(s, 4) for n, s in self._agent_sharpes.items()},
            "agent_stats": self.get_agent_stats(),
        }
