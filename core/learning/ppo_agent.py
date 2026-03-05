"""Proximal Policy Optimization (PPO) agent with continuous Gaussian policy.

Uses LayerNorm, 256-dim hidden layers, small weight init on final layers.
"""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from loguru import logger

from core.learning.base_agent import BaseAgent
from core.learning.replay_buffer import Transition

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


CHECKPOINT_VERSION = 2


def _init_final_layer(layer: nn.Linear, bound: float = 3e-3) -> None:
    """Initialize final layer with small uniform weights."""
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """Shared-trunk actor-critic with Gaussian policy head + LayerNorm."""

        def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            # Actor head: mean + log_std
            self.mean_head = nn.Linear(hidden_dim, 1)
            self.log_std_head = nn.Linear(hidden_dim, 1)
            # Critic head: state value
            self.value_head = nn.Linear(hidden_dim, 1)

            _init_final_layer(self.mean_head)
            _init_final_layer(self.log_std_head)
            _init_final_layer(self.value_head)

        def forward(self, state: torch.Tensor):
            x = self.shared(state)
            mean = self.mean_head(x)
            log_std = self.log_std_head(x).clamp(-20, 2)
            value = self.value_head(x)
            return mean, log_std, value

        def get_action(self, state: torch.Tensor):
            mean, log_std, value = self.forward(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            u = dist.rsample()
            action = torch.tanh(u)
            log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action.squeeze(-1), log_prob.squeeze(-1), value.squeeze(-1), mean.squeeze(-1), std.squeeze(-1)

        def evaluate(self, states: torch.Tensor, actions_raw: torch.Tensor):
            """Evaluate actions for PPO update (re-compute log_prob, value, entropy)."""
            mean, log_std, value = self.forward(states)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            # Inverse tanh to get u from action
            actions_clamped = actions_raw.unsqueeze(-1).clamp(-0.999, 0.999)
            u = torch.atanh(actions_clamped)
            log_prob = dist.log_prob(u) - torch.log(1 - actions_clamped.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True).squeeze(-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            return log_prob, value.squeeze(-1), entropy


# ---------------------------------------------------------------------------
# Rollout buffer for on-policy data
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """On-policy rollout storage for PPO."""

    def __init__(self, capacity: int = 2048) -> None:
        self._capacity = capacity
        self.states: list[list[float]] = []
        self.actions: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def push(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    @property
    def full(self) -> bool:
        return len(self.states) >= self._capacity

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent(BaseAgent):
    """Continuous PPO with Gaussian policy, GAE advantages, LayerNorm.

    Features:
    - 256-dim hidden layers with LayerNorm
    - Small weight init on final layers
    - Device agnostic (cpu/cuda)
    - Checkpoint versioning
    """

    def __init__(
        self,
        state_dim: int = 23,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        rollout_size: int = 2048,
        n_epochs_per_update: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: torch.device | str | None = None,
    ) -> None:
        self._state_dim = state_dim
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._clip_epsilon = clip_epsilon
        self._entropy_coeff = entropy_coeff
        self._value_coeff = value_coeff
        self._n_epochs = n_epochs_per_update
        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm

        self._rollout = RolloutBuffer(capacity=rollout_size)
        self._episode = 0
        self._total_reward = 0.0
        self._training = True

        # Cache last action info for store_transition
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0

        if TORCH_AVAILABLE:
            if device is None:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, str):
                self._device = torch.device(device)
            else:
                self._device = device
            self._net = ActorCritic(state_dim, hidden_dim).to(self._device)
            self._optimizer = optim.Adam(self._net.parameters(), lr=lr)
        else:
            self._device = None
            self._net = None
            self._optimizer = None

    # -- BaseAgent interface --

    @property
    def epsilon(self) -> float:
        return self._clip_epsilon

    @property
    def episode(self) -> int:
        return self._episode

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, val: bool) -> None:
        self._training = val

    @property
    def buffer_size(self) -> int:
        return len(self._rollout)

    def select_action(self, state: list[float]) -> tuple[float, dict]:
        if not TORCH_AVAILABLE or self._net is None:
            action = random.uniform(-1.0, 1.0)
            return action, {"mean": 0.0, "std": 1.0, "value": 0.0}

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)
            action, log_prob, value, mean, std = self._net.get_action(state_t)

            action_val = float(action.item())
            self._last_log_prob = float(log_prob.item())
            self._last_value = float(value.item())

        return action_val, {
            "mean": float(mean.item()),
            "std": float(std.item()),
            "value": float(value.item()),
        }

    def store_transition(self, transition: Transition) -> None:
        self._rollout.push(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            done=transition.done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )

    def train_step(self) -> float | None:
        if not TORCH_AVAILABLE or self._net is None:
            return None
        if not self._rollout.full:
            return None

        # Compute GAE advantages
        advantages, returns = self._compute_gae()

        # Convert to tensors
        states = torch.FloatTensor(self._rollout.states).to(self._device)
        actions = torch.FloatTensor(self._rollout.actions).to(self._device)
        old_log_probs = torch.FloatTensor(self._rollout.log_probs).to(self._device)
        advantages_t = torch.FloatTensor(advantages).to(self._device)
        returns_t = torch.FloatTensor(returns).to(self._device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss = 0.0
        n = len(self._rollout)

        for _ in range(self._n_epochs):
            # Mini-batch sampling
            indices = np.random.permutation(n)
            for start in range(0, n, self._batch_size):
                end = min(start + self._batch_size, n)
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages_t[idx]
                mb_returns = returns_t[idx]

                # Evaluate current policy
                new_log_probs, values, entropy = self._net.evaluate(mb_states, mb_actions)

                # Clipped surrogate
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = ratio.clamp(1 - self._clip_epsilon, 1 + self._clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Total loss
                loss = policy_loss + self._value_coeff * value_loss - self._entropy_coeff * entropy

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), self._max_grad_norm)
                self._optimizer.step()

                total_loss += float(loss.item())

        avg_loss = total_loss / max(self._n_epochs, 1)
        logger.debug(
            f"PPO train | policy_loss={float(policy_loss.item()):.4f} "
            f"value_loss={float(value_loss.item()):.4f} "
            f"entropy={float(entropy.item()):.4f} "
            f"total_loss={avg_loss:.4f} "
            f"rollout_size={len(self._rollout)} "
            f"epochs={self._n_epochs}"
        )

        self._rollout.clear()
        return total_loss / max(self._n_epochs, 1)

    def _compute_gae(self) -> tuple[list[float], list[float]]:
        """Compute Generalized Advantage Estimation."""
        rewards = self._rollout.rewards
        values = self._rollout.values
        dones = self._rollout.dones
        n = len(rewards)

        advantages = [0.0] * n
        returns = [0.0] * n
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(n)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self._gamma * next_value * mask - values[t]
            gae = delta + self._gamma * self._gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        return advantages, returns

    def end_episode(self, episode_reward: float) -> None:
        self._episode += 1
        self._total_reward += episode_reward

    def save_checkpoint(self) -> bytes | None:
        if not TORCH_AVAILABLE or self._net is None:
            return None
        buf = io.BytesIO()
        torch.save({
            "version": CHECKPOINT_VERSION,
            "net": self._net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "episode": self._episode,
            "total_reward": self._total_reward,
        }, buf)
        return buf.getvalue()

    def load_checkpoint(self, data: bytes) -> None:
        if not TORCH_AVAILABLE or self._net is None:
            return
        buf = io.BytesIO(data)
        checkpoint = torch.load(buf, map_location=self._device, weights_only=False)
        self._net.load_state_dict(checkpoint["net"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._episode = checkpoint["episode"]
        self._total_reward = checkpoint["total_reward"]
        logger.info(f"Loaded PPO checkpoint v{checkpoint.get('version', 0)}: episode={self._episode}")

    def get_stats(self) -> dict[str, Any]:
        return {
            "episode": self._episode,
            "epsilon": round(self._clip_epsilon, 4),
            "total_reward": round(self._total_reward, 2),
            "buffer_size": len(self._rollout),
            "training": self._training,
            "torch_available": TORCH_AVAILABLE,
            "agent_type": "ppo",
        }
