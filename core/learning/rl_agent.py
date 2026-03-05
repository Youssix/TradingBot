"""DQN reinforcement learning agent — adapted to BaseAgent continuous interface."""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from loguru import logger

from core.learning.base_agent import BaseAgent
from core.learning.replay_buffer import ReplayBuffer, Transition

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — RL agent will use random fallback")


# ---------------------------------------------------------------------------
# Neural network (only defined if torch is available)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class DQNetwork(nn.Module):
        """Small MLP for Q-value estimation: state_dim → 64 → 32 → 3 actions."""

        def __init__(self, state_dim: int = 23, action_dim: int = 3) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


# ---------------------------------------------------------------------------
# DQN Agent (BaseAgent adapter)
# ---------------------------------------------------------------------------

# Discrete-to-continuous mapping:
# HOLD(0) → 0.0, BUY(1) → +1.0, SELL(2) → -1.0
_DISCRETE_TO_CONTINUOUS = {0: 0.0, 1: 1.0, 2: -1.0}
_CONTINUOUS_TO_DISCRETE_THRESHOLDS = 0.3  # |action| > 0.3 → trade


class RLAgent(BaseAgent):
    """DQN agent adapted to continuous BaseAgent interface.

    Internally still uses discrete actions (0=HOLD, 1=BUY, 2=SELL)
    but wraps output as continuous float and accepts float actions.
    """

    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(
        self,
        state_dim: int = 23,
        action_dim: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 50,
    ) -> None:
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._gamma = gamma
        self._epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq

        self._buffer = ReplayBuffer(capacity=buffer_capacity)
        self._episode = 0
        self._total_reward = 0.0
        self._training = True

        if TORCH_AVAILABLE:
            self._device = torch.device("cpu")
            self._policy_net = DQNetwork(state_dim, action_dim).to(self._device)
            self._target_net = DQNetwork(state_dim, action_dim).to(self._device)
            self._target_net.load_state_dict(self._policy_net.state_dict())
            self._target_net.eval()
            self._optimizer = optim.Adam(self._policy_net.parameters(), lr=lr)
            self._loss_fn = nn.SmoothL1Loss()
        else:
            self._policy_net = None
            self._target_net = None
            self._optimizer = None
            self._loss_fn = None

    @property
    def epsilon(self) -> float:
        return self._epsilon

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
        return len(self._buffer)

    def select_action(self, state: list[float]) -> tuple[float, dict]:
        """Select action using epsilon-greedy, return as continuous float.

        Returns:
            (action_float, info_dict) where action_float in [-1, +1].
        """
        if not TORCH_AVAILABLE or self._policy_net is None:
            discrete_action = random.randint(0, self._action_dim - 1)
            q_values = [0.0] * self._action_dim
        elif self._training and random.random() < self._epsilon:
            discrete_action = random.randint(0, self._action_dim - 1)
            q_values = [0.0] * self._action_dim
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)
                q_values_t = self._policy_net(state_t)
                q_values = q_values_t.squeeze().tolist()
                discrete_action = int(q_values_t.argmax(dim=1).item())

        # Map discrete → continuous
        action_float = _DISCRETE_TO_CONTINUOUS[discrete_action]

        return action_float, {
            "q_values": q_values,
            "discrete_action": discrete_action,
            "q_value": max(q_values) if q_values else 0.0,
        }

    def store_transition(self, transition: Transition) -> None:
        """Store transition — convert float action back to discrete internally."""
        # Convert continuous action to discrete for internal DQN buffer
        action_float = transition.action
        if action_float > _CONTINUOUS_TO_DISCRETE_THRESHOLDS:
            discrete = 1  # BUY
        elif action_float < -_CONTINUOUS_TO_DISCRETE_THRESHOLDS:
            discrete = 2  # SELL
        else:
            discrete = 0  # HOLD

        # Store with float action (buffer accepts float now)
        self._buffer.push(Transition(
            state=transition.state,
            action=float(discrete),  # store discrete as float for DQN training
            reward=transition.reward,
            next_state=transition.next_state,
            done=transition.done,
        ))

    def train_step(self) -> float | None:
        """Perform one DQN training step."""
        if not TORCH_AVAILABLE or self._policy_net is None:
            return None

        if len(self._buffer) < self._batch_size:
            return None

        batch = self._buffer.sample_arrays(self._batch_size)
        states = torch.FloatTensor(batch["states"]).to(self._device)
        actions = torch.LongTensor(batch["actions"].astype(np.int64)).to(self._device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self._device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self._device)
        dones = torch.FloatTensor(batch["dones"]).to(self._device)

        q_values = self._policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self._target_net(next_states).max(dim=1)[0]
            target_q = rewards + self._gamma * next_q * (1 - dones)

        loss = self._loss_fn(q_values, target_q)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), 1.0)
        self._optimizer.step()

        return float(loss.item())

    def end_episode(self, episode_reward: float) -> None:
        self._episode += 1
        self._total_reward += episode_reward
        self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)

        if TORCH_AVAILABLE and self._episode % self._target_update_freq == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())
            logger.debug(f"Target network updated at episode {self._episode}")

    def save_checkpoint(self) -> bytes | None:
        if not TORCH_AVAILABLE or self._policy_net is None:
            return None
        buf = io.BytesIO()
        torch.save({
            "policy_net": self._policy_net.state_dict(),
            "target_net": self._target_net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "epsilon": self._epsilon,
            "episode": self._episode,
            "total_reward": self._total_reward,
        }, buf)
        return buf.getvalue()

    def load_checkpoint(self, data: bytes) -> None:
        if not TORCH_AVAILABLE or self._policy_net is None:
            return
        buf = io.BytesIO(data)
        checkpoint = torch.load(buf, map_location=self._device, weights_only=False)
        self._policy_net.load_state_dict(checkpoint["policy_net"])
        self._target_net.load_state_dict(checkpoint["target_net"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._epsilon = checkpoint["epsilon"]
        self._episode = checkpoint["episode"]
        self._total_reward = checkpoint["total_reward"]
        logger.info(f"Loaded RL checkpoint: episode={self._episode}, epsilon={self._epsilon:.4f}")

    def get_stats(self) -> dict[str, Any]:
        return {
            "episode": self._episode,
            "epsilon": round(self._epsilon, 4),
            "total_reward": round(self._total_reward, 2),
            "buffer_size": len(self._buffer),
            "training": self._training,
            "torch_available": TORCH_AVAILABLE,
            "agent_type": "dqn",
        }
