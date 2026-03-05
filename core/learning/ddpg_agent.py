"""Deep Deterministic Policy Gradient (DDPG) agent with continuous actions.

Deterministic policy with Ornstein-Uhlenbeck exploration noise.
"""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
from loguru import logger

from core.learning.base_agent import BaseAgent
from core.learning.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, Transition

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — DDPG agent will use random fallback")


CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck noise
# ---------------------------------------------------------------------------

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(
        self,
        size: int = 1,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ) -> None:
        self._mu = mu
        self._theta = theta
        self._sigma = sigma
        self._state = np.full(size, mu, dtype=np.float64)

    def reset(self) -> None:
        self._state = np.full_like(self._state, self._mu)

    def sample(self) -> np.ndarray:
        dx = self._theta * (self._mu - self._state) + self._sigma * np.random.randn(*self._state.shape)
        self._state += dx
        return self._state.copy()


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------

def _init_final_layer(layer: nn.Linear, bound: float = 3e-3) -> None:
    """Initialize final layer with small uniform weights."""
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


if TORCH_AVAILABLE:
    class DeterministicActor(nn.Module):
        """Deterministic policy: state → action in [-1, +1]."""

        def __init__(self, state_dim: int, hidden_dim: int = 256, device: torch.device | None = None) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.output_head = nn.Linear(hidden_dim, 1)
            _init_final_layer(self.output_head)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = self.net(state)
            return torch.tanh(self.output_head(x))

    class DDPGCritic(nn.Module):
        """Q-network: (state, action) → scalar Q-value."""

        def __init__(self, state_dim: int, hidden_dim: int = 256, device: torch.device | None = None) -> None:
            super().__init__()
            input_dim = state_dim + 1
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.output_head = nn.Linear(hidden_dim, 1)
            _init_final_layer(self.output_head)

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            if action.dim() == 1:
                action = action.unsqueeze(-1)
            sa = torch.cat([state, action], dim=-1)
            x = self.net(sa)
            return self.output_head(x)


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------

class DDPGAgent(BaseAgent):
    """Deep Deterministic Policy Gradient with OU noise exploration.

    Uses a deterministic actor (state → action) with target networks
    and soft updates. Optionally uses Prioritized Experience Replay.
    """

    def __init__(
        self,
        state_dim: int = 23,
        hidden_dim: int = 256,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        ou_theta: float = 0.15,
        ou_sigma: float = 0.2,
        initial_random_steps: int = 10_000,
        use_per: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        device: torch.device | str | None = None,
    ) -> None:
        self._state_dim = state_dim
        self._hidden_dim = hidden_dim
        self._gamma = gamma
        self._tau = tau
        self._batch_size = batch_size
        self._initial_random_steps = initial_random_steps
        self._use_per = use_per

        self._episode = 0
        self._total_reward = 0.0
        self._training = True
        self._total_steps = 0

        # OU noise
        self._noise = OUNoise(size=1, theta=ou_theta, sigma=ou_sigma)

        # Buffer
        if use_per:
            self._buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity, alpha=per_alpha, beta_start=per_beta_start,
            )
        else:
            self._buffer = ReplayBuffer(capacity=buffer_capacity)

        if TORCH_AVAILABLE:
            if device is None:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, str):
                self._device = torch.device(device)
            else:
                self._device = device

            # Actor
            self._actor = DeterministicActor(state_dim, hidden_dim).to(self._device)
            self._actor_target = DeterministicActor(state_dim, hidden_dim).to(self._device)
            self._actor_target.load_state_dict(self._actor.state_dict())
            self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=actor_lr)

            # Critic
            self._critic = DDPGCritic(state_dim, hidden_dim).to(self._device)
            self._critic_target = DDPGCritic(state_dim, hidden_dim).to(self._device)
            self._critic_target.load_state_dict(self._critic.state_dict())
            self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=critic_lr)
        else:
            self._device = None
            self._actor = None
            self._actor_target = None
            self._critic = None
            self._critic_target = None

    # -- BaseAgent interface --

    @property
    def epsilon(self) -> float:
        return self._noise._sigma

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
        if val:
            self._noise.reset()

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def select_action(self, state: list[float]) -> tuple[float, dict]:
        self._total_steps += 1

        if not TORCH_AVAILABLE or self._actor is None:
            return random.uniform(-1.0, 1.0), {"mean": 0.0, "noise": 0.0}

        # Random actions during initial exploration
        if self._training and self._total_steps < self._initial_random_steps:
            action = random.uniform(-1.0, 1.0)
            return action, {"mean": 0.0, "noise": action, "random": True}

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)
            action_t = self._actor(state_t)
            mean_action = float(action_t.squeeze().item())

        if self._training:
            noise = float(self._noise.sample()[0])
            action = float(np.clip(mean_action + noise, -1.0, 1.0))
        else:
            noise = 0.0
            action = mean_action

        # Get Q-value estimate
        q_value = 0.0
        if TORCH_AVAILABLE and self._critic is not None:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)
                action_t = torch.FloatTensor([action]).to(self._device)
                q_value = float(self._critic(state_t, action_t).item())

        return action, {
            "mean": mean_action,
            "noise": noise,
            "q_value": q_value,
        }

    def store_transition(self, transition: Transition) -> None:
        self._buffer.push(transition)

    def train_step(self) -> float | None:
        if not TORCH_AVAILABLE or self._actor is None:
            return None
        if len(self._buffer) < self._batch_size:
            return None
        if self._total_steps < self._initial_random_steps:
            return None

        # Sample batch
        if self._use_per:
            batch, weights, indices = self._buffer.sample(self._batch_size)
            weights_t = torch.FloatTensor(weights).to(self._device)
        else:
            batch = self._buffer.sample_arrays(self._batch_size)
            weights_t = torch.ones(self._batch_size, device=self._device)
            indices = None

        states = torch.FloatTensor(batch["states"]).to(self._device)
        actions = torch.FloatTensor(batch["actions"]).to(self._device)
        rewards = torch.FloatTensor(batch["rewards"]).unsqueeze(-1).to(self._device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self._device)
        dones = torch.FloatTensor(batch["dones"]).unsqueeze(-1).to(self._device)

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self._actor_target(next_states).squeeze(-1)
            target_q = self._critic_target(next_states, next_actions)
            target_q = rewards + self._gamma * (1 - dones) * target_q

        current_q = self._critic(states, actions)
        td_errors = (current_q - target_q).detach()

        critic_loss = (weights_t.unsqueeze(-1) * F.mse_loss(current_q, target_q, reduction="none")).mean()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1.0)
        self._critic_optimizer.step()

        # --- Actor update ---
        actor_actions = self._actor(states).squeeze(-1)
        actor_loss = -self._critic(states, actor_actions).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 1.0)
        self._actor_optimizer.step()

        # --- Soft target update ---
        self._soft_update(self._actor, self._actor_target)
        self._soft_update(self._critic, self._critic_target)

        # Update PER priorities
        if self._use_per and indices is not None:
            td_np = td_errors.squeeze(-1).cpu().numpy()
            self._buffer.update_priorities(indices, td_np)

        # Log training metrics
        with torch.no_grad():
            mean_q = float(current_q.mean().item())
            q_std = float(current_q.std().item())

        logger.debug(
            f"DDPG train | critic_loss={critic_loss.item():.4f} actor_loss={actor_loss.item():.4f} "
            f"mean_q={mean_q:.4f} q_std={q_std:.4f} buffer={len(self._buffer)} "
            f"{'mean_priority=' + str(round(self._buffer.mean_priority, 4)) + ' ' if self._use_per else ''}"
        )

        return float(critic_loss.item())

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self._tau * p.data + (1 - self._tau) * tp.data)

    def end_episode(self, episode_reward: float) -> None:
        self._episode += 1
        self._total_reward += episode_reward
        self._noise.reset()

    def save_checkpoint(self) -> bytes | None:
        if not TORCH_AVAILABLE or self._actor is None:
            return None
        buf = io.BytesIO()
        torch.save({
            "version": CHECKPOINT_VERSION,
            "actor": self._actor.state_dict(),
            "actor_target": self._actor_target.state_dict(),
            "critic": self._critic.state_dict(),
            "critic_target": self._critic_target.state_dict(),
            "actor_optimizer": self._actor_optimizer.state_dict(),
            "critic_optimizer": self._critic_optimizer.state_dict(),
            "episode": self._episode,
            "total_reward": self._total_reward,
            "total_steps": self._total_steps,
        }, buf)
        return buf.getvalue()

    def load_checkpoint(self, data: bytes) -> None:
        if not TORCH_AVAILABLE or self._actor is None:
            return
        buf = io.BytesIO(data)
        checkpoint = torch.load(buf, map_location=self._device, weights_only=False)
        self._actor.load_state_dict(checkpoint["actor"])
        self._actor_target.load_state_dict(checkpoint["actor_target"])
        self._critic.load_state_dict(checkpoint["critic"])
        self._critic_target.load_state_dict(checkpoint["critic_target"])
        self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._episode = checkpoint["episode"]
        self._total_reward = checkpoint["total_reward"]
        self._total_steps = checkpoint.get("total_steps", 0)
        logger.info(f"Loaded DDPG checkpoint v{checkpoint.get('version', 0)}: episode={self._episode}")

    def get_stats(self) -> dict[str, Any]:
        return {
            "episode": self._episode,
            "epsilon": round(self._noise._sigma, 4),
            "total_reward": round(self._total_reward, 2),
            "buffer_size": len(self._buffer),
            "training": self._training,
            "torch_available": TORCH_AVAILABLE,
            "agent_type": "ddpg",
            "total_steps": self._total_steps,
            "use_per": self._use_per,
            "mean_priority": round(self._buffer.mean_priority, 4) if self._use_per else 0.0,
        }
