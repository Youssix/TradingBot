"""Soft Actor-Critic (SAC) agent with continuous tanh-squashed Gaussian policy.

Supports optional quantile distributional critics for risk-aware trading.
Uses LayerNorm, 256-dim hidden layers, and small weight init on final layers.
Supports Prioritized Experience Replay (PER).
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
    logger.warning("PyTorch not installed — SAC agent will use random fallback")


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6
CHECKPOINT_VERSION = 2


def _init_final_layer(layer: nn.Linear, bound: float = 3e-3) -> None:
    """Initialize final layer with small uniform weights."""
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class GaussianActor(nn.Module):
        """Gaussian policy network with tanh squashing + LayerNorm."""

        def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.mean_head = nn.Linear(hidden_dim, 1)
            self.log_std_head = nn.Linear(hidden_dim, 1)
            _init_final_layer(self.mean_head)
            _init_final_layer(self.log_std_head)

        def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.net(state)
            mean = self.mean_head(x)
            log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
            return mean, log_std

        def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Sample action with reparameterization trick.

            Returns (action, log_prob, mean) where action is tanh-squashed.
            """
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            u = normal.rsample()  # reparameterization trick
            action = torch.tanh(u)

            # Log-prob correction for tanh squashing
            log_prob = normal.log_prob(u) - torch.log(1 - action.pow(2) + EPSILON)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            return action.squeeze(-1), log_prob.squeeze(-1), mean.squeeze(-1)

    class TwinQ(nn.Module):
        """Twin Q-networks for continuous (state, action) inputs + LayerNorm."""

        def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
            super().__init__()
            input_dim = state_dim + 1  # state + action
            self.q1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.q1_head = nn.Linear(hidden_dim, 1)
            _init_final_layer(self.q1_head)

            self.q2 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.q2_head = nn.Linear(hidden_dim, 1)
            _init_final_layer(self.q2_head)

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if action.dim() == 1:
                action = action.unsqueeze(-1)
            sa = torch.cat([state, action], dim=-1)
            return self.q1_head(self.q1(sa)), self.q2_head(self.q2(sa))

    class QuantileTwinQ(nn.Module):
        """Twin quantile Q-networks for distributional RL + LayerNorm."""

        def __init__(self, state_dim: int, hidden_dim: int = 256, n_quantiles: int = 32) -> None:
            super().__init__()
            self.n_quantiles = n_quantiles
            input_dim = state_dim + 1

            self.q1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.q1_head = nn.Linear(hidden_dim, n_quantiles)
            _init_final_layer(self.q1_head)

            self.q2 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.q2_head = nn.Linear(hidden_dim, n_quantiles)
            _init_final_layer(self.q2_head)

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if action.dim() == 1:
                action = action.unsqueeze(-1)
            sa = torch.cat([state, action], dim=-1)
            return self.q1_head(self.q1(sa)), self.q2_head(self.q2(sa))

        def q_values(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Return mean of quantiles as Q-values."""
            q1, q2 = self.forward(state, action)
            return q1.mean(dim=-1, keepdim=True), q2.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent(BaseAgent):
    """Soft Actor-Critic with continuous tanh-squashed Gaussian policy.

    Features:
    - LayerNorm + 256-dim hidden layers + small weight init
    - Optional quantile distributional critics for risk-aware trading
    - Risk sensitivity parameter for adjusting action selection
    - Prioritized Experience Replay support
    - Initial random steps before training begins
    """

    def __init__(
        self,
        state_dim: int = 23,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float = -1.0,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        use_quantile: bool = False,
        n_quantiles: int = 32,
        risk_sensitivity: float = 0.0,
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
        self._use_quantile = use_quantile
        self._n_quantiles = n_quantiles
        self._risk_sensitivity = np.clip(risk_sensitivity, -1.0, 1.0)
        self._initial_random_steps = initial_random_steps
        self._use_per = use_per

        # Buffer
        if use_per:
            self._buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity, alpha=per_alpha, beta_start=per_beta_start,
            )
        else:
            self._buffer = ReplayBuffer(capacity=buffer_capacity)

        self._episode = 0
        self._total_reward = 0.0
        self._training = True
        self._total_steps = 0

        if TORCH_AVAILABLE:
            if device is None:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, str):
                self._device = torch.device(device)
            else:
                self._device = device

            # Actor
            self._actor = GaussianActor(state_dim, hidden_dim).to(self._device)
            self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=actor_lr)

            # Critics
            if use_quantile:
                self._critic = QuantileTwinQ(state_dim, hidden_dim, n_quantiles).to(self._device)
                self._critic_target = QuantileTwinQ(state_dim, hidden_dim, n_quantiles).to(self._device)
            else:
                self._critic = TwinQ(state_dim, hidden_dim).to(self._device)
                self._critic_target = TwinQ(state_dim, hidden_dim).to(self._device)

            self._critic_target.load_state_dict(self._critic.state_dict())
            self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=critic_lr)

            # Entropy temperature (auto-tuned)
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            self._alpha_optimizer = optim.Adam([self._log_alpha], lr=alpha_lr)
            self._target_entropy = target_entropy
        else:
            self._device = None
            self._actor = None
            self._critic = None
            self._critic_target = None
            self._log_alpha = None

    @property
    def alpha(self) -> float:
        if TORCH_AVAILABLE and self._log_alpha is not None:
            return float(self._log_alpha.exp().item())
        return 0.2

    # -- BaseAgent interface --

    @property
    def epsilon(self) -> float:
        return self.alpha  # SAC uses entropy instead of epsilon

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
        self._total_steps += 1

        if not TORCH_AVAILABLE or self._actor is None:
            action = random.uniform(-1.0, 1.0)
            return action, {"mean": 0.0, "std": 1.0, "log_prob": 0.0, "q_value": 0.0}

        # Random actions during initial exploration
        if self._training and self._total_steps < self._initial_random_steps:
            action = random.uniform(-1.0, 1.0)
            return action, {"mean": 0.0, "std": 1.0, "log_prob": 0.0, "q_value": 0.0, "random": True}

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)

            if self._training:
                action, log_prob, mean = self._actor.sample(state_t)
                action_val = float(action.item())
                log_prob_val = float(log_prob.item())
                mean_val = float(mean.item())
            else:
                mean, log_std = self._actor(state_t)
                action_val = float(torch.tanh(mean).squeeze().item())
                log_prob_val = 0.0
                mean_val = float(mean.squeeze().item())

            # Risk-sensitive action selection when distributional
            if self._use_quantile and abs(self._risk_sensitivity) > 0.01:
                action_val = self._risk_adjusted_action(state_t, action_val)

            # Get Q-value estimate
            action_t = torch.FloatTensor([action_val]).to(self._device)
            if self._use_quantile:
                q1, q2 = self._critic.q_values(state_t, action_t)
            else:
                q1, q2 = self._critic(state_t, action_t)
            q_value = float(torch.min(q1, q2).item())

            _, log_std = self._actor(state_t)
            std_val = float(log_std.exp().squeeze().item())

        return action_val, {
            "mean": mean_val,
            "std": std_val,
            "log_prob": log_prob_val,
            "q_value": q_value,
        }

    def _risk_adjusted_action(self, state_t: torch.Tensor, base_action: float) -> float:
        """Adjust action based on risk_sensitivity using quantile estimates.

        risk_sensitivity < 0: risk-averse (weight lower quantiles / CVaR)
        risk_sensitivity > 0: risk-seeking (weight upper quantiles)
        """
        if not self._use_quantile:
            return base_action

        # Evaluate a few candidate actions around the base
        candidates = np.clip(
            [base_action + delta for delta in [-0.2, -0.1, 0.0, 0.1, 0.2]],
            -1.0, 1.0,
        )
        best_action = base_action
        best_score = -float("inf")

        for ca in candidates:
            action_t = torch.FloatTensor([ca]).to(self._device)
            q1_quantiles, q2_quantiles = self._critic(state_t, action_t)
            quantiles = torch.min(q1_quantiles, q2_quantiles).squeeze()

            if self._risk_sensitivity < 0:
                # Risk-averse: weight toward lower quantiles (CVaR-like)
                k = max(1, int(self._n_quantiles * 0.2))
                score = float(quantiles[:k].mean().item())
            else:
                # Risk-seeking: weight toward upper quantiles
                k = max(1, int(self._n_quantiles * 0.2))
                score = float(quantiles[-k:].mean().item())

            # Blend with mean based on sensitivity magnitude
            mean_q = float(quantiles.mean().item())
            rs = abs(self._risk_sensitivity)
            score = rs * score + (1 - rs) * mean_q

            if score > best_score:
                best_score = score
                best_action = ca

        return float(best_action)

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
            weights_t = torch.FloatTensor(weights).unsqueeze(-1).to(self._device)
        else:
            batch = self._buffer.sample_arrays(self._batch_size)
            weights_t = torch.ones(self._batch_size, 1, device=self._device)
            indices = None

        states = torch.FloatTensor(batch["states"]).to(self._device)
        actions = torch.FloatTensor(batch["actions"]).to(self._device)
        rewards = torch.FloatTensor(batch["rewards"]).unsqueeze(-1).to(self._device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self._device)
        dones = torch.FloatTensor(batch["dones"]).unsqueeze(-1).to(self._device)

        alpha = self._log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self._actor.sample(next_states)
            next_log_prob = next_log_prob.unsqueeze(-1)  # [B] -> [B, 1]
            if self._use_quantile:
                next_q1_mean, next_q2_mean = self._critic_target.q_values(next_states, next_action)
                next_q = torch.min(next_q1_mean, next_q2_mean) - alpha * next_log_prob
            else:
                next_q1, next_q2 = self._critic_target(next_states, next_action)
                next_q = torch.min(next_q1, next_q2) - alpha * next_log_prob
            target_q = rewards + self._gamma * (1 - dones) * next_q

        if self._use_quantile:
            q1_quantiles, q2_quantiles = self._critic(states, actions)
            critic_loss = (
                self._quantile_huber_loss(q1_quantiles, target_q, weights_t) +
                self._quantile_huber_loss(q2_quantiles, target_q, weights_t)
            )
        else:
            q1, q2 = self._critic(states, actions)
            td_error1 = F.mse_loss(q1, target_q, reduction="none")
            td_error2 = F.mse_loss(q2, target_q, reduction="none")
            critic_loss = (weights_t * (td_error1 + td_error2)).mean()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1.0)
        self._critic_optimizer.step()

        # --- Actor update ---
        new_action, log_prob, _ = self._actor.sample(states)
        if self._use_quantile:
            q1_new, q2_new = self._critic.q_values(states, new_action)
        else:
            q1_new, q2_new = self._critic(states, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_new).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 1.0)
        self._actor_optimizer.step()

        # --- Alpha update ---
        alpha_loss = -(self._log_alpha * (log_prob.detach() + self._target_entropy)).mean()
        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()

        # --- Soft target update ---
        self._soft_update()

        # --- Update PER priorities ---
        if self._use_per and indices is not None:
            with torch.no_grad():
                if self._use_quantile:
                    q1_cur, q2_cur = self._critic.q_values(states, actions)
                else:
                    q1_cur, q2_cur = self._critic(states, actions)
                q_cur = torch.min(q1_cur, q2_cur)
                td_np = (q_cur - target_q).squeeze(-1).abs().cpu().numpy()
            self._buffer.update_priorities(indices, td_np)

        # --- Logging ---
        with torch.no_grad():
            if self._use_quantile:
                q1_log, q2_log = self._critic.q_values(states, actions)
            else:
                q1_log, q2_log = self._critic(states, actions)
            mean_q = float(torch.min(q1_log, q2_log).mean().item())
            q_std = float(torch.min(q1_log, q2_log).std().item())

        logger.debug(
            f"SAC train | critic_loss={critic_loss.item():.4f} actor_loss={actor_loss.item():.4f} "
            f"alpha={self.alpha:.4f} mean_q={mean_q:.4f} q_std={q_std:.4f} "
            f"buffer={len(self._buffer)} "
            f"{'mean_priority=' + str(round(self._buffer.mean_priority, 4)) + ' ' if self._use_per else ''}"
        )

        return float(critic_loss.item())

    def _quantile_huber_loss(
        self, quantiles: torch.Tensor, target: torch.Tensor,
        weights: torch.Tensor | None = None, kappa: float = 1.0,
    ) -> torch.Tensor:
        """Quantile Huber loss for distributional critics."""
        n_quantiles = quantiles.shape[-1]
        # target: [B, 1], quantiles: [B, N]
        target_expanded = target.expand(-1, n_quantiles)  # [B, N]
        td_error = target_expanded - quantiles  # [B, N]

        # Quantile midpoints
        tau = torch.arange(n_quantiles, device=quantiles.device, dtype=torch.float32)
        tau = (tau + 0.5) / n_quantiles  # [N]
        tau = tau.unsqueeze(0)  # [1, N]

        huber = torch.where(
            td_error.abs() <= kappa,
            0.5 * td_error.pow(2),
            kappa * (td_error.abs() - 0.5 * kappa),
        )
        element_loss = torch.abs(tau - (td_error < 0).float()) * huber  # [B, N]
        if weights is not None:
            element_loss = weights * element_loss.mean(dim=-1, keepdim=True)
        return element_loss.mean()

    def _soft_update(self) -> None:
        """Polyak-average target network update."""
        for p, tp in zip(self._critic.parameters(), self._critic_target.parameters()):
            tp.data.copy_(self._tau * p.data + (1 - self._tau) * tp.data)

    def end_episode(self, episode_reward: float) -> None:
        self._episode += 1
        self._total_reward += episode_reward

    def save_checkpoint(self) -> bytes | None:
        if not TORCH_AVAILABLE or self._actor is None:
            return None
        buf = io.BytesIO()
        torch.save({
            "version": CHECKPOINT_VERSION,
            "actor": self._actor.state_dict(),
            "critic": self._critic.state_dict(),
            "critic_target": self._critic_target.state_dict(),
            "actor_optimizer": self._actor_optimizer.state_dict(),
            "critic_optimizer": self._critic_optimizer.state_dict(),
            "log_alpha": self._log_alpha.data,
            "alpha_optimizer": self._alpha_optimizer.state_dict(),
            "episode": self._episode,
            "total_reward": self._total_reward,
            "total_steps": self._total_steps,
            "use_quantile": self._use_quantile,
            "risk_sensitivity": self._risk_sensitivity,
        }, buf)
        return buf.getvalue()

    def load_checkpoint(self, data: bytes) -> None:
        if not TORCH_AVAILABLE or self._actor is None:
            return
        buf = io.BytesIO(data)
        checkpoint = torch.load(buf, map_location=self._device, weights_only=False)
        self._actor.load_state_dict(checkpoint["actor"])
        self._critic.load_state_dict(checkpoint["critic"])
        self._critic_target.load_state_dict(checkpoint["critic_target"])
        self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._log_alpha.data = checkpoint["log_alpha"]
        self._alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self._episode = checkpoint["episode"]
        self._total_reward = checkpoint["total_reward"]
        self._total_steps = checkpoint.get("total_steps", 0)
        self._risk_sensitivity = checkpoint.get("risk_sensitivity", 0.0)
        logger.info(f"Loaded SAC checkpoint v{checkpoint.get('version', 0)}: episode={self._episode}")

    def get_stats(self) -> dict[str, Any]:
        return {
            "episode": self._episode,
            "epsilon": round(self.alpha, 4),
            "total_reward": round(self._total_reward, 2),
            "buffer_size": len(self._buffer),
            "training": self._training,
            "torch_available": TORCH_AVAILABLE,
            "agent_type": "sac",
            "alpha": round(self.alpha, 4),
            "use_quantile": self._use_quantile,
            "risk_sensitivity": self._risk_sensitivity,
            "total_steps": self._total_steps,
            "use_per": self._use_per,
            "mean_priority": round(self._buffer.mean_priority, 4) if self._use_per else 0.0,
        }

    def get_risk_metrics(self, state: list[float], action: float) -> dict[str, float]:
        """Get risk metrics from quantile critics (only if use_quantile=True)."""
        if not TORCH_AVAILABLE or not self._use_quantile or self._critic is None:
            return {"cvar_5": 0.0, "var_5": 0.0, "q_mean": 0.0, "q_std": 0.0, "upside": 0.0}

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self._device)
            action_t = torch.FloatTensor([action]).to(self._device)
            q1_quantiles, q2_quantiles = self._critic(state_t, action_t)
            quantiles = torch.min(q1_quantiles, q2_quantiles).squeeze().cpu().numpy()

            q_mean = float(np.mean(quantiles))
            q_std = float(np.std(quantiles))
            var_5 = float(np.percentile(quantiles, 5))
            mask = quantiles <= var_5
            cvar_5 = float(quantiles[mask].mean()) if mask.any() else var_5
            upside = float(np.percentile(quantiles, 75))

        return {
            "cvar_5": cvar_5,
            "var_5": var_5,
            "q_mean": q_mean,
            "q_std": q_std,
            "upside": upside,
        }
