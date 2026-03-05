"""Factory: create RL agent from AppConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from core.learning.base_agent import BaseAgent

if TYPE_CHECKING:
    from config import AppConfig


def create_agent(cfg: AppConfig) -> BaseAgent:
    """Create an RL agent based on config.rl.agent_type.

    Supported types: "sac", "ppo", "ddpg", "dqn", "ensemble".
    """
    agent_type = getattr(cfg.rl, "agent_type", "sac")
    state_dim = cfg.rl.state_dim

    # Check if transformer is enabled — use embed_dim as effective state_dim
    transformer_cfg = getattr(cfg, "transformer", None)
    if transformer_cfg and transformer_cfg.enabled:
        state_dim = transformer_cfg.embed_dim

    if agent_type == "sac":
        return _create_sac(cfg, state_dim)
    elif agent_type == "ppo":
        return _create_ppo(cfg, state_dim)
    elif agent_type == "ddpg":
        return _create_ddpg(cfg, state_dim)
    elif agent_type == "dqn":
        return _create_dqn(cfg)
    elif agent_type == "ensemble":
        return _create_ensemble(cfg, state_dim)
    else:
        logger.warning(f"Unknown agent_type '{agent_type}', falling back to SAC")
        return _create_sac(cfg, state_dim)


def _get_per_config(cfg: AppConfig) -> tuple[bool, float, float]:
    """Extract PER config: (use_per, alpha, beta_start)."""
    per_cfg = getattr(cfg, "per", None)
    if per_cfg:
        return per_cfg.enabled, per_cfg.alpha, per_cfg.beta_start
    return True, 0.6, 0.4


def _create_sac(cfg: AppConfig, state_dim: int) -> BaseAgent:
    from core.learning.sac_agent import SACAgent
    sac_cfg = getattr(cfg, "sac", None)
    use_per, per_alpha, per_beta_start = _get_per_config(cfg)

    if sac_cfg:
        return SACAgent(
            state_dim=state_dim,
            hidden_dim=sac_cfg.hidden_dim,
            actor_lr=sac_cfg.actor_lr,
            critic_lr=sac_cfg.critic_lr,
            alpha_lr=sac_cfg.alpha_lr,
            gamma=cfg.rl.gamma,
            tau=sac_cfg.tau,
            target_entropy=sac_cfg.target_entropy,
            buffer_capacity=sac_cfg.buffer_capacity,
            batch_size=sac_cfg.batch_size,
            use_quantile=sac_cfg.use_quantile,
            n_quantiles=sac_cfg.n_quantiles,
            risk_sensitivity=sac_cfg.risk_sensitivity,
            initial_random_steps=sac_cfg.initial_random_steps,
            use_per=use_per,
            per_alpha=per_alpha,
            per_beta_start=per_beta_start,
        )
    else:
        return SACAgent(
            state_dim=state_dim,
            gamma=cfg.rl.gamma,
            buffer_capacity=cfg.rl.buffer_capacity,
            batch_size=cfg.rl.batch_size,
            use_per=use_per,
        )


def _create_ppo(cfg: AppConfig, state_dim: int) -> BaseAgent:
    from core.learning.ppo_agent import PPOAgent
    ppo_cfg = getattr(cfg, "ppo", None)

    if ppo_cfg:
        return PPOAgent(
            state_dim=state_dim,
            hidden_dim=ppo_cfg.hidden_dim,
            lr=ppo_cfg.lr,
            gamma=cfg.rl.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            clip_epsilon=ppo_cfg.clip_epsilon,
            entropy_coeff=ppo_cfg.entropy_coeff,
            value_coeff=ppo_cfg.value_coeff,
            rollout_size=ppo_cfg.rollout_size,
            n_epochs_per_update=ppo_cfg.n_epochs_per_update,
            batch_size=ppo_cfg.batch_size,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )
    else:
        return PPOAgent(
            state_dim=state_dim,
            gamma=cfg.rl.gamma,
        )


def _create_ddpg(cfg: AppConfig, state_dim: int) -> BaseAgent:
    from core.learning.ddpg_agent import DDPGAgent
    ddpg_cfg = getattr(cfg, "ddpg", None)
    use_per, per_alpha, per_beta_start = _get_per_config(cfg)

    if ddpg_cfg:
        return DDPGAgent(
            state_dim=state_dim,
            hidden_dim=ddpg_cfg.hidden_dim,
            actor_lr=ddpg_cfg.actor_lr,
            critic_lr=ddpg_cfg.critic_lr,
            gamma=cfg.rl.gamma,
            tau=ddpg_cfg.tau,
            buffer_capacity=ddpg_cfg.buffer_capacity,
            batch_size=ddpg_cfg.batch_size,
            ou_theta=ddpg_cfg.ou_theta,
            ou_sigma=ddpg_cfg.ou_sigma,
            initial_random_steps=ddpg_cfg.initial_random_steps,
            use_per=use_per,
            per_alpha=per_alpha,
            per_beta_start=per_beta_start,
        )
    else:
        return DDPGAgent(
            state_dim=state_dim,
            gamma=cfg.rl.gamma,
            use_per=use_per,
        )


def _create_dqn(cfg: AppConfig) -> BaseAgent:
    from core.learning.rl_agent import RLAgent
    return RLAgent(
        state_dim=cfg.rl.state_dim,
        lr=cfg.rl.lr,
        gamma=cfg.rl.gamma,
        epsilon_start=cfg.rl.epsilon_start,
        epsilon_end=cfg.rl.epsilon_end,
        epsilon_decay=cfg.rl.epsilon_decay,
        buffer_capacity=cfg.rl.buffer_capacity,
        batch_size=cfg.rl.batch_size,
        target_update_freq=cfg.rl.target_update_freq,
    )


def _create_ensemble(cfg: AppConfig, state_dim: int) -> BaseAgent:
    from core.learning.agent_ensemble import AgentEnsemble
    ensemble_cfg = getattr(cfg, "ensemble_agent", None)

    agent_types = ("sac", "ppo", "ddpg") if not ensemble_cfg else ensemble_cfg.agents
    sharpe_window = 100 if not ensemble_cfg else ensemble_cfg.sharpe_window
    eval_interval = 20 if not ensemble_cfg else ensemble_cfg.eval_interval
    strategy = "weighted_average" if not ensemble_cfg else ensemble_cfg.strategy

    agents: dict[str, BaseAgent] = {}
    for at in agent_types:
        if at == "sac":
            agents["sac"] = _create_sac(cfg, state_dim)
        elif at == "ppo":
            agents["ppo"] = _create_ppo(cfg, state_dim)
        elif at == "ddpg":
            agents["ddpg"] = _create_ddpg(cfg, state_dim)
        elif at == "dqn":
            agents["dqn"] = _create_dqn(cfg)

    if not agents:
        agents["sac"] = _create_sac(cfg, state_dim)

    return AgentEnsemble(
        agents=agents,
        strategy=strategy,
        sharpe_window=sharpe_window,
        eval_interval=eval_interval,
    )
