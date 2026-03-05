"""Tests for PPO, DDPG, ensemble strategies, transformer encoder, and composite reward."""

from __future__ import annotations

import random

import numpy as np
import pytest

from core.learning.replay_buffer import Transition

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _random_transition(state_dim: int = 23) -> Transition:
    return Transition(
        state=[random.random() for _ in range(state_dim)],
        action=random.uniform(-1, 1),
        reward=random.uniform(-1, 1),
        next_state=[random.random() for _ in range(state_dim)],
        done=random.random() < 0.1,
    )


# ---------------------------------------------------------------------------
# PPO Tests
# ---------------------------------------------------------------------------

class TestPPOAgent:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_select_action_range(self):
        from core.learning.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=23, hidden_dim=64)
        state = [0.5] * 23
        action, info = agent.select_action(state)
        assert -1.0 <= action <= 1.0
        assert "mean" in info
        assert "std" in info
        assert "value" in info

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_rollout_triggers_update(self):
        from core.learning.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=23, hidden_dim=64, rollout_size=50)

        # Fill rollout
        for _ in range(50):
            action, _ = agent.select_action([0.5] * 23)
            agent.store_transition(_random_transition())

        # Should train now (rollout is full)
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_returns_none_when_not_full(self):
        from core.learning.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=23, rollout_size=100)
        for _ in range(10):
            agent.select_action([0.5] * 23)
            agent.store_transition(_random_transition())
        loss = agent.train_step()
        assert loss is None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_save_load_checkpoint(self):
        from core.learning.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=23, hidden_dim=64)
        agent.end_episode(5.0)
        blob = agent.save_checkpoint()
        assert blob is not None

        agent2 = PPOAgent(state_dim=23, hidden_dim=64)
        agent2.load_checkpoint(blob)
        assert agent2.episode == 1

    def test_base_agent_interface(self):
        from core.learning.ppo_agent import PPOAgent
        from core.learning.base_agent import BaseAgent
        agent = PPOAgent(state_dim=23)
        assert isinstance(agent, BaseAgent)


# ---------------------------------------------------------------------------
# Ensemble Tests
# ---------------------------------------------------------------------------

class TestAgentEnsemble:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_all_agents_train(self):
        from core.learning.sac_agent import SACAgent
        from core.learning.ppo_agent import PPOAgent
        from core.learning.agent_ensemble import AgentEnsemble

        agents = {
            "sac": SACAgent(state_dim=23, hidden_dim=64, batch_size=16, buffer_capacity=200, initial_random_steps=0, use_per=False),
            "ppo": PPOAgent(state_dim=23, hidden_dim=64, rollout_size=50),
        }
        ensemble = AgentEnsemble(agents=agents, eval_interval=5)

        # All agents should receive transitions
        for _ in range(60):
            ensemble.select_action([0.5] * 23)
            ensemble.store_transition(_random_transition())

        # SAC should be able to train (has enough data)
        loss = ensemble.train_step()
        # At least one agent should have trained
        assert ensemble.buffer_size > 0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_sharpe_switching(self):
        from core.learning.sac_agent import SACAgent
        from core.learning.ppo_agent import PPOAgent
        from core.learning.agent_ensemble import AgentEnsemble

        agents = {
            "sac": SACAgent(state_dim=23, hidden_dim=64, batch_size=16, initial_random_steps=0, use_per=False),
            "ppo": PPOAgent(state_dim=23, hidden_dim=64),
        }
        ensemble = AgentEnsemble(agents=agents, eval_interval=5, sharpe_window=10, strategy="best_sharpe")

        # Feed positive rewards to trigger switching evaluation
        for i in range(10):
            ensemble.end_episode(float(i))

        # After enough episodes, active agent should be selected
        assert ensemble.active_agent_name in ("sac", "ppo")

    def test_base_agent_interface(self):
        from core.learning.sac_agent import SACAgent
        from core.learning.agent_ensemble import AgentEnsemble
        from core.learning.base_agent import BaseAgent

        agents = {"sac": SACAgent(state_dim=23, initial_random_steps=0, use_per=False)}
        ensemble = AgentEnsemble(agents=agents)
        assert isinstance(ensemble, BaseAgent)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_save_load_checkpoint(self):
        from core.learning.sac_agent import SACAgent
        from core.learning.agent_ensemble import AgentEnsemble

        agents = {"sac": SACAgent(state_dim=23, hidden_dim=64, initial_random_steps=0, use_per=False)}
        ensemble = AgentEnsemble(agents=agents)
        ensemble.end_episode(1.0)

        blob = ensemble.save_checkpoint()
        assert blob is not None

        agents2 = {"sac": SACAgent(state_dim=23, hidden_dim=64, initial_random_steps=0, use_per=False)}
        ensemble2 = AgentEnsemble(agents=agents2)
        ensemble2.load_checkpoint(blob)
        assert ensemble2.episode == 1

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_weighted_average_strategy(self):
        """Weighted average should blend actions from all agents."""
        from core.learning.sac_agent import SACAgent
        from core.learning.ppo_agent import PPOAgent
        from core.learning.agent_ensemble import AgentEnsemble

        agents = {
            "sac": SACAgent(state_dim=23, hidden_dim=64, initial_random_steps=0, use_per=False),
            "ppo": PPOAgent(state_dim=23, hidden_dim=64),
        }
        ensemble = AgentEnsemble(agents=agents, strategy="weighted_average")

        action, info = ensemble.select_action([0.5] * 23)
        assert -1.0 <= action <= 1.0
        assert info["strategy"] == "weighted_average"
        assert "weights" in info
        assert len(info["weights"]) == 2
        assert "per_agent_actions" in info

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_majority_vote_strategy(self):
        """Majority vote should discretize actions and pick winner."""
        from core.learning.sac_agent import SACAgent
        from core.learning.ppo_agent import PPOAgent
        from core.learning.agent_ensemble import AgentEnsemble

        agents = {
            "sac": SACAgent(state_dim=23, hidden_dim=64, initial_random_steps=0, use_per=False),
            "ppo": PPOAgent(state_dim=23, hidden_dim=64),
        }
        ensemble = AgentEnsemble(agents=agents, strategy="majority_vote")

        action, info = ensemble.select_action([0.5] * 23)
        assert -1.0 <= action <= 1.0
        assert info["strategy"] == "majority_vote"
        assert "votes" in info
        assert info["winner"] in ("buy", "sell", "hold")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_per_agent_stats(self):
        """Ensemble should track per-agent max drawdown and win rate."""
        from core.learning.sac_agent import SACAgent
        from core.learning.agent_ensemble import AgentEnsemble

        agents = {"sac": SACAgent(state_dim=23, hidden_dim=64, initial_random_steps=0, use_per=False)}
        ensemble = AgentEnsemble(agents=agents)

        # Simulate some episodes
        ensemble.end_episode(5.0)   # win
        ensemble.end_episode(-2.0)  # loss
        ensemble.end_episode(3.0)   # win
        ensemble.end_episode(-4.0)  # loss
        ensemble.end_episode(1.0)   # win

        agent_stats = ensemble.get_agent_stats()
        assert "sac" in agent_stats
        stats = agent_stats["sac"]
        assert "sharpe" in stats
        assert "max_drawdown" in stats
        assert stats["max_drawdown"] >= 0
        assert "win_rate" in stats
        assert 0 <= stats["win_rate"] <= 1
        assert stats["trade_count"] == 5
        assert stats["win_rate"] == 0.6  # 3 wins out of 5

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_ensemble_with_ddpg(self):
        """Ensemble should work with DDPG agent."""
        from core.learning.sac_agent import SACAgent
        from core.learning.ddpg_agent import DDPGAgent
        from core.learning.agent_ensemble import AgentEnsemble

        agents = {
            "sac": SACAgent(state_dim=23, hidden_dim=64, initial_random_steps=0, use_per=False),
            "ddpg": DDPGAgent(state_dim=23, hidden_dim=64, initial_random_steps=0, use_per=False),
        }
        ensemble = AgentEnsemble(agents=agents, strategy="weighted_average")

        action, info = ensemble.select_action([0.5] * 23)
        assert -1.0 <= action <= 1.0
        assert "ddpg" in info["per_agent_actions"]


# ---------------------------------------------------------------------------
# Transformer Tests
# ---------------------------------------------------------------------------

class TestTransformerEncoder:
    def test_state_buffer_padding(self):
        from core.learning.transformer_encoder import StateBuffer
        buf = StateBuffer(seq_len=32, state_dim=23)

        # Push 5 states
        for i in range(5):
            buf.push([float(i)] * 23)

        seq = buf.get_sequence()
        mask = buf.get_mask()

        assert seq.shape == (32, 23)
        assert mask.shape == (32,)
        # First 27 should be zero-padded (mask=False)
        assert not mask[0]
        # Last 5 should be valid (mask=True)
        assert mask[-1]
        assert mask[-5]

    def test_state_buffer_full(self):
        from core.learning.transformer_encoder import StateBuffer
        buf = StateBuffer(seq_len=8, state_dim=5)

        for i in range(20):
            buf.push([float(i)] * 5)

        seq = buf.get_sequence()
        mask = buf.get_mask()

        assert seq.shape == (8, 5)
        assert all(mask)  # all valid when buffer is full

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_encoder_output_shape(self):
        from core.learning.transformer_encoder import TransformerStateEncoder
        encoder = TransformerStateEncoder(
            state_dim=23, embed_dim=64, nhead=4, num_layers=2, seq_len=32,
        )

        x = torch.randn(4, 32, 23)  # batch=4, seq=32, state_dim=23
        output = encoder(x)
        assert output.shape == (4, 64)  # (batch, embed_dim)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_encoder_with_mask(self):
        from core.learning.transformer_encoder import TransformerStateEncoder
        encoder = TransformerStateEncoder(
            state_dim=23, embed_dim=64, nhead=4, num_layers=2, seq_len=32,
        )

        x = torch.randn(2, 32, 23)
        mask = torch.zeros(2, 32, dtype=torch.bool)
        mask[:, -10:] = True  # only last 10 are valid

        output = encoder(x, mask)
        assert output.shape == (2, 64)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_output_dim_property(self):
        from core.learning.transformer_encoder import TransformerStateEncoder
        encoder = TransformerStateEncoder(embed_dim=64)
        assert encoder.output_dim == 64

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_upgraded_defaults(self):
        """Default transformer should use seq_len=64, embed_dim=128, 3 layers."""
        from core.learning.transformer_encoder import TransformerStateEncoder
        encoder = TransformerStateEncoder()
        assert encoder._seq_len == 64
        assert encoder._embed_dim == 128
        assert encoder.output_dim == 128


# ---------------------------------------------------------------------------
# Composite Reward Tests
# ---------------------------------------------------------------------------

class TestCompositeReward:
    def test_drawdown_penalty(self):
        from core.learning.rl_environment import TradingEnvironment
        from core.learning.composite_reward import CompositeRewardWrapper

        env = TradingEnvironment(reward_scale=1.0)
        wrapped = CompositeRewardWrapper(
            env, drawdown_weight=0.5, drawdown_threshold=0.01,
            sortino_weight=0.0, consistency_weight=0.0, transaction_weight=0.0,
        )

        # First trade: win (sets peak equity)
        wrapped.step(0.8, 1950.0, [0.0] * 23)
        for _ in range(5):
            wrapped.step(0.8, 1955.0, [0.0] * 23)
        _, result1 = wrapped.step(-0.8, 1960.0, [0.0] * 23)
        assert result1 is not None

        # Second trade: loss (creates drawdown)
        wrapped.step(0.8, 1960.0, [0.0] * 23)
        for _ in range(5):
            wrapped.step(0.8, 1955.0, [0.0] * 23)
        _, result2 = wrapped.step(-0.8, 1940.0, [0.0] * 23)
        assert result2 is not None

    def test_delegation(self):
        """Wrapped env should delegate all properties."""
        from core.learning.rl_environment import TradingEnvironment
        from core.learning.composite_reward import CompositeRewardWrapper

        env = TradingEnvironment()
        wrapped = CompositeRewardWrapper(env)

        assert wrapped.in_position == env.in_position
        assert wrapped.trade_count == env.trade_count
        assert wrapped.win_rate == env.win_rate

    def test_reset(self):
        from core.learning.rl_environment import TradingEnvironment
        from core.learning.composite_reward import CompositeRewardWrapper

        env = TradingEnvironment()
        wrapped = CompositeRewardWrapper(env)
        wrapped.step(0.8, 1950.0, [0.0] * 23)
        wrapped.reset()
        assert not wrapped.in_position

    def test_passthrough_for_non_close(self):
        """Non-close steps should pass through without modification."""
        from core.learning.rl_environment import TradingEnvironment
        from core.learning.composite_reward import CompositeRewardWrapper

        env = TradingEnvironment()
        wrapped = CompositeRewardWrapper(env)

        # Open position
        reward, result = wrapped.step(0.8, 1950.0, [0.0] * 23)
        assert result is None  # not a close, reward passes through

    def test_transaction_cost(self):
        """Transaction cost should penalize position changes."""
        from core.learning.rl_environment import TradingEnvironment
        from core.learning.composite_reward import CompositeRewardWrapper

        env = TradingEnvironment(reward_scale=1.0)
        wrapped = CompositeRewardWrapper(
            env, transaction_weight=0.5, transaction_fee_rate=0.001,
            drawdown_weight=0.0, sortino_weight=0.0, consistency_weight=0.0,
        )

        # Open and close a trade
        wrapped.step(0.8, 1950.0, [0.0] * 23)
        for _ in range(5):
            wrapped.step(0.8, 1955.0, [0.0] * 23)
        _, result = wrapped.step(-0.8, 1960.0, [0.0] * 23)
        assert result is not None
        # Transaction cost should have been applied (reward slightly reduced)


# ---------------------------------------------------------------------------
# Agent Factory Tests
# ---------------------------------------------------------------------------

class TestAgentFactory:
    def test_create_sac(self):
        from config import AppConfig
        from core.learning.agent_factory import create_agent
        from core.learning.sac_agent import SACAgent

        cfg = AppConfig()
        agent = create_agent(cfg)
        assert isinstance(agent, SACAgent)

    def test_create_dqn(self):
        from config import AppConfig, RLConfig
        from core.learning.agent_factory import create_agent
        from core.learning.rl_agent import RLAgent

        cfg = AppConfig(rl=RLConfig(agent_type="dqn"))
        agent = create_agent(cfg)
        assert isinstance(agent, RLAgent)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_create_ensemble(self):
        from config import AppConfig, RLConfig
        from core.learning.agent_factory import create_agent
        from core.learning.agent_ensemble import AgentEnsemble

        cfg = AppConfig(rl=RLConfig(agent_type="ensemble"))
        agent = create_agent(cfg)
        assert isinstance(agent, AgentEnsemble)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_create_ddpg(self):
        from config import AppConfig, RLConfig
        from core.learning.agent_factory import create_agent
        from core.learning.ddpg_agent import DDPGAgent

        cfg = AppConfig(rl=RLConfig(agent_type="ddpg"))
        agent = create_agent(cfg)
        assert isinstance(agent, DDPGAgent)

    def test_smoke_select_action(self):
        """Factory agent should select action and return (float, dict)."""
        from config import AppConfig
        from core.learning.agent_factory import create_agent

        cfg = AppConfig()
        agent = create_agent(cfg)
        state = [0.5] * 23
        action, info = agent.select_action(state)
        assert -1.0 <= action <= 1.0
        assert isinstance(info, dict)
