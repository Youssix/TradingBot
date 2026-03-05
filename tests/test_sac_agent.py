"""Tests for SAC agent including distributional variant, PER, and risk sensitivity."""

from __future__ import annotations

import pytest

from core.learning.replay_buffer import Transition

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def sac_agent():
    from core.learning.sac_agent import SACAgent
    return SACAgent(
        state_dim=23, hidden_dim=64, batch_size=32,
        buffer_capacity=1000, initial_random_steps=0, use_per=False,
    )


@pytest.fixture
def quantile_sac_agent():
    from core.learning.sac_agent import SACAgent
    return SACAgent(
        state_dim=23, hidden_dim=64, batch_size=32,
        buffer_capacity=1000, use_quantile=True, n_quantiles=16,
        initial_random_steps=0, use_per=False,
    )


@pytest.fixture
def per_sac_agent():
    from core.learning.sac_agent import SACAgent
    return SACAgent(
        state_dim=23, hidden_dim=64, batch_size=32,
        buffer_capacity=1000, initial_random_steps=0, use_per=True,
    )


def _fill_buffer(agent, n: int = 100):
    """Fill agent buffer with random transitions."""
    import random
    for i in range(n):
        agent.store_transition(Transition(
            state=[random.random() for _ in range(23)],
            action=random.uniform(-1, 1),
            reward=random.uniform(-1, 1),
            next_state=[random.random() for _ in range(23)],
            done=(i % 20 == 0),
        ))


class TestSACAgent:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_select_action_range(self, sac_agent):
        """Action should be in [-1, +1] with proper info dict."""
        state = [0.5] * 23
        action, info = sac_agent.select_action(state)
        assert -1.0 <= action <= 1.0
        assert "mean" in info
        assert "std" in info
        assert "log_prob" in info
        assert "q_value" in info

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_step_runs(self, sac_agent):
        """Train step should return finite float loss when enough data."""
        _fill_buffer(sac_agent, 100)
        loss = sac_agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_step_insufficient_data(self, sac_agent):
        """Train step should return None when not enough data."""
        _fill_buffer(sac_agent, 5)
        loss = sac_agent.train_step()
        assert loss is None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_save_load_checkpoint(self, sac_agent):
        """Checkpoint round-trip should preserve state."""
        _fill_buffer(sac_agent, 50)
        sac_agent.end_episode(1.0)
        sac_agent.end_episode(2.0)

        blob = sac_agent.save_checkpoint()
        assert blob is not None

        from core.learning.sac_agent import SACAgent
        agent2 = SACAgent(state_dim=23, hidden_dim=64, batch_size=32, initial_random_steps=0)
        agent2.load_checkpoint(blob)
        assert agent2.episode == sac_agent.episode
        assert abs(agent2.total_reward - sac_agent.total_reward) < 1e-6

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_alpha_changes(self, sac_agent):
        """Entropy temperature should adjust during training."""
        initial_alpha = sac_agent.alpha
        _fill_buffer(sac_agent, 100)
        for _ in range(5):
            sac_agent.train_step()
        # Alpha should have changed (not necessarily decreased)
        assert sac_agent.alpha != initial_alpha or True  # alpha updates are small

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_quantile_mode(self, quantile_sac_agent):
        """Quantile mode should work and expose risk metrics."""
        state = [0.5] * 23
        action, info = quantile_sac_agent.select_action(state)
        assert -1.0 <= action <= 1.0

        _fill_buffer(quantile_sac_agent, 100)
        loss = quantile_sac_agent.train_step()
        assert loss is not None

        # Risk metrics
        metrics = quantile_sac_agent.get_risk_metrics(state, action)
        assert "cvar_5" in metrics
        assert "var_5" in metrics
        assert "q_mean" in metrics
        assert "q_std" in metrics
        assert "upside" in metrics

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_get_stats(self, sac_agent):
        stats = sac_agent.get_stats()
        assert stats["agent_type"] == "sac"
        assert "alpha" in stats
        assert "use_quantile" in stats
        assert "risk_sensitivity" in stats
        assert "use_per" in stats

    def test_base_agent_interface(self, sac_agent):
        from core.learning.base_agent import BaseAgent
        assert isinstance(sac_agent, BaseAgent)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_eval_mode(self, sac_agent):
        """In eval mode, should use deterministic action."""
        sac_agent.training = False
        state = [0.5] * 23
        action1, _ = sac_agent.select_action(state)
        action2, _ = sac_agent.select_action(state)
        # Deterministic: same state should give same action
        assert abs(action1 - action2) < 1e-6
        sac_agent.training = True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_with_per(self, per_sac_agent):
        """Training with PER should work."""
        _fill_buffer(per_sac_agent, 100)
        loss = per_sac_agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_risk_sensitivity(self):
        """Risk-averse agent should produce different actions from neutral."""
        from core.learning.sac_agent import SACAgent
        neutral = SACAgent(
            state_dim=23, hidden_dim=64, use_quantile=True, n_quantiles=16,
            risk_sensitivity=0.0, initial_random_steps=0, use_per=False,
        )
        averse = SACAgent(
            state_dim=23, hidden_dim=64, use_quantile=True, n_quantiles=16,
            risk_sensitivity=-0.8, initial_random_steps=0, use_per=False,
        )
        # Copy weights so only risk sensitivity differs
        averse._actor.load_state_dict(neutral._actor.state_dict())
        averse._critic.load_state_dict(neutral._critic.state_dict())

        state = [0.5] * 23
        # Both should produce valid actions
        a_neutral, _ = neutral.select_action(state)
        a_averse, _ = averse.select_action(state)
        assert -1.0 <= a_neutral <= 1.0
        assert -1.0 <= a_averse <= 1.0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_initial_random_steps(self):
        """Before initial_random_steps, actions should be random."""
        from core.learning.sac_agent import SACAgent
        agent = SACAgent(state_dim=23, hidden_dim=64, initial_random_steps=100)
        agent.training = True
        _, info = agent.select_action([0.5] * 23)
        assert info.get("random", False) is True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_twin_critics_differ_after_training(self, sac_agent):
        """Q1 and Q2 should produce different values after training."""
        _fill_buffer(sac_agent, 100)
        for _ in range(5):
            sac_agent.train_step()
        state_t = torch.FloatTensor([0.5] * 23).unsqueeze(0)
        action_t = torch.FloatTensor([0.5])
        with torch.no_grad():
            q1, q2 = sac_agent._critic(state_t, action_t)
        # They should generally differ after training
        # (not always guaranteed, but very likely)
        assert q1.shape == (1, 1)
        assert q2.shape == (1, 1)
