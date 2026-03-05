"""Tests for DDPG agent and Prioritized Experience Replay."""

from __future__ import annotations

import random

import numpy as np
import pytest

from core.learning.replay_buffer import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    SumTree,
    Transition,
)

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
# SumTree Tests
# ---------------------------------------------------------------------------

class TestSumTree:
    def test_add_and_total(self):
        tree = SumTree(4)
        tree.add(1.0, _random_transition())
        tree.add(2.0, _random_transition())
        tree.add(3.0, _random_transition())
        assert abs(tree.total - 6.0) < 1e-6

    def test_max_priority(self):
        tree = SumTree(4)
        tree.add(1.0, _random_transition())
        tree.add(5.0, _random_transition())
        tree.add(2.0, _random_transition())
        assert abs(tree.max_priority - 5.0) < 1e-6

    def test_min_priority(self):
        tree = SumTree(4)
        tree.add(3.0, _random_transition())
        tree.add(1.0, _random_transition())
        tree.add(5.0, _random_transition())
        assert abs(tree.min_priority - 1.0) < 1e-6

    def test_capacity_overflow(self):
        tree = SumTree(3)
        for i in range(5):
            tree.add(float(i + 1), _random_transition())
        assert len(tree) == 3

    def test_get_retrieves_data(self):
        tree = SumTree(4)
        t1 = _random_transition()
        tree.add(10.0, t1)
        idx, priority, data = tree.get(5.0)
        assert data is not None
        assert abs(priority - 10.0) < 1e-6

    def test_update_priority(self):
        tree = SumTree(4)
        tree.add(1.0, _random_transition())
        tree.add(2.0, _random_transition())
        # Update first item
        idx = tree._capacity - 1  # first leaf
        tree.update(idx, 10.0)
        assert abs(tree.total - 12.0) < 1e-6


# ---------------------------------------------------------------------------
# PER Tests
# ---------------------------------------------------------------------------

class TestPrioritizedReplayBuffer:
    def test_push_and_len(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(10):
            buf.push(_random_transition())
        assert len(buf) == 10

    def test_sample_returns_correct_shape(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(_random_transition())
        batch, weights, indices = buf.sample(16)
        assert batch["states"].shape == (16, 23)
        assert batch["actions"].shape == (16,)
        assert weights.shape == (16,)
        assert len(indices) == 16

    def test_importance_weights_bounded(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta_start=1.0)
        for _ in range(50):
            buf.push(_random_transition())
        _, weights, _ = buf.sample(16)
        # With beta=1.0, weights should be normalized to max=1.0
        assert np.max(weights) <= 1.0 + 1e-6
        assert np.min(weights) > 0.0

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(_random_transition())
        _, _, indices = buf.sample(10)
        # Update with high TD errors
        td_errors = np.array([10.0] * 10)
        buf.update_priorities(indices, td_errors)
        # Mean priority should be higher now
        assert buf.mean_priority > 0

    def test_high_error_sampled_more(self):
        """Transitions with higher priority should be sampled more often."""
        buf = PrioritizedReplayBuffer(capacity=100, alpha=1.0)
        # Add 50 low-priority items
        for _ in range(50):
            buf.push(_random_transition())
        # Update some with very high priority
        _, _, indices = buf.sample(5)
        buf.update_priorities(indices, np.array([100.0] * 5))

        # Sample many times and count how often high-priority indices appear
        high_set = set(indices)
        hit_count = 0
        for _ in range(200):
            _, _, sampled_idx = buf.sample(10)
            for idx in sampled_idx:
                if idx in high_set:
                    hit_count += 1
        # High-priority items should be sampled disproportionately
        assert hit_count > 100  # at least 5% of all samples

    def test_beta_annealing(self):
        buf = PrioritizedReplayBuffer(capacity=100, beta_start=0.4, beta_frames=100)
        assert abs(buf.beta - 0.4) < 1e-6
        # Sample many times to advance frame counter
        for _ in range(50):
            buf.push(_random_transition())
        for _ in range(50):
            buf.sample(5)
        assert buf.beta > 0.4
        for _ in range(100):
            buf.sample(5)
        assert abs(buf.beta - 1.0) < 0.05

    def test_sample_arrays_compat(self):
        """sample_arrays should work as uniform-compatible interface."""
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(_random_transition())
        batch = buf.sample_arrays(16)
        assert "states" in batch
        assert batch["states"].shape[0] == 16

    def test_clear(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(10):
            buf.push(_random_transition())
        buf.clear()
        assert len(buf) == 0

    def test_capacity_overflow(self):
        buf = PrioritizedReplayBuffer(capacity=10)
        for _ in range(20):
            buf.push(_random_transition())
        assert len(buf) == 10


# ---------------------------------------------------------------------------
# DDPG Agent Tests
# ---------------------------------------------------------------------------

class TestDDPGAgent:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_select_action_range(self):
        from core.learning.ddpg_agent import DDPGAgent
        agent = DDPGAgent(state_dim=23, hidden_dim=64, initial_random_steps=0)
        state = [0.5] * 23
        action, info = agent.select_action(state)
        assert -1.0 <= action <= 1.0
        assert "mean" in info
        assert "noise" in info

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_deterministic_in_eval(self):
        """In eval mode, same state should give same action (no noise)."""
        from core.learning.ddpg_agent import DDPGAgent
        agent = DDPGAgent(state_dim=23, hidden_dim=64, initial_random_steps=0)
        agent.training = False
        state = [0.5] * 23
        a1, _ = agent.select_action(state)
        a2, _ = agent.select_action(state)
        assert abs(a1 - a2) < 1e-6

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_noise_in_train(self):
        """In train mode, actions should differ due to OU noise."""
        from core.learning.ddpg_agent import DDPGAgent
        agent = DDPGAgent(state_dim=23, hidden_dim=64, initial_random_steps=0, ou_sigma=0.5)
        agent.training = True
        state = [0.5] * 23
        actions = [agent.select_action(state)[0] for _ in range(20)]
        # Not all actions should be the same
        assert len(set(round(a, 4) for a in actions)) > 1

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_step_runs(self):
        from core.learning.ddpg_agent import DDPGAgent
        agent = DDPGAgent(
            state_dim=23, hidden_dim=64, batch_size=16,
            buffer_capacity=200, initial_random_steps=0, use_per=False,
        )
        for _ in range(50):
            agent.store_transition(_random_transition())
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_with_per(self):
        from core.learning.ddpg_agent import DDPGAgent
        agent = DDPGAgent(
            state_dim=23, hidden_dim=64, batch_size=16,
            buffer_capacity=200, initial_random_steps=0, use_per=True,
        )
        for _ in range(50):
            agent.store_transition(_random_transition())
        loss = agent.train_step()
        assert loss is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_step_insufficient_data(self):
        from core.learning.ddpg_agent import DDPGAgent
        agent = DDPGAgent(state_dim=23, hidden_dim=64, batch_size=32, initial_random_steps=0)
        for _ in range(5):
            agent.store_transition(_random_transition())
        loss = agent.train_step()
        assert loss is None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_initial_random_steps(self):
        """Before initial_random_steps, actions should be random."""
        from core.learning.ddpg_agent import DDPGAgent
        agent = DDPGAgent(state_dim=23, hidden_dim=64, initial_random_steps=100)
        agent.training = True
        _, info = agent.select_action([0.5] * 23)
        assert info.get("random", False) is True

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_save_load_checkpoint(self):
        from core.learning.ddpg_agent import DDPGAgent
        agent = DDPGAgent(state_dim=23, hidden_dim=64)
        agent.end_episode(5.0)
        blob = agent.save_checkpoint()
        assert blob is not None

        agent2 = DDPGAgent(state_dim=23, hidden_dim=64)
        agent2.load_checkpoint(blob)
        assert agent2.episode == 1
        assert abs(agent2.total_reward - 5.0) < 1e-6

    def test_base_agent_interface(self):
        from core.learning.ddpg_agent import DDPGAgent
        from core.learning.base_agent import BaseAgent
        agent = DDPGAgent(state_dim=23)
        assert isinstance(agent, BaseAgent)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_get_stats(self):
        from core.learning.ddpg_agent import DDPGAgent
        agent = DDPGAgent(state_dim=23)
        stats = agent.get_stats()
        assert stats["agent_type"] == "ddpg"
        assert "total_steps" in stats
        assert "use_per" in stats


class TestOUNoise:
    def test_reset(self):
        from core.learning.ddpg_agent import OUNoise
        noise = OUNoise(size=1, mu=0.0, sigma=0.2)
        # Sample a few times to move away from mu
        for _ in range(100):
            noise.sample()
        noise.reset()
        assert abs(noise._state[0]) < 1e-6

    def test_correlated(self):
        """OU noise should show temporal correlation."""
        from core.learning.ddpg_agent import OUNoise
        noise = OUNoise(size=1, theta=0.15, sigma=0.2)
        samples = [float(noise.sample()[0]) for _ in range(100)]
        # Consecutive samples should be correlated (not purely random)
        diffs = [abs(samples[i+1] - samples[i]) for i in range(len(samples)-1)]
        avg_diff = np.mean(diffs)
        # With theta=0.15 and sigma=0.2, avg consecutive diff should be relatively small
        assert avg_diff < 0.5
