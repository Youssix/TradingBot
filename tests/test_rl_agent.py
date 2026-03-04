"""Tests for RL agent, environment, and replay buffer."""

from __future__ import annotations

import numpy as np
import pytest

from core.learning.replay_buffer import ReplayBuffer, Transition
from core.learning.rl_environment import TradingEnvironment, Action, TradeResult
from core.learning.rl_agent import RLAgent, TORCH_AVAILABLE


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(Transition(
                state=[float(i)] * 5,
                action=i % 3,
                reward=float(i),
                next_state=[float(i + 1)] * 5,
            ))
        assert len(buf) == 10

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(Transition(
                state=[0.0], action=0, reward=0.0, next_state=[0.0],
            ))
        assert len(buf) == 5

    def test_sample_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(Transition(
                state=[float(i)], action=0, reward=1.0, next_state=[float(i)],
            ))
        batch = buf.sample(5)
        assert len(batch) == 5

    def test_sample_arrays(self):
        buf = ReplayBuffer(capacity=100)
        state_dim = 23
        for i in range(50):
            buf.push(Transition(
                state=[float(i)] * state_dim,
                action=i % 3,
                reward=float(i) * 0.1,
                next_state=[float(i + 1)] * state_dim,
                done=(i == 49),
            ))
        arrays = buf.sample_arrays(16)
        assert arrays["states"].shape == (16, state_dim)
        assert arrays["actions"].shape == (16,)
        assert arrays["rewards"].shape == (16,)
        assert arrays["next_states"].shape == (16, state_dim)
        assert arrays["dones"].shape == (16,)

    def test_clear(self):
        buf = ReplayBuffer(capacity=10)
        buf.push(Transition(state=[1.0], action=0, reward=0.0, next_state=[1.0]))
        buf.clear()
        assert len(buf) == 0

    def test_to_list(self):
        buf = ReplayBuffer(capacity=10)
        buf.push(Transition(state=[1.0], action=1, reward=0.5, next_state=[2.0], done=True))
        data = buf.to_list()
        assert len(data) == 1
        assert data[0]["action"] == 1
        assert data[0]["done"] is True


class TestTradingEnvironment:
    def test_initial_state(self):
        env = TradingEnvironment()
        assert not env.in_position
        assert env.total_reward == 0.0
        assert env.trade_count == 0

    def test_buy_and_close(self):
        env = TradingEnvironment(reward_scale=1.0, penalty_scale=1.5)
        # Open BUY at 1950
        reward, result = env.step(Action.BUY, 1950.0, [0.0] * 23)
        assert env.in_position
        assert result is None

        # Hold for a few bars
        for _ in range(5):
            env.step(Action.BUY, 1951.0, [0.0] * 23)

        # Close with opposite signal (SELL)
        reward, result = env.step(Action.SELL, 1955.0, [0.0] * 23)
        assert result is not None
        assert result.pnl > 0
        assert result.reward > 0
        assert not env.in_position

    def test_sell_and_close(self):
        env = TradingEnvironment()
        env.step(Action.SELL, 1960.0, [0.0] * 23)
        assert env.in_position

        for _ in range(5):
            env.step(Action.SELL, 1958.0, [0.0] * 23)

        reward, result = env.step(Action.BUY, 1950.0, [0.0] * 23)
        assert result is not None
        assert result.pnl > 0

    def test_losing_trade(self):
        env = TradingEnvironment()
        env.step(Action.BUY, 1950.0, [0.0] * 23)
        for _ in range(5):
            env.step(Action.BUY, 1948.0, [0.0] * 23)
        reward, result = env.step(Action.SELL, 1940.0, [0.0] * 23)
        assert result is not None
        assert result.pnl < 0
        assert result.reward < 0

    def test_hold_penalty(self):
        env = TradingEnvironment(hold_penalty=-0.05)
        reward, _ = env.step(Action.HOLD, 1950.0, [0.0] * 23)
        assert reward == -0.05

    def test_max_hold_force_close(self):
        env = TradingEnvironment(max_hold_bars=5)
        env.step(Action.BUY, 1950.0, [0.0] * 23)
        result = None
        # HOLD action triggers force-close after max_hold_bars
        for i in range(10):
            _, result = env.step(Action.HOLD, 1950.0 + i * 0.1, [0.0] * 23)
            if result:
                break
        assert result is not None  # Should have been force-closed

    def test_reset(self):
        env = TradingEnvironment()
        env.step(Action.BUY, 1950.0, [0.0] * 23)
        env.reset()
        assert not env.in_position

    def test_win_rate(self):
        env = TradingEnvironment()
        # Win
        env.step(Action.BUY, 1950.0, [0.0] * 23)
        for _ in range(5):
            env.step(Action.BUY, 1955.0, [0.0] * 23)
        env.step(Action.SELL, 1960.0, [0.0] * 23)
        # Loss
        env.step(Action.BUY, 1960.0, [0.0] * 23)
        for _ in range(5):
            env.step(Action.BUY, 1955.0, [0.0] * 23)
        env.step(Action.SELL, 1950.0, [0.0] * 23)
        assert env.win_rate == 0.5

    def test_get_state_dict(self):
        env = TradingEnvironment()
        state = env.get_state_dict()
        assert "in_position" in state
        assert "trade_count" in state


class TestRLAgent:
    def test_select_action_shape(self):
        agent = RLAgent(state_dim=23, action_dim=3)
        state = [0.5] * 23
        action, q_values = agent.select_action(state)
        assert 0 <= action <= 2
        assert len(q_values) == 3

    def test_store_transition(self):
        agent = RLAgent(state_dim=5, action_dim=3, buffer_capacity=100)
        t = Transition(state=[0.0] * 5, action=1, reward=0.5, next_state=[0.0] * 5)
        agent.store_transition(t)
        assert agent.buffer_size == 1

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_step_insufficient_data(self):
        agent = RLAgent(state_dim=5, action_dim=3, batch_size=32)
        # Only 5 transitions — not enough for batch_size=32
        for i in range(5):
            agent.store_transition(Transition(
                state=[float(i)] * 5, action=0, reward=0.1, next_state=[0.0] * 5,
            ))
        loss = agent.train_step()
        assert loss is None  # Not enough data

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_train_step_runs(self):
        agent = RLAgent(state_dim=5, action_dim=3, batch_size=16)
        for i in range(50):
            agent.store_transition(Transition(
                state=[float(i % 10)] * 5,
                action=i % 3,
                reward=float(i % 5) * 0.1,
                next_state=[float((i + 1) % 10)] * 5,
            ))
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)

    def test_epsilon_decay(self):
        agent = RLAgent(epsilon_start=1.0, epsilon_decay=0.9, epsilon_end=0.01)
        initial = agent.epsilon
        agent.end_episode(1.0)
        assert agent.epsilon < initial

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_save_load_checkpoint(self):
        agent = RLAgent(state_dim=5, action_dim=3)
        agent.end_episode(10.0)
        blob = agent.save_checkpoint()
        assert blob is not None

        agent2 = RLAgent(state_dim=5, action_dim=3)
        agent2.load_checkpoint(blob)
        assert agent2.episode == agent.episode
        assert abs(agent2.epsilon - agent.epsilon) < 1e-6

    def test_get_stats(self):
        agent = RLAgent()
        stats = agent.get_stats()
        assert "episode" in stats
        assert "epsilon" in stats
        assert "total_reward" in stats
        assert "torch_available" in stats

    def test_training_toggle(self):
        agent = RLAgent()
        assert agent.training is True
        agent.training = False
        assert agent.training is False
