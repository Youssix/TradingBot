"""Tests for RL backtester — training on historical data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.learning.feature_engine import FeatureEngine
from core.learning.regime_detector import RegimeDetector
from core.learning.rl_agent import RLAgent, TORCH_AVAILABLE
from core.learning.rl_backtester import (
    RLBacktestConfig,
    RLBacktester,
    RLBacktestResult,
    RLTradeRecord,
)
from core.learning.rl_environment import TradingEnvironment


def _make_ohlcv(n: int = 500, base_price: float = 1950.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a trend and noise."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    trend = np.linspace(0, 50, n)
    noise = np.cumsum(np.random.randn(n) * 2)
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n)) * 3
    low = close - np.abs(np.random.randn(n)) * 3
    open_ = close + np.random.randn(n) * 1.5
    volume = np.random.randint(100, 5000, n)

    return pd.DataFrame({
        "datetime": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestRLBacktestConfig:
    def test_defaults(self):
        cfg = RLBacktestConfig()
        assert cfg.timeframe == "H1"
        assert cfg.count == 2000
        assert cfg.epochs == 3
        assert cfg.train_every == 5

    def test_custom(self):
        cfg = RLBacktestConfig(timeframe="M5", count=1000, epochs=5, train_every=10)
        assert cfg.timeframe == "M5"
        assert cfg.count == 1000
        assert cfg.epochs == 5


class TestRLBacktester:
    def test_run_returns_result(self):
        """Backtester should complete and return a valid result."""
        data = _make_ohlcv(200)
        config = RLBacktestConfig(epochs=1, train_every=10)
        backtester = RLBacktester(config=config)
        result = backtester.run(data)

        assert isinstance(result, RLBacktestResult)
        assert result.epochs_completed == 1
        assert result.total_trades >= 0

    def test_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        data = _make_ohlcv(10)  # way too few
        config = RLBacktestConfig(epochs=1)
        backtester = RLBacktester(config=config)
        result = backtester.run(data)

        assert result.total_trades == 0
        assert result.epochs_completed == 0

    def test_multiple_epochs_produce_more_trades(self):
        """Running more epochs should produce at least as many trades."""
        data = _make_ohlcv(300)

        config1 = RLBacktestConfig(epochs=1, train_every=10)
        bt1 = RLBacktester(config=config1)
        r1 = bt1.run(data)

        config2 = RLBacktestConfig(epochs=3, train_every=10)
        bt2 = RLBacktester(config=config2)
        r2 = bt2.run(data)

        # 3 epochs should produce >= 1 epoch worth of trades
        assert r2.total_trades >= r1.total_trades

    def test_epsilon_decays(self):
        """Epsilon should be lower after training than at start."""
        data = _make_ohlcv(300)
        config = RLBacktestConfig(epochs=2, train_every=5)
        backtester = RLBacktester(config=config)
        result = backtester.run(data)

        # If there were any trades (episodes), epsilon should have decayed
        if result.episodes > 0:
            assert result.final_epsilon < 1.0

    def test_equity_curve_tracks_trades(self):
        """Equity curve should have one point per trade."""
        data = _make_ohlcv(300)
        config = RLBacktestConfig(epochs=2, train_every=5)
        backtester = RLBacktester(config=config)
        result = backtester.run(data)

        assert len(result.equity_curve) == result.total_trades

    def test_epoch_stats_reported(self):
        """Should have stats for each epoch."""
        data = _make_ohlcv(200)
        config = RLBacktestConfig(epochs=3, train_every=10)
        backtester = RLBacktester(config=config)
        result = backtester.run(data)

        assert len(result.epoch_stats) == 3
        for s in result.epoch_stats:
            assert "epoch" in s
            assert "trades" in s
            assert "win_rate" in s

    def test_trade_records_have_all_fields(self):
        """Each trade record should have the expected fields."""
        data = _make_ohlcv(300)
        config = RLBacktestConfig(epochs=1, train_every=5)
        backtester = RLBacktester(config=config)
        result = backtester.run(data)

        if result.trades:
            t = result.trades[0]
            assert isinstance(t, RLTradeRecord)
            assert t.action in ("BUY", "SELL")
            assert t.entry_price > 0
            assert t.exit_price > 0
            assert t.hold_bars >= 0

    def test_custom_components(self):
        """Backtester should accept custom RL components."""
        data = _make_ohlcv(200)
        agent = RLAgent(state_dim=23, action_dim=3, epsilon_decay=0.99)
        env = TradingEnvironment(reward_scale=2.0)
        fe = FeatureEngine(primary_tf="H1")
        rd = RegimeDetector(primary_tf="H1")

        config = RLBacktestConfig(epochs=1, train_every=10)
        backtester = RLBacktester(
            config=config, agent=agent, env=env,
            feature_engine=fe, regime_detector=rd,
        )
        result = backtester.run(data)
        assert isinstance(result, RLBacktestResult)

    def test_agent_accessible(self):
        """Agent should be accessible after run for checkpoint saving."""
        data = _make_ohlcv(200)
        config = RLBacktestConfig(epochs=1, train_every=10)
        backtester = RLBacktester(config=config)
        backtester.run(data)

        agent = backtester.agent
        assert agent is not None
        stats = agent.get_stats()
        assert "episode" in stats

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_training_loss_decreases_or_runs(self):
        """When torch is available, training should actually run."""
        data = _make_ohlcv(500)
        config = RLBacktestConfig(epochs=2, train_every=3)
        backtester = RLBacktester(config=config)
        result = backtester.run(data)

        # At least some epochs should have non-zero avg_train_loss
        losses = [s["avg_train_loss"] for s in result.epoch_stats]
        if result.total_trades > 10:
            assert any(l > 0 for l in losses)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing equity has 0 drawdown."""
        curve = [
            {"trade_index": i, "cumulative_pnl": float(i)}
            for i in range(1, 6)
        ]
        dd = RLBacktester._compute_max_drawdown(curve)
        assert dd == 0.0

    def test_known_drawdown(self):
        """Known drawdown should be computed correctly."""
        curve = [
            {"trade_index": 1, "cumulative_pnl": 10.0},
            {"trade_index": 2, "cumulative_pnl": 5.0},   # dd = 5
            {"trade_index": 3, "cumulative_pnl": 15.0},
            {"trade_index": 4, "cumulative_pnl": 8.0},   # dd = 7
        ]
        dd = RLBacktester._compute_max_drawdown(curve)
        assert dd == 7.0

    def test_empty_curve(self):
        dd = RLBacktester._compute_max_drawdown([])
        assert dd == 0.0
