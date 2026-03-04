"""Tests for feature engine, regime detector, ensemble, and performance tracker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest
from datetime import datetime, timedelta

from core.learning.feature_engine import FeatureEngine, CONTEXT_TF_MAP
from core.learning.regime_detector import RegimeDetector, Regime
from core.learning.performance_tracker import PerformanceTracker
from core.learning.confidence_scorer import ConfidenceScorer
from core.learning.ensemble import StrategyEnsemble
from core.learning.regime_detector import RegimeState


def _make_ohlcv(n: int = 200, base: float = 1950.0, trend: float = 0.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n, freq="1min")
    closes = base + np.cumsum(np.random.randn(n) * 0.5 + trend)
    opens = closes + np.random.randn(n) * 0.2
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 0.5
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 0.5
    volumes = np.random.randint(100, 5000, n)
    return pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestFeatureEngine:
    def test_extract_returns_correct_dim(self):
        engine = FeatureEngine(primary_tf="M1")
        df = _make_ohlcv(200)
        result = engine.extract(df)
        assert len(result["vector"]) == FeatureEngine.FEATURE_DIM
        assert len(result["names"]) == FeatureEngine.FEATURE_DIM

    def test_extract_values_in_range(self):
        engine = FeatureEngine(primary_tf="M1")
        df = _make_ohlcv(200)
        result = engine.extract(df)
        for i, v in enumerate(result["vector"]):
            assert -1.5 <= v <= 1.5, f"Feature {i} ({result['names'][i]}) = {v} out of range"

    def test_extract_with_htf_data(self):
        engine = FeatureEngine(primary_tf="M1")
        df = _make_ohlcv(200)
        htf = {"M5": _make_ohlcv(100), "M15": _make_ohlcv(100)}
        result = engine.extract(df, htf)
        assert result["vector"][19] in (0.0, 1.0)  # htf1 binary
        assert result["vector"][20] in (0.0, 1.0)  # htf2 binary

    def test_extract_insufficient_data(self):
        engine = FeatureEngine(primary_tf="M1")
        df = _make_ohlcv(5)
        result = engine.extract(df)
        assert result["vector"] == [0.0] * FeatureEngine.FEATURE_DIM

    def test_context_timeframes(self):
        for tf, expected in CONTEXT_TF_MAP.items():
            engine = FeatureEngine(primary_tf=tf)
            assert engine.context_timeframes == expected

    def test_session_detection(self):
        engine = FeatureEngine(primary_tf="M1")
        df = _make_ohlcv(200)
        # Override last candle time to London session
        df.loc[df.index[-1], "datetime"] = datetime(2024, 1, 15, 10, 0)
        result = engine.extract(df)
        assert result["session"] == "london"


class TestRegimeDetector:
    def test_initial_regime_is_ranging(self):
        detector = RegimeDetector()
        assert detector.current.regime == Regime.RANGING

    def test_trending_detection(self):
        detector = RegimeDetector(primary_tf="M1", trending_threshold=20.0)
        # Feed high ADX features repeatedly
        for _ in range(15):
            detector.update({"adx": 35.0, "atr": 2.0, "close": 1950.0})
        assert detector.current.regime == Regime.TRENDING

    def test_volatile_detection(self):
        detector = RegimeDetector(primary_tf="M1", volatile_threshold=1.0)
        for _ in range(15):
            detector.update({"adx": 15.0, "atr": 30.0, "close": 1950.0})
        assert detector.current.regime == Regime.VOLATILE

    def test_hysteresis_prevents_flip(self):
        detector = RegimeDetector(primary_tf="M1")
        # 5 trending signals (not enough for M1 hysteresis of 10)
        for _ in range(5):
            detector.update({"adx": 40.0, "atr": 2.0, "close": 1950.0})
        assert detector.current.regime == Regime.RANGING  # still ranging

    def test_history_tracking(self):
        detector = RegimeDetector()
        detector.update({"adx": 25.0, "atr": 1.0, "close": 1950.0})
        assert len(detector.history) == 1
        assert detector.history[0]["regime"] == "ranging"


class TestPerformanceTracker:
    def test_record_and_stats(self):
        tracker = PerformanceTracker(window_size=10)
        for pnl in [10, -5, 15, -3, 8, 12, -7, 20, -2, 6]:
            tracker.record_trade("ema_crossover", pnl)
        stats = tracker.get_stats("ema_crossover")
        assert stats.trades == 10
        assert stats.wins == 6
        assert stats.losses == 4
        assert stats.win_rate == 0.6

    def test_empty_stats(self):
        tracker = PerformanceTracker()
        stats = tracker.get_stats("nonexistent")
        assert stats.trades == 0
        assert stats.win_rate == 0.0

    def test_weight_scores(self):
        tracker = PerformanceTracker(window_size=50)
        for _ in range(10):
            tracker.record_trade("good_strategy", 10.0)
        for _ in range(10):
            tracker.record_trade("bad_strategy", -5.0)
        scores = tracker.get_weight_scores()
        assert scores["good_strategy"] > scores["bad_strategy"]

    def test_should_optimize(self):
        tracker = PerformanceTracker(window_size=50)
        for _ in range(25):
            tracker.record_trade("weak", -1.0)
        assert tracker.should_optimize("weak", min_trades=20, min_win_rate=0.4)

    def test_rl_reward_curve(self):
        tracker = PerformanceTracker()
        for r in [1.0, -0.5, 2.0]:
            tracker.record_rl_reward(r)
        curve = tracker.get_rl_reward_curve()
        assert curve == [1.0, 0.5, 2.5]


class TestConfidenceScorer:
    def test_score_returns_0_to_1(self):
        scorer = ConfidenceScorer()
        features = {
            "vector": [0.5] * 23,
            "session": "london",
        }
        regime = RegimeState(regime=Regime.TRENDING, confidence=0.8)
        score = scorer.score("BUY", features, regime, strategy_win_rate=0.6)
        assert 0.0 <= score <= 1.0

    def test_london_session_higher_than_asian(self):
        scorer = ConfidenceScorer()
        features_london = {"vector": [0.5] * 23, "session": "london"}
        features_asian = {"vector": [0.5] * 23, "session": "asian"}
        regime = RegimeState(regime=Regime.RANGING)
        s_london = scorer.score("BUY", features_london, regime)
        s_asian = scorer.score("BUY", features_asian, regime)
        assert s_london > s_asian

    def test_rl_bonus(self):
        scorer = ConfidenceScorer()
        features = {"vector": [0.5] * 23, "session": "london"}
        regime = RegimeState(regime=Regime.TRENDING)
        s_no_rl = scorer.score("BUY", features, regime)
        s_with_rl = scorer.score("BUY", features, regime, rl_q_values=[-1.0, 5.0, -1.0])
        assert s_with_rl >= s_no_rl


class TestEnsemble:
    def test_combine_unanimous_buy(self):
        ensemble = StrategyEnsemble()
        votes = {
            "ema_crossover": {"direction": "BUY", "confidence": 0.8},
            "rl_agent": {"direction": "BUY", "confidence": 0.9},
            "claude_ai": {"direction": "BUY", "confidence": 0.7},
        }
        result = ensemble.combine(votes)
        assert result["direction"] == "BUY"
        assert result["confidence"] > 0.5

    def test_combine_hold_on_conflict(self):
        ensemble = StrategyEnsemble(confidence_threshold=0.6)
        votes = {
            "ema_crossover": {"direction": "BUY", "confidence": 0.5},
            "rl_agent": {"direction": "SELL", "confidence": 0.5},
        }
        result = ensemble.combine(votes)
        # Conflicting signals should result in low confidence → HOLD
        assert result["direction"] == "HOLD"

    def test_combine_empty_votes(self):
        ensemble = StrategyEnsemble()
        result = ensemble.combine({})
        assert result["direction"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_weight_update_from_tracker(self):
        tracker = PerformanceTracker()
        for _ in range(20):
            tracker.record_trade("rl_agent", 5.0)
        for _ in range(20):
            tracker.record_trade("ema_crossover", -2.0)
        ensemble = StrategyEnsemble(performance_tracker=tracker)
        weights = ensemble.update_weights()
        assert weights["rl_agent"] > weights["ema_crossover"]
