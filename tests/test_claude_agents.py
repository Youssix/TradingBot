"""Tests for Claude strategy and reviewer with mocked API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from core.learning.claude_strategy import ClaudeStrategy, ANTHROPIC_AVAILABLE
from core.learning.claude_reviewer import ClaudeReviewer


def _make_features() -> dict:
    """Create sample features dict."""
    return {
        "vector": [0.5] * 23,
        "names": ["feat"] * 23,
        "session": "london",
        "close": 1950.0,
        "atr": 3.5,
        "rsi": 55.0,
        "adx": 30.0,
        "timestamp": "2024-01-15T10:00:00",
    }


def _mock_response(text: str):
    """Create a mock Anthropic response."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


class TestClaudeStrategy:
    def test_disabled_without_api_key(self):
        strategy = ClaudeStrategy(api_key="")
        assert not strategy.enabled
        result = strategy.analyze(_make_features())
        assert result["direction"] == "HOLD"

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_analyze_returns_valid_result(self):
        strategy = ClaudeStrategy(api_key="test-key", min_interval=0)
        mock_json = json.dumps({
            "direction": "BUY",
            "confidence": 0.75,
            "reasoning": "Strong bullish momentum"
        })
        strategy._client = MagicMock()
        strategy._client.messages.create.return_value = _mock_response(mock_json)

        result = strategy.analyze(_make_features(), regime="trending")
        assert result["direction"] == "BUY"
        assert result["confidence"] == 0.75
        assert "bullish" in result["reasoning"].lower()

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_analyze_handles_invalid_json(self):
        strategy = ClaudeStrategy(api_key="test-key", min_interval=0)
        strategy._client = MagicMock()
        strategy._client.messages.create.return_value = _mock_response("not valid json")

        result = strategy.analyze(_make_features())
        assert result["direction"] == "HOLD"
        assert "JSON" in result["reasoning"] or "parse" in result["reasoning"].lower()

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_analyze_handles_api_error(self):
        strategy = ClaudeStrategy(api_key="test-key", min_interval=0)
        strategy._client = MagicMock()
        strategy._client.messages.create.side_effect = Exception("API rate limit")

        result = strategy.analyze(_make_features())
        assert result["direction"] == "HOLD"
        assert "error" in result["reasoning"].lower()

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_rate_limiting(self):
        strategy = ClaudeStrategy(api_key="test-key", min_interval=60)
        mock_json = json.dumps({"direction": "SELL", "confidence": 0.6, "reasoning": "test"})
        strategy._client = MagicMock()
        strategy._client.messages.create.return_value = _mock_response(mock_json)

        # First call
        result1 = strategy.analyze(_make_features())
        assert result1["direction"] == "SELL"

        # Second call within rate limit — should return cached result
        result2 = strategy.analyze(_make_features())
        assert result2["direction"] == "SELL"
        assert strategy._client.messages.create.call_count == 1

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_validates_direction(self):
        strategy = ClaudeStrategy(api_key="test-key", min_interval=0)
        mock_json = json.dumps({"direction": "INVALID", "confidence": 0.5, "reasoning": "test"})
        strategy._client = MagicMock()
        strategy._client.messages.create.return_value = _mock_response(mock_json)

        result = strategy.analyze(_make_features())
        assert result["direction"] == "HOLD"  # invalid → HOLD

    def test_call_count(self):
        strategy = ClaudeStrategy(api_key="")
        assert strategy.call_count == 0


class TestClaudeReviewer:
    def test_disabled_without_api_key(self):
        reviewer = ClaudeReviewer(api_key="")
        assert not reviewer.enabled

    def test_fallback_review_no_trades(self):
        reviewer = ClaudeReviewer(api_key="")
        result = reviewer.review_trades([])
        assert result["analysis"] == "No trades to review"

    def test_fallback_review_with_trades(self):
        reviewer = ClaudeReviewer(api_key="")
        trades = [
            {"pnl": 10, "direction": "BUY", "regime": "trending", "session": "london"},
            {"pnl": -5, "direction": "SELL", "regime": "ranging", "session": "asian"},
            {"pnl": -3, "direction": "BUY", "regime": "ranging", "session": "asian"},
            {"pnl": 8, "direction": "BUY", "regime": "trending", "session": "london"},
            {"pnl": -2, "direction": "SELL", "regime": "volatile", "session": "asian"},
        ]
        result = reviewer.review_trades(trades)
        assert "5 trades" in result["analysis"]
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["strategy_rules"], list)

    def test_fallback_detects_session_losses(self):
        reviewer = ClaudeReviewer(api_key="")
        trades = [
            {"pnl": -5, "session": "asian"},
            {"pnl": -3, "session": "asian"},
            {"pnl": -4, "session": "asian"},
            {"pnl": 10, "session": "london"},
        ]
        result = reviewer.review_trades(trades)
        # Should recommend avoiding asian session
        has_session_rule = any(
            "asian" in str(r).lower()
            for r in result.get("strategy_rules", [])
        )
        assert has_session_rule

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_review_with_mock_api(self):
        reviewer = ClaudeReviewer(api_key="test-key")
        mock_json = json.dumps({
            "analysis": "RL agent performs well in trends",
            "recommendations": ["Add volatility filter"],
            "strategy_rules": [{"condition": "ADX < 20", "action": "HOLD", "reason": "low trend"}],
            "market_brief": "Bullish outlook",
            "score": 0.7,
        })
        reviewer._client = MagicMock()
        reviewer._client.messages.create.return_value = _mock_response(mock_json)

        trades = [{"pnl": 10, "direction": "BUY", "regime": "trending", "session": "london"}]
        result = reviewer.review_trades(trades)
        assert result["analysis"] == "RL agent performs well in trends"
        assert len(result["recommendations"]) == 1
        assert result["score"] == 0.7

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_review_handles_api_error(self):
        reviewer = ClaudeReviewer(api_key="test-key")
        reviewer._client = MagicMock()
        reviewer._client.messages.create.side_effect = Exception("timeout")

        trades = [{"pnl": 5, "direction": "BUY", "regime": "trending", "session": "london"}]
        result = reviewer.review_trades(trades)
        # Should fall back to rule-based review
        assert "Fallback" in result["analysis"]

    def test_insights_stored(self):
        reviewer = ClaudeReviewer(api_key="")
        trades = [{"pnl": 5, "session": "london"}]
        reviewer.review_trades(trades)
        assert len(reviewer.insights) == 1
        assert reviewer.insights[0]["review_type"] == "fallback_review"

    def test_market_brief_without_api(self):
        reviewer = ClaudeReviewer(api_key="")
        result = reviewer.generate_market_brief(_make_features(), "trending")
        assert "trending" in result["brief"]
        assert result["bias"] == "neutral"

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_market_brief_with_mock_api(self):
        reviewer = ClaudeReviewer(api_key="test-key")
        mock_json = json.dumps({
            "brief": "Gold is testing resistance at 1960",
            "bias": "bullish",
            "key_levels": [1940, 1960],
        })
        reviewer._client = MagicMock()
        reviewer._client.messages.create.return_value = _mock_response(mock_json)

        result = reviewer.generate_market_brief(_make_features(), "trending")
        assert result["bias"] == "bullish"
        assert len(result["key_levels"]) == 2


class TestClaudeRulesFilter:
    """Test the _apply_claude_rules function from api/main.py."""

    def _make_mock_db(self, rules):
        """Create a mock DB that returns given rules."""
        db = MagicMock()
        db.get_active_rules.return_value = [{"rules_json": rules}]
        return db

    def test_no_rules_passthrough(self):
        from api.main import _apply_claude_rules
        db = MagicMock()
        db.get_active_rules.return_value = []
        action, reason = _apply_claude_rules(1, {"adx": 30}, "trending", "london", db)
        assert action == 1
        assert reason == ""

    def test_adx_rule_triggers_hold(self):
        from api.main import _apply_claude_rules
        rules = [{"condition": "ADX < 20", "action": "HOLD", "reason": "low trend"}]
        db = self._make_mock_db(rules)
        action, reason = _apply_claude_rules(1, {"adx": 15}, "trending", "london", db)
        assert action == 0  # HOLD
        assert "low trend" in reason

    def test_adx_rule_no_trigger(self):
        from api.main import _apply_claude_rules
        rules = [{"condition": "ADX < 20", "action": "HOLD", "reason": "low trend"}]
        db = self._make_mock_db(rules)
        action, reason = _apply_claude_rules(1, {"adx": 30}, "trending", "london", db)
        assert action == 1  # BUY unchanged
        assert reason == ""

    def test_session_rule_triggers(self):
        from api.main import _apply_claude_rules
        rules = [{"condition": "session == asian AND regime == ranging", "action": "HOLD", "reason": "avoid asian ranges"}]
        db = self._make_mock_db(rules)
        action, reason = _apply_claude_rules(2, {"adx": 25}, "ranging", "asian", db)
        assert action == 0  # HOLD
        assert "asian" in reason

    def test_session_rule_no_match(self):
        from api.main import _apply_claude_rules
        rules = [{"condition": "session == asian AND regime == ranging", "action": "HOLD", "reason": "avoid asian ranges"}]
        db = self._make_mock_db(rules)
        action, reason = _apply_claude_rules(2, {"adx": 25}, "trending", "asian", db)
        assert action == 2  # SELL unchanged (regime doesn't match)

    def test_confidence_rule(self):
        from api.main import _apply_claude_rules
        rules = [{"condition": "confidence < 0.5", "action": "HOLD", "reason": "low confidence"}]
        db = self._make_mock_db(rules)
        action, reason = _apply_claude_rules(1, {"confidence": 0.3}, "trending", "london", db)
        assert action == 0
        assert "confidence" in reason

    def test_no_db_passthrough(self):
        from api.main import _apply_claude_rules
        action, reason = _apply_claude_rules(1, {}, "trending", "london", None)
        assert action == 1
