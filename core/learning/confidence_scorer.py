"""Score trading signals 0-1 from multiple confidence factors."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from core.learning.regime_detector import Regime, RegimeState


class ConfidenceScorer:
    """Multi-factor confidence scoring for trade signals.

    Factors (weighted):
    1. Indicator confluence (0.25) — how many indicators agree
    2. Regime alignment (0.20) — signal matches market regime
    3. Recent performance (0.20) — strategy's rolling win rate
    4. Session quality (0.15) — London/NY > Asian for XAUUSD
    5. HTF alignment (0.15) — higher timeframe trend agreement
    6. RL Q-value spread (bonus) — added on top if RL is confident

    Output: confidence score in [0, 1]
    """

    def __init__(
        self,
        indicator_weight: float = 0.25,
        regime_weight: float = 0.20,
        performance_weight: float = 0.20,
        session_weight: float = 0.15,
        htf_weight: float = 0.15,
        rl_bonus_weight: float = 0.05,
    ) -> None:
        self._weights = {
            "indicator": indicator_weight,
            "regime": regime_weight,
            "performance": performance_weight,
            "session": session_weight,
            "htf": htf_weight,
            "rl_bonus": rl_bonus_weight,
        }

    def score(
        self,
        direction: str,
        features: dict[str, Any],
        regime: RegimeState,
        strategy_win_rate: float = 0.5,
        rl_q_values: list[float] | None = None,
    ) -> float:
        """Calculate confidence score for a proposed trade.

        Args:
            direction: "BUY" or "SELL"
            features: Dict from FeatureEngine.extract()
            regime: Current RegimeState
            strategy_win_rate: Rolling win rate of the proposing strategy
            rl_q_values: Optional [Q_HOLD, Q_BUY, Q_SELL] from RL agent

        Returns:
            Confidence score in [0, 1].
        """
        vector = features.get("vector", [0.5] * 23)
        is_buy = direction.upper() == "BUY"

        # 1. Indicator confluence
        indicators_agreeing = 0
        total_indicators = 5

        # RSI direction agreement
        rsi_norm = vector[0] if len(vector) > 0 else 0.5
        if (is_buy and rsi_norm < 0.65) or (not is_buy and rsi_norm > 0.35):
            indicators_agreeing += 1

        # EMA cross agreement
        ema_cross = vector[5] if len(vector) > 5 else 0.5
        if (is_buy and ema_cross > 0.5) or (not is_buy and ema_cross < 0.5):
            indicators_agreeing += 1

        # MACD direction
        macd_norm = vector[6] if len(vector) > 6 else 0.0
        if (is_buy and macd_norm > 0) or (not is_buy and macd_norm < 0):
            indicators_agreeing += 1

        # Bollinger position
        bb_pos = vector[8] if len(vector) > 8 else 0.5
        if (is_buy and bb_pos < 0.7) or (not is_buy and bb_pos > 0.3):
            indicators_agreeing += 1

        # Stochastic
        stoch_k = vector[10] if len(vector) > 10 else 0.5
        if (is_buy and stoch_k < 0.75) or (not is_buy and stoch_k > 0.25):
            indicators_agreeing += 1

        indicator_score = indicators_agreeing / total_indicators

        # 2. Regime alignment
        regime_score = 0.5
        if regime.regime == Regime.TRENDING:
            regime_score = 0.8  # trending is good for entries
        elif regime.regime == Regime.RANGING:
            regime_score = 0.4  # ranging = more mean-reversion
        elif regime.regime == Regime.VOLATILE:
            regime_score = 0.3  # volatile = risky

        # 3. Recent performance
        perf_score = np.clip(strategy_win_rate, 0.0, 1.0)

        # 4. Session quality (London/NY better for gold)
        session = features.get("session", "unknown")
        session_scores = {"london": 0.9, "newyork": 0.8, "asian": 0.4, "unknown": 0.5}
        session_score = session_scores.get(session, 0.5)

        # 5. HTF alignment
        htf1 = vector[19] if len(vector) > 19 else 0.5  # normalized 0-1
        htf2 = vector[20] if len(vector) > 20 else 0.5
        if is_buy:
            htf_score = (htf1 + htf2) / 2  # > 0.5 means bullish HTF
        else:
            htf_score = 1.0 - (htf1 + htf2) / 2  # inverted for sells

        # 6. RL Q-value spread bonus
        rl_bonus = 0.0
        if rl_q_values and len(rl_q_values) == 3:
            action_idx = 1 if is_buy else 2
            q_chosen = rl_q_values[action_idx]
            q_others = [rl_q_values[i] for i in range(3) if i != action_idx]
            spread = q_chosen - max(q_others)
            rl_bonus = np.clip(spread, 0, 1)

        # Weighted combination
        confidence = (
            indicator_score * self._weights["indicator"]
            + regime_score * self._weights["regime"]
            + perf_score * self._weights["performance"]
            + session_score * self._weights["session"]
            + htf_score * self._weights["htf"]
            + rl_bonus * self._weights["rl_bonus"]
        )

        return round(float(np.clip(confidence, 0.0, 1.0)), 4)
