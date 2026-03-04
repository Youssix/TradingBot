"""Strategy ensemble: combine multiple strategy signals with dynamic weighting."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from core.learning.performance_tracker import PerformanceTracker


class StrategyEnsemble:
    """Combine strategy signals using performance-weighted voting.

    Each strategy votes BUY (+1), SELL (-1), or HOLD (0).
    Votes are weighted by rolling performance.
    Final decision uses threshold on weighted score.
    """

    DEFAULT_WEIGHTS = {
        "ema_crossover": 0.25,
        "asian_breakout": 0.15,
        "rl_agent": 0.35,
        "claude_ai": 0.25,
    }

    def __init__(
        self,
        performance_tracker: PerformanceTracker | None = None,
        base_weights: dict[str, float] | None = None,
        agreement_bonus: float = 0.15,
        confidence_threshold: float = 0.4,
    ) -> None:
        self._tracker = performance_tracker
        self._base_weights = base_weights or dict(self.DEFAULT_WEIGHTS)
        self._agreement_bonus = agreement_bonus
        self._confidence_threshold = confidence_threshold
        self._current_weights: dict[str, float] = dict(self._base_weights)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._current_weights)

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, val: float) -> None:
        self._confidence_threshold = max(0.0, min(1.0, val))

    def update_weights(self) -> dict[str, float]:
        """Recalculate weights from performance tracker."""
        if not self._tracker:
            return self._current_weights

        perf_scores = self._tracker.get_weight_scores()
        if not perf_scores:
            return self._current_weights

        for name in self._base_weights:
            base = self._base_weights[name]
            perf = perf_scores.get(name, 0.5)
            # Blend base weight with performance: 40% base, 60% performance
            self._current_weights[name] = base * 0.4 + perf * 0.6

        # Normalize to sum = 1.0
        total = sum(self._current_weights.values())
        if total > 0:
            self._current_weights = {
                k: round(v / total, 4) for k, v in self._current_weights.items()
            }

        return self._current_weights

    def combine(
        self,
        votes: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Combine strategy votes into a single decision.

        Args:
            votes: Dict mapping strategy_name → {
                'direction': 'BUY'/'SELL'/'HOLD',
                'confidence': float (0-1),
                'reasoning': str (optional),
            }

        Returns:
            Dict with 'direction', 'confidence', 'agreement', 'breakdown'.
        """
        if not votes:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "agreement": 0.0,
                "breakdown": {},
            }

        # Convert direction to numeric vote
        dir_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        weighted_score = 0.0
        total_weight = 0.0
        breakdown: dict[str, dict[str, Any]] = {}

        for name, vote in votes.items():
            direction = vote.get("direction", "HOLD")
            conf = vote.get("confidence", 0.5)
            weight = self._current_weights.get(name, 0.1)

            numeric = dir_map.get(direction, 0.0)
            contribution = numeric * conf * weight
            weighted_score += contribution
            total_weight += weight

            breakdown[name] = {
                "direction": direction,
                "confidence": round(conf, 3),
                "weight": round(weight, 3),
                "contribution": round(contribution, 4),
            }

        # Normalize
        if total_weight > 0:
            weighted_score /= total_weight

        # Agreement bonus: if all non-HOLD votes agree on direction
        active_votes = [v for v in votes.values() if v.get("direction", "HOLD") != "HOLD"]
        agreement = 0.0
        if len(active_votes) >= 2:
            dirs = [v["direction"] for v in active_votes]
            if len(set(dirs)) == 1:
                agreement = self._agreement_bonus

        # Final decision
        abs_score = abs(weighted_score) + agreement
        if abs_score >= self._confidence_threshold:
            direction = "BUY" if weighted_score > 0 else "SELL"
        else:
            direction = "HOLD"

        return {
            "direction": direction,
            "confidence": round(min(abs_score, 1.0), 4),
            "agreement": round(agreement, 3),
            "weighted_score": round(weighted_score, 4),
            "breakdown": breakdown,
        }

    def get_status(self) -> dict[str, Any]:
        """Return ensemble status for API."""
        return {
            "weights": self._current_weights,
            "confidence_threshold": self._confidence_threshold,
            "agreement_bonus": self._agreement_bonus,
        }
