"""Classify market regime: trending / ranging / volatile with hysteresis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

# Hysteresis bar counts adapt to timeframe
HYSTERESIS_BARS: dict[str, int] = {
    "M1": 10,
    "M5": 8,
    "M15": 6,
    "H1": 4,
    "H4": 3,
    "D1": 2,
}


class Regime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class RegimeState:
    regime: Regime = Regime.RANGING
    confidence: float = 0.5
    bars_in_regime: int = 0
    adx: float = 25.0
    atr_ratio: float = 0.0


class RegimeDetector:
    """Classify market regime using ADX + ATR ratio with hysteresis.

    Rules:
    - ADX > trending_threshold AND ATR ratio < volatile_threshold → TRENDING
    - ATR ratio > volatile_threshold → VOLATILE
    - Otherwise → RANGING

    Hysteresis: regime only changes after N consecutive bars signal the new regime.
    """

    def __init__(
        self,
        primary_tf: str = "M1",
        trending_threshold: float = 25.0,
        volatile_threshold: float = 1.5,
    ) -> None:
        self._tf = primary_tf
        self._trending_threshold = trending_threshold
        self._volatile_threshold = volatile_threshold
        self._hysteresis = HYSTERESIS_BARS.get(primary_tf, 5)
        self._state = RegimeState()
        self._pending_regime: Regime | None = None
        self._pending_count: int = 0
        self._history: list[dict[str, Any]] = []

    @property
    def current(self) -> RegimeState:
        return self._state

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history[-100:])

    def update(self, features: dict[str, Any]) -> RegimeState:
        """Update regime classification from feature dict.

        Args:
            features: Dict from FeatureEngine.extract() with 'adx', 'atr', 'close', etc.

        Returns:
            Updated RegimeState.
        """
        adx = features.get("adx", 25.0)
        atr = features.get("atr", 1.0)
        close = features.get("close", 1.0)
        atr_ratio = (atr / close * 100) if close > 0 else 0.0  # ATR as % of price

        # Determine raw classification
        if atr_ratio > self._volatile_threshold:
            raw_regime = Regime.VOLATILE
            confidence = min(atr_ratio / (self._volatile_threshold * 2), 1.0)
        elif adx > self._trending_threshold:
            raw_regime = Regime.TRENDING
            confidence = min(adx / 100.0 * 2, 1.0)
        else:
            raw_regime = Regime.RANGING
            confidence = 1.0 - (adx / self._trending_threshold)

        # Hysteresis logic
        if raw_regime != self._state.regime:
            if raw_regime == self._pending_regime:
                self._pending_count += 1
            else:
                self._pending_regime = raw_regime
                self._pending_count = 1

            if self._pending_count >= self._hysteresis:
                old = self._state.regime
                self._state = RegimeState(
                    regime=raw_regime,
                    confidence=confidence,
                    bars_in_regime=0,
                    adx=adx,
                    atr_ratio=atr_ratio,
                )
                self._pending_regime = None
                self._pending_count = 0
                logger.info(f"Regime change: {old.value} → {raw_regime.value} (ADX={adx:.1f})")
        else:
            self._state.bars_in_regime += 1
            self._state.confidence = confidence
            self._state.adx = adx
            self._state.atr_ratio = atr_ratio
            self._pending_regime = None
            self._pending_count = 0

        # Record history
        self._history.append({
            "regime": self._state.regime.value,
            "confidence": round(confidence, 3),
            "adx": round(adx, 2),
            "atr_ratio": round(atr_ratio, 4),
            "timestamp": features.get("timestamp"),
        })
        if len(self._history) > 500:
            self._history = self._history[-250:]

        return self._state
