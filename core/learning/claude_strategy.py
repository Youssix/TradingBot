"""Claude AI as a live trading strategy via Anthropic SDK."""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

SYSTEM_PROMPT = """You are an expert gold (XAU/USD) trading analyst. You analyze market data and provide trading decisions.

Given the current market state, respond with EXACTLY one JSON object:
{
  "direction": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}

Rules:
- Only output valid JSON, no other text
- HOLD if uncertain (confidence < 0.3)
- Consider trend, momentum, volatility, and session timing
- Be conservative: prefer HOLD over low-confidence trades
- Gold-specific: respect London/NY session momentum, Asian consolidation"""


class ClaudeStrategy:
    """Claude AI as a trading strategy.

    Sends structured market data to Claude and receives BUY/SELL/HOLD decisions.
    Rate-limited and cached to control API costs.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        min_interval: float = 30.0,
        max_tokens: int = 256,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._min_interval = min_interval
        self._max_tokens = max_tokens
        self._last_call_time = 0.0
        self._last_result: dict[str, Any] | None = None
        self._client: Any = None
        self._call_count = 0
        self._enabled = bool(api_key) and ANTHROPIC_AVAILABLE

        if self._enabled:
            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude strategy initialized")
        else:
            if not ANTHROPIC_AVAILABLE:
                logger.warning("anthropic SDK not installed — Claude strategy disabled")
            elif not api_key:
                logger.warning("No ANTHROPIC_API_KEY — Claude strategy disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def call_count(self) -> int:
        return self._call_count

    def analyze(self, features: dict[str, Any], regime: str = "unknown") -> dict[str, Any]:
        """Analyze market state and return trading decision.

        Args:
            features: Dict from FeatureEngine.extract()
            regime: Current regime string (trending/ranging/volatile)

        Returns:
            Dict with 'direction', 'confidence', 'reasoning'.
        """
        # Rate limiting + cache
        now = time.time()
        if now - self._last_call_time < self._min_interval and self._last_result:
            return self._last_result

        if not self._enabled:
            return {"direction": "HOLD", "confidence": 0.0, "reasoning": "Claude strategy disabled"}

        # Build market summary for Claude
        vector = features.get("vector", [])
        names = features.get("names", [])
        session = features.get("session", "unknown")
        close = features.get("close", 0.0)
        atr = features.get("atr", 0.0)
        rsi = features.get("rsi", 50.0)
        adx = features.get("adx", 25.0)

        # Build human-readable feature summary
        feature_lines = []
        for name, val in zip(names[:16], vector[:16]):
            feature_lines.append(f"  {name}: {val:.4f}")
        feature_str = "\n".join(feature_lines)

        prompt = f"""Current Gold (XAU/USD) Market State:
Price: {close:.2f}
ATR: {atr:.2f}
RSI: {rsi:.1f}
ADX: {adx:.1f}
Regime: {regime}
Session: {session}

Normalized Features:
{feature_str}

HTF1 Trend: {'Bullish' if len(vector) > 19 and vector[19] > 0.5 else 'Bearish'}
HTF2 Trend: {'Bullish' if len(vector) > 20 and vector[20] > 0.5 else 'Bearish'}

Provide your trading decision as JSON."""

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            # Parse JSON from response
            result = json.loads(text)
            result.setdefault("direction", "HOLD")
            result.setdefault("confidence", 0.0)
            result.setdefault("reasoning", "")

            # Validate
            if result["direction"] not in ("BUY", "SELL", "HOLD"):
                result["direction"] = "HOLD"
            result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

            self._last_call_time = now
            self._last_result = result
            self._call_count += 1
            logger.info(
                f"Claude analysis: {result['direction']} "
                f"(conf={result['confidence']:.2f}) — {result['reasoning'][:80]}"
            )
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Claude returned invalid JSON: {e}")
            return {"direction": "HOLD", "confidence": 0.0, "reasoning": f"JSON parse error: {e}"}
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {"direction": "HOLD", "confidence": 0.0, "reasoning": f"API error: {e}"}
