"""Claude AI reviewer: reviews RL decisions, generates rules, runs backtest loops."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


REVIEW_SYSTEM = """You are a quantitative trading strategy analyst reviewing an RL agent's recent trading decisions on Gold (XAU/USD).

Analyze the trades and provide:
1. Pattern analysis: what works, what doesn't
2. Specific filter rules (e.g., "avoid SELL when ADX < 20")
3. Regime-specific suggestions
4. A concise market brief

Respond with EXACTLY one JSON object:
{
  "analysis": "brief summary of patterns found",
  "recommendations": ["list", "of", "specific", "improvements"],
  "strategy_rules": [
    {"condition": "ADX < 20", "action": "HOLD", "reason": "low trend strength"},
    {"condition": "session == asian AND regime == ranging", "action": "HOLD", "reason": "avoid asian ranges"}
  ],
  "market_brief": "short market outlook statement",
  "score": 0.0 to 1.0
}"""


class ClaudeReviewer:
    """Reviews RL agent performance and generates strategy rules.

    - Periodically reviews recent trades
    - Generates human-readable rules for filtering
    - Stores insights in DB
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = None
        self._enabled = bool(api_key) and ANTHROPIC_AVAILABLE
        self._insights: list[dict[str, Any]] = []
        self._review_count = 0

        if self._enabled:
            self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def insights(self) -> list[dict[str, Any]]:
        return list(self._insights[-50:])

    def review_trades(
        self,
        trades: list[dict[str, Any]],
        rl_stats: dict[str, Any] | None = None,
        regime: str = "unknown",
    ) -> dict[str, Any]:
        """Review a batch of recent trades and generate insights.

        Args:
            trades: List of trade dicts with pnl, direction, features, etc.
            rl_stats: RL agent stats dict
            regime: Current market regime

        Returns:
            Dict with analysis, recommendations, rules, market_brief, score.
        """
        if not self._enabled:
            return self._fallback_review(trades)

        if not trades:
            return {
                "analysis": "No trades to review",
                "recommendations": [],
                "strategy_rules": [],
                "market_brief": "Insufficient data",
                "score": 0.5,
            }

        # Build trade summary
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        trade_lines = []
        for t in trades[-20:]:  # Last 20 trades
            trade_lines.append(
                f"  {t.get('direction', '?')} | PnL: {t.get('pnl', 0):.2f} | "
                f"Regime: {t.get('regime', '?')} | Session: {t.get('session', '?')}"
            )
        trade_str = "\n".join(trade_lines)

        rl_info = ""
        if rl_stats:
            rl_info = (
                f"\nRL Agent Stats: episode={rl_stats.get('episode', 0)}, "
                f"epsilon={rl_stats.get('epsilon', 0):.3f}, "
                f"total_reward={rl_stats.get('total_reward', 0):.1f}"
            )

        prompt = f"""Review these recent Gold (XAU/USD) trades:

Summary: {len(trades)} trades, {len(wins)}W/{len(losses)}L, Total PnL: ${total_pnl:.2f}
Win Rate: {len(wins)/len(trades)*100:.1f}%
Current Regime: {regime}
{rl_info}

Recent Trades:
{trade_str}

Analyze patterns and suggest specific filter rules."""

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=REVIEW_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            result = json.loads(text)

            # Validate structure
            result.setdefault("analysis", "")
            result.setdefault("recommendations", [])
            result.setdefault("strategy_rules", [])
            result.setdefault("market_brief", "")
            result.setdefault("score", 0.5)

            insight = {
                "review_type": "trade_review",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trade_count": len(trades),
                "win_rate": len(wins) / len(trades) if trades else 0,
                **result,
            }
            self._insights.append(insight)
            self._review_count += 1

            logger.info(f"Claude review #{self._review_count}: {result['analysis'][:100]}")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Claude review returned invalid JSON: {e}")
            return self._fallback_review(trades)
        except Exception as e:
            logger.error(f"Claude review API error: {e}")
            return self._fallback_review(trades)

    def _fallback_review(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Simple rule-based review when Claude is unavailable."""
        if not trades:
            return {
                "analysis": "No trades to review",
                "recommendations": [],
                "strategy_rules": [],
                "market_brief": "Insufficient data",
                "score": 0.5,
            }

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        win_rate = len(wins) / len(trades) if trades else 0

        recommendations = []
        rules = []

        if win_rate < 0.4:
            recommendations.append("Win rate below 40% — consider tighter entry filters")
            rules.append({
                "condition": "confidence < 0.5",
                "action": "HOLD",
                "reason": "low win rate requires higher confidence threshold",
            })

        # Check if losses concentrate in certain sessions
        session_losses: dict[str, int] = {}
        for t in trades:
            if t.get("pnl", 0) <= 0:
                s = t.get("session", "unknown")
                session_losses[s] = session_losses.get(s, 0) + 1
        worst_session = max(session_losses, key=session_losses.get) if session_losses else None
        if worst_session and session_losses.get(worst_session, 0) > len(trades) * 0.3:
            recommendations.append(f"High loss concentration in {worst_session} session")
            rules.append({
                "condition": f"session == {worst_session}",
                "action": "HOLD",
                "reason": f"most losses occur in {worst_session} session",
            })

        result = {
            "analysis": f"Reviewed {len(trades)} trades. Win rate: {win_rate:.1%}. Fallback analysis.",
            "recommendations": recommendations,
            "strategy_rules": rules,
            "market_brief": "Market analysis unavailable (Claude offline)",
            "score": win_rate,
        }

        self._insights.append({
            "review_type": "fallback_review",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_count": len(trades),
            "win_rate": win_rate,
            **result,
        })

        return result

    def generate_market_brief(self, features: dict[str, Any], regime: str) -> dict[str, Any]:
        """Generate a market brief/outlook."""
        if not self._enabled:
            return {
                "brief": f"Market in {regime} regime. Claude unavailable for detailed analysis.",
                "bias": "neutral",
                "key_levels": [],
            }

        close = features.get("close", 0)
        rsi = features.get("rsi", 50)
        adx = features.get("adx", 25)
        session = features.get("session", "unknown")

        prompt = f"""Provide a brief Gold (XAU/USD) market outlook:
Price: {close:.2f}, RSI: {rsi:.1f}, ADX: {adx:.1f}
Regime: {regime}, Session: {session}

Respond as JSON: {{"brief": "...", "bias": "bullish/bearish/neutral", "key_levels": [price1, price2]}}"""

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            result = json.loads(response.content[0].text.strip())
            result.setdefault("brief", "")
            result.setdefault("bias", "neutral")
            result.setdefault("key_levels", [])

            self._insights.append({
                "review_type": "market_brief",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **result,
            })
            return result
        except Exception as e:
            logger.error(f"Market brief error: {e}")
            return {"brief": f"Error: {e}", "bias": "neutral", "key_levels": []}
