"""Bot status and strategy mode endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import BotStatusResponse, StrategyModeRequest, StrategyModeResponse

router = APIRouter(prefix="/api/bot", tags=["bot"])


@router.get("/status", response_model=BotStatusResponse)
async def get_bot_status(request: Request) -> BotStatusResponse:
    """Return current bot configuration and operational status."""
    config = request.app.state.config

    # Read runtime strategy mode state
    strategy_mode = getattr(request.app.state, "strategy_mode", "independent")
    enabled_strategies = getattr(
        request.app.state,
        "enabled_strategies",
        ["ema_crossover", "asian_breakout", "bos", "candle_pattern"],
    )

    # Get actual strategy names from bot if available
    bot = getattr(request.app.state, "bot", None)
    if bot and hasattr(bot, "_strategies"):
        strategies = [s.name for s in bot._strategies]
    else:
        strategies = list(enabled_strategies)

    return BotStatusResponse(
        mode=config.mode,
        symbol=config.mt5.symbol,
        timeframe=config.mt5.timeframe,
        use_mock=config.mt5.use_mock,
        strategies=strategies,
        strategy_mode=strategy_mode,
        enabled_strategies=list(enabled_strategies),
    )


@router.post("/strategy-mode", response_model=StrategyModeResponse)
async def set_strategy_mode(request: Request, body: StrategyModeRequest) -> StrategyModeResponse:
    """Set the strategy mode (independent or ensemble) and enabled strategies."""
    request.app.state.strategy_mode = body.mode
    request.app.state.enabled_strategies = body.enabled_strategies
    return StrategyModeResponse(
        status="ok",
        mode=body.mode,
        enabled_strategies=body.enabled_strategies,
    )
