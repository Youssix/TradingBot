"""Bot status endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import BotStatusResponse

router = APIRouter(prefix="/api/bot", tags=["bot"])


@router.get("/status", response_model=BotStatusResponse)
async def get_bot_status(request: Request) -> BotStatusResponse:
    """Return current bot configuration and operational status."""
    config = request.app.state.config
    return BotStatusResponse(
        mode=config.mode,
        symbol=config.mt5.symbol,
        timeframe=config.mt5.timeframe,
        use_mock=config.mt5.use_mock,
        strategies=["ema_crossover", "asian_breakout"],
    )
