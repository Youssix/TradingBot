"""Candle / OHLCV data endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from api.schemas import CandleResponse

router = APIRouter(prefix="/api/candles", tags=["candles"])


@router.get("", response_model=list[CandleResponse])
async def get_candles(
    request: Request,
    count: int = Query(default=200, ge=100, le=5000),
    timeframe: str = Query(default="H1"),
) -> list[CandleResponse]:
    """Return OHLCV candle data for the configured symbol."""
    mt5_client = request.app.state.mt5_client
    config = request.app.state.config
    symbol = config.mt5.symbol

    df = mt5_client.get_rates(symbol, timeframe, count)

    candles: list[CandleResponse] = []
    for _, row in df.iterrows():
        candles.append(
            CandleResponse(
                datetime=row["datetime"].isoformat()
                if hasattr(row["datetime"], "isoformat")
                else str(row["datetime"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
        )
    return candles
