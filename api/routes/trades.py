"""Trade history endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query, Request

from api.schemas import TradeResponse

router = APIRouter(prefix="/api/trades", tags=["trades"])


@router.get("/open", response_model=list[TradeResponse])
async def get_open_trades(request: Request) -> list[TradeResponse]:
    """Return all currently open trades (status = open or dry-run, not yet closed)."""
    db = request.app.state.db
    assert db._conn is not None
    cursor = db._conn.execute(
        "SELECT * FROM trades WHERE status IN ('open', 'dry-run') AND closed_at IS NULL"
    )
    rows = [dict(row) for row in cursor.fetchall()]
    return [TradeResponse(**row) for row in rows]


@router.get("", response_model=list[TradeResponse])
async def get_trades(
    request: Request,
    status: str | None = Query(default=None, pattern=r"^(open|closed|dry-run)$"),
    strategy: str | None = Query(default=None),
    trade_date: date | None = Query(default=None, alias="date"),
) -> list[TradeResponse]:
    """Return trades with optional filters.

    Query params:
        status   - "open" or "closed"
        strategy - filter by strategy name
        date     - filter by open date (YYYY-MM-DD)
    """
    db = request.app.state.db

    if trade_date is not None:
        rows = db.get_trades_by_date(trade_date)
    elif strategy is not None:
        rows = db.get_trades_by_strategy(strategy)
    elif status == "open":
        rows = db.get_open_trades()
    else:
        # Return today's trades as a sensible default for "all recent"
        rows = db.get_trades_by_date(date.today())

    # Apply status filter if combined with other filters
    if status is not None and trade_date is not None:
        rows = [r for r in rows if r["status"] == status]
    if status is not None and strategy is not None:
        rows = [r for r in rows if r["status"] == status]

    return [TradeResponse(**row) for row in rows]
