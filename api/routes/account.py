"""Account information endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import AccountResponse

router = APIRouter(prefix="/api/account", tags=["account"])


@router.get("", response_model=AccountResponse)
async def get_account(request: Request) -> AccountResponse:
    """Return current MT5 account information."""
    mt5_client = request.app.state.mt5_client
    info = mt5_client.get_account_info()
    return AccountResponse(**info)
