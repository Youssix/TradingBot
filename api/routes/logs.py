"""Stream bot logs to the frontend."""

from __future__ import annotations

import glob
import os
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api", tags=["logs"])


@router.get("/logs")
def get_logs(lines: int = Query(100, ge=1, le=1000)) -> list[str]:
    """Return the last N lines from today's bot log file."""
    log_dir = Path("logs")
    if not log_dir.exists():
        return []

    # Find the most recent bot log file
    log_files = sorted(glob.glob(str(log_dir / "bot_*.log")), reverse=True)
    if not log_files:
        return []

    log_path = log_files[0]
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
        return [line.rstrip("\n") for line in all_lines[-lines:]]
    except Exception:
        return []


@router.delete("/logs")
def clear_logs() -> dict[str, str]:
    """Truncate today's bot log file."""
    log_dir = Path("logs")
    if not log_dir.exists():
        return {"status": "ok"}

    log_files = sorted(glob.glob(str(log_dir / "bot_*.log")), reverse=True)
    if not log_files:
        return {"status": "ok"}

    try:
        with open(log_files[0], "w") as f:
            f.truncate(0)
    except Exception:
        pass
    return {"status": "ok"}
