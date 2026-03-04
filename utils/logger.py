from __future__ import annotations
import sys
from loguru import logger


def setup_logger(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure loguru with console and file sinks."""
    from pathlib import Path
    Path(log_dir).mkdir(exist_ok=True)

    logger.remove()  # Remove default handler

    # Console sink
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # General file sink with rotation
    logger.add(
        f"{log_dir}/bot_{{time:YYYY-MM-DD}}.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )

    # Trade-specific file sink
    logger.add(
        f"{log_dir}/trades_{{time:YYYY-MM-DD}}.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        rotation="10 MB",
        retention="90 days",
        filter=lambda record: "trade" in record["extra"],
    )


def get_trade_logger():
    """Get a logger instance with trade context for trade-specific logging."""
    return logger.bind(trade=True)
