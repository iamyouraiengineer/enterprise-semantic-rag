"""
config/log_config.py
Production-grade logging configuration using Loguru.

Design decisions:
- We remove the default loguru handler first to avoid duplicate logs.
- Console output is human-readable with colors for local dev.
- File output is rotating and UTF-8 encoded for production forensics.
- enqueue=True makes the file handler thread-safe and async-safe.
- backtrace/diagnose are enabled only in debug mode to prevent leaking
  sensitive variable values in production logs.
"""

import sys
from pathlib import Path

from loguru import logger

from config.settings import get_settings

# Guard against double configuration if imported multiple times
_is_configured: bool = False


def configure_logging() -> None:
    """
    Configure global Loguru handlers.
    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _is_configured
    if _is_configured:
        return

    settings = get_settings()

    # ------------------------------------------------------------------
    # 1. Remove the default loguru handler
    # ------------------------------------------------------------------
    logger.remove()

    # ------------------------------------------------------------------
    # 2. Console Handler (stderr) — colored, human-readable
    # ------------------------------------------------------------------
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=settings.debug,
        diagnose=settings.debug,
    )

    # ------------------------------------------------------------------
    # 3. File Handler — rotating, persistent
    # ------------------------------------------------------------------
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "app.log",
        rotation="10 MB",      # New file when current hits 10 MB
        retention="30 days",   # Auto-delete files older than 30 days
        level=settings.log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        enqueue=True,          # Thread-safe / async-safe queue
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 4. Log the successful configuration
    # ------------------------------------------------------------------
    logger.info(
        "Logging configured | level={} | app={} | version={}",
        settings.log_level,
        settings.app_name,
        settings.app_version,
    )

    _is_configured = True


# Re-export the global logger so other modules can do:
#   from config.log_config import logger
__all__ = ["configure_logging", "logger"]



