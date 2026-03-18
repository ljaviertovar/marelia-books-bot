from __future__ import annotations

import logging
import os
import re
import sys
import traceback as _traceback
from typing import Set

from pydantic import BaseModel, ValidationError


class Settings(BaseModel):
    telegram_bot_token: str
    openai_api_key: str
    notion_api_key: str
    notion_database_id: str
    notion_template_id: str
    allowed_chat_ids: Set[int]
    dry_run: bool = False


_NOTION_ID_RE = re.compile(r"([0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})")


def _require_non_empty(name: str, raw: str) -> str:
    value = raw.strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} cannot be empty")
    return value


def _normalize_notion_id(name: str, raw: str) -> str:
    value = _require_non_empty(name, raw)
    match = _NOTION_ID_RE.search(value)
    if not match:
        raise RuntimeError(
            f"Environment variable {name} must contain a valid Notion id or URL"
        )

    compact = match.group(1).replace("-", "").lower()
    return (
        f"{compact[:8]}-{compact[8:12]}-{compact[12:16]}-"
        f"{compact[16:20]}-{compact[20:]}"
    )



def _parse_allowed_chat_ids(raw: str) -> Set[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise RuntimeError("Environment variable ALLOWED_CHAT_IDS must contain at least one chat id")
    return {int(item) for item in values}



def _parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}



def get_settings() -> Settings:
    try:
        return Settings(
            telegram_bot_token=_require_non_empty(
                "TELEGRAM_BOT_TOKEN", os.environ["TELEGRAM_BOT_TOKEN"]
            ),
            openai_api_key=_require_non_empty("OPENAI_API_KEY", os.environ["OPENAI_API_KEY"]),
            notion_api_key=_require_non_empty("NOTION_API_KEY", os.environ["NOTION_API_KEY"]),
            notion_database_id=_normalize_notion_id(
                "NOTION_DATABASE_ID", os.environ["NOTION_DATABASE_ID"]
            ),
            notion_template_id=_normalize_notion_id(
                "NOTION_TEMPLATE_ID", os.environ["NOTION_TEMPLATE_ID"]
            ),
            allowed_chat_ids=_parse_allowed_chat_ids(os.environ["ALLOWED_CHAT_IDS"]),
            dry_run=_parse_bool(os.environ.get("DRY_RUN")),
        )
    except KeyError as exc:
        raise RuntimeError(f"Missing required environment variable: {exc.args[0]}") from exc
    except (ValueError, ValidationError) as exc:
        raise RuntimeError(f"Invalid environment configuration: {exc}") from exc



_R = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


class _FriendlyFormatter(logging.Formatter):
    """Human-friendly terminal formatter with step markers."""

    _USE_COLOR = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        c = self._USE_COLOR

        if record.levelno == logging.DEBUG:
            out = f"   · {msg}"
            return f"{_DIM}{out}{_R}" if c else out

        if record.levelno == logging.INFO:
            return f"{_GREEN}{_BOLD}==>> {_R}{msg}" if c else f"==>> {msg}"

        if record.levelno == logging.WARNING:
            out = f" !!  {msg}"
            return f"{_YELLOW}{out}{_R}" if c else out

        # ERROR / CRITICAL
        out = f" XX  {msg}"
        parts = [f"{_RED}{out}{_R}" if c else out]
        if record.exc_info:
            tb = "".join(_traceback.format_exception(*record.exc_info))
            parts.append(f"{_RED}{tb.rstrip()}{_R}" if c else tb.rstrip())
        return "\n".join(parts)


def configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(_FriendlyFormatter())
    logging.basicConfig(level=level, handlers=[handler])
    # Silenciar logs internos de librerías HTTP
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
