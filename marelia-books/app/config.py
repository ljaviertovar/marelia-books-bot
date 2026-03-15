from __future__ import annotations

import logging
import os
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



def _parse_allowed_chat_ids(raw: str) -> Set[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return {int(item) for item in values}



def _parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}



def get_settings() -> Settings:
    try:
        return Settings(
            telegram_bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            notion_api_key=os.environ["NOTION_API_KEY"],
            notion_database_id=os.environ["NOTION_DATABASE_ID"],
            notion_template_id=os.environ["NOTION_TEMPLATE_ID"],
            allowed_chat_ids=_parse_allowed_chat_ids(os.environ["ALLOWED_CHAT_IDS"]),
            dry_run=_parse_bool(os.environ.get("DRY_RUN")),
        )
    except KeyError as exc:
        raise RuntimeError(f"Missing required environment variable: {exc.args[0]}") from exc
    except (ValueError, ValidationError) as exc:
        raise RuntimeError(f"Invalid environment configuration: {exc}") from exc



def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
