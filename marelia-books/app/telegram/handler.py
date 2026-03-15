from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TelegramChat(BaseModel):
    id: int


class TelegramPhoto(BaseModel):
    file_id: str
    width: int
    height: int


class TelegramMessage(BaseModel):
    message_id: int
    chat: TelegramChat
    text: str | None = None
    caption: str | None = None
    photo: list[TelegramPhoto] = Field(default_factory=list)


class TelegramUpdate(BaseModel):
    update_id: int
    message: TelegramMessage | None = None


@dataclass
class ParsedCommand:
    kind: str  # text | image
    title: str | None = None
    file_id: str | None = None


class UpdateDeduplicator:
    def __init__(self, max_entries: int = 2000) -> None:
        self._seen = set()
        self._queue: deque[int] = deque()
        self._max_entries = max_entries

    def is_duplicate(self, update_id: int) -> bool:
        if update_id in self._seen:
            return True

        self._seen.add(update_id)
        self._queue.append(update_id)

        while len(self._queue) > self._max_entries:
            old = self._queue.popleft()
            self._seen.discard(old)

        return False


class TelegramClient:
    def __init__(self, bot_token: str, timeout_seconds: float = 20.0) -> None:
        self._bot_token = bot_token
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def send_message(self, chat_id: int, text: str) -> None:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        response = await self._client.post(url, json={"chat_id": chat_id, "text": text})
        response.raise_for_status()

    async def download_file(self, file_id: str) -> tuple[bytes, str]:
        file_info = await self._client.get(
            f"https://api.telegram.org/bot{self._bot_token}/getFile",
            params={"file_id": file_id},
        )
        file_info.raise_for_status()
        file_path = file_info.json()["result"]["file_path"]

        content = await self._client.get(f"https://api.telegram.org/file/bot{self._bot_token}/{file_path}")
        content.raise_for_status()

        mime_type = "image/jpeg"
        if file_path.endswith(".png"):
            mime_type = "image/png"

        return content.content, mime_type



def parse_supported_command(message: TelegramMessage) -> ParsedCommand | None:
    text = (message.text or "").strip()
    caption = (message.caption or "").strip()

    if text.startswith("Add Book ") and len(text) > len("Add Book "):
        title = text[len("Add Book ") :].strip()
        if title:
            return ParsedCommand(kind="text", title=title)

    if message.photo and caption == "Add Book":
        photo = sorted(message.photo, key=lambda x: x.width * x.height)[-1]
        return ParsedCommand(kind="image", file_id=photo.file_id)

    return None



def log_incoming_command(update: TelegramUpdate, command: ParsedCommand | None) -> None:
    chat_id = update.message.chat.id if update.message else None
    logger.info(
        "event=incoming_command update_id=%s chat_id=%s command_kind=%s",
        update.update_id,
        chat_id,
        command.kind if command else "unsupported",
    )
