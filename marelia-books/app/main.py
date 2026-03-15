from __future__ import annotations
from dotenv import load_dotenv

import logging

from fastapi import FastAPI, Request

from app.books.metadata import MetadataResolver
from app.config import configure_logging, get_settings
from app.notion.client import NotionClient
from app.openai.vision import OpenAIVisionClient
from app.services.book_service import BookService
from app.telegram.handler import (
    TelegramClient,
    TelegramUpdate,
    UpdateDeduplicator,
    log_incoming_command,
    parse_supported_command,
)

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

load_dotenv()

app = FastAPI(title="Marelia Books")

deduplicator = UpdateDeduplicator()
telegram_client = TelegramClient(settings.telegram_bot_token)
vision_client = OpenAIVisionClient(settings.openai_api_key)
notion_client = NotionClient(
    settings.notion_api_key,
    settings.notion_database_id,
    settings.notion_template_id,
)
metadata_resolver = MetadataResolver()
book_service = BookService(
    notion_client=notion_client,
    telegram_client=telegram_client,
    vision_client=vision_client,
    metadata_resolver=metadata_resolver,
    dry_run=settings.dry_run,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request) -> dict[str, bool]:
    payload = await request.json()
    update = TelegramUpdate.model_validate(payload)

    if deduplicator.is_duplicate(update.update_id):
        return {"ok": True}

    if not update.message:
        return {"ok": True}

    chat_id = update.message.chat.id
    if chat_id not in settings.allowed_chat_ids:
        logger.info("event=unauthorized_chat chat_id=%s", chat_id)
        return {"ok": True}

    command = parse_supported_command(update.message)
    log_incoming_command(update, command)

    if not command:
        await telegram_client.send_message(
            chat_id,
            "Unsupported input. Use 'Add Book <title>' or send a photo with caption 'Add Book'.",
        )
        return {"ok": True}

    try:
        if command.kind == "text" and command.title:
            result = await book_service.process_text_command(command.title)
        elif command.kind == "image" and command.file_id:
            result = await book_service.process_image_command(command.file_id)
        else:
            await telegram_client.send_message(chat_id, "Unsupported input.")
            return {"ok": True}

        await telegram_client.send_message(chat_id, result.message)
        return {"ok": True}
    except Exception as exc:
        logger.exception("event=processing_error error=%s", exc)
        await telegram_client.send_message(
            chat_id, "Failed to process book request. Please try again."
        )
        return {"ok": True}


@app.on_event("shutdown")
async def shutdown() -> None:
    await telegram_client.close()
    await vision_client.close()
    await notion_client.close()
    await metadata_resolver.close()
