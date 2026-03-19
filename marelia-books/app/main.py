from __future__ import annotations

import logging

from fastapi import FastAPI, Request

from app.books.metadata import MetadataResolver
from app.config import configure_logging, get_settings
from app.notion.client import NotionClient
from app.gemini.vision import GeminiVisionClient
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

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

settings = get_settings()

app = FastAPI(title="Marelia Books")

deduplicator = UpdateDeduplicator()
telegram_client = TelegramClient(settings.telegram_bot_token)
vision_client = GeminiVisionClient(settings.gemini_api_key)
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
    logger.debug("Update recibido [id=%s]", update.update_id)

    if deduplicator.is_duplicate(update.update_id):
        logger.debug("Update duplicado, ignorado [id=%s]", update.update_id)
        return {"ok": True}

    if not update.message:
        logger.debug("Sin mensaje en el update [id=%s]", update.update_id)
        return {"ok": True}

    chat_id = update.message.chat.id
    if chat_id not in settings.allowed_chat_ids:
        logger.warning("Chat no autorizado [chat_id=%s]", chat_id)
        return {"ok": True}

    command = parse_supported_command(update.message)

    try:
        if not command:
            text = (update.message.text or "").strip()
            if text in ("1", "2", "3") and book_service.has_pending(chat_id):
                logger.info("Selection received: %s [chat_id=%s]", text, chat_id)
                result = await book_service.process_selection(chat_id, int(text))
            else:
                log_incoming_command(update, command)
                logger.info("Comando no reconocido [chat_id=%s]", chat_id)
                await telegram_client.send_message(
                    chat_id,
                    "Hey Taviz! 👋 I didn't quite get that. Send 'Add Book <title>' or a photo of the cover with the caption 'Add Book'.",
                )
                return {"ok": True}
        elif command.kind == "text" and command.title:
            log_incoming_command(update, command)
            result = await book_service.process_text_command(command.title, chat_id)
        elif command.kind == "image" and command.file_id:
            log_incoming_command(update, command)
            result = await book_service.process_image_command(command.file_id)
        else:
            log_incoming_command(update, command)
            await telegram_client.send_message(chat_id, "Hey Taviz, I didn't understand that one 🤔 Try 'Add Book <title>'.")
            return {"ok": True}

        logger.info("Respuesta enviada [chat_id=%s]: %s", chat_id, result.message)
        await telegram_client.send_message(chat_id, result.message)
        return {"ok": True}
    except Exception as exc:
        logger.exception("Error al procesar el mensaje: %s", exc)
        await telegram_client.send_message(
            chat_id, "Oops Taviz, something went wrong on my end 😕 Please try again in a moment!"
        )
        return {"ok": True}


@app.on_event("shutdown")
async def shutdown() -> None:
    await telegram_client.close()
    await vision_client.close()
    await notion_client.close()
    await metadata_resolver.close()
