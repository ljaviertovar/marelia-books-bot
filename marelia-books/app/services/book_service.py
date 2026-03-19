from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from app.books.deduplication import find_matching_page
from app.books.metadata import BookCandidate, MetadataResolver, ResolvedBookMetadata, VisionBookExtraction
from app.notion.client import NotionClient
from app.openai.vision import OpenAIVisionClient

logger = logging.getLogger(__name__)


class TelegramGateway(Protocol):
    async def download_file(self, file_id: str) -> tuple[bytes, str]: ...


@dataclass
class ProcessResult:
    ok: bool
    message: str


class BookService:
    def __init__(
        self,
        notion_client: NotionClient,
        telegram_client: TelegramGateway,
        vision_client: OpenAIVisionClient,
        metadata_resolver: MetadataResolver,
        dry_run: bool,
    ) -> None:
        self._notion = notion_client
        self._telegram = telegram_client
        self._vision = vision_client
        self._resolver = metadata_resolver
        self._dry_run = dry_run
        self._pending: dict[int, list[BookCandidate]] = {}

    def has_pending(self, chat_id: int) -> bool:
        return chat_id in self._pending

    async def process_text_command(self, title: str, chat_id: int) -> ProcessResult:
        logger.info("Buscando '%s' en Open Library...", title)
        # Clear any previous pending selection for this chat
        self._pending.pop(chat_id, None)

        candidates = await self._resolver.search_candidates(title, limit=3)

        if not candidates:
            logger.info("Sin resultados en Open Library — usando título literal")
            resolved = ResolvedBookMetadata(title=title)
            return await self._upsert_book(resolved)

        if len(candidates) == 1:
            logger.info("Un único resultado — procediendo directamente")
            resolved = await self._resolver.resolve_from_candidate(candidates[0])
            return await self._upsert_book(resolved)

        self._pending[chat_id] = candidates
        lines = ["Here's what I found for you, Taviz! 📚\n"]
        for i, c in enumerate(candidates, 1):
            author = c.author or "unknown author"
            details: list[str] = []
            if c.year:
                details.append(str(c.year))
            if c.publisher:
                details.append(c.publisher)
            if c.language:
                details.append(c.language)
            extra = f"  ({', '.join(details)})" if details else ""
            lines.append(f"{i}. {c.title} — {author}{extra}")
        lines.append("\nWhich one is it? Reply with 1, 2 or 3 😊")
        logger.info("Mostrando %d opciones al usuario [chat_id=%s]", len(candidates), chat_id)
        return ProcessResult(ok=True, message="\n".join(lines))

    async def process_selection(self, chat_id: int, choice: int) -> ProcessResult:
        candidates = self._pending.pop(chat_id, None)
        if not candidates or choice < 1 or choice > len(candidates):
            return ProcessResult(ok=False, message="Hmm, that doesn't match any of the options, Taviz. Try searching again with 'Add Book <title>' 😉")

        selected = candidates[choice - 1]
        logger.info("Usuario seleccionó opción %d: '%s' — %s", choice, selected.title, selected.author)
        resolved = await self._resolver.resolve_from_candidate(selected)
        logger.info("Metadatos resueltos: '%s' — %s", resolved.title, resolved.author or "sin autor")
        logger.debug("Metadata completa:\n%s", "\n".join(f"  {k}: {v}" for k, v in resolved.model_dump().items()))
        return await self._upsert_book(resolved)

    async def process_image_command(self, file_id: str) -> ProcessResult:
        logger.info("Procesando imagen de portada")
        image_bytes, mime_type = await self._telegram.download_file(file_id)
        logger.debug("Imagen descargada (%d bytes, %s)", len(image_bytes), mime_type)
        extraction = await self._vision.extract_book_data(image_bytes, mime_type)

        logger.info(
            "OpenAI detectó: '%s' por %s (confianza %.0f%%)",
            extraction.title or "desconocido",
            extraction.authors[0] if extraction.authors else "autor desconocido",
            extraction.confidence * 100,
        )

        check = self._validate_vision(extraction)
        if check is not None:
            return check

        title = extraction.title or ""
        author = extraction.authors[0] if extraction.authors else None
        resolved = await self._resolver.resolve(title=title, author=author)
        if extraction.series_or_edition and not resolved.series:
            resolved.series = extraction.series_or_edition

        logger.info(
            "Metadatos resueltos: '%s' — %s",
            resolved.title,
            resolved.author or "sin autor",
        )
        logger.debug("Metadata completa:\n%s", "\n".join(f"  {k}: {v}" for k, v in resolved.model_dump().items()))
        return await self._upsert_book(resolved)

    def _validate_vision(self, extraction: VisionBookExtraction) -> ProcessResult | None:
        if not extraction.is_book_cover:
            reason = extraction.reason_if_not_book or "that doesn't look like a book cover to me."
            return ProcessResult(ok=False, message=f"Oops, Taviz! I couldn't add it — {reason}")

        if extraction.confidence < 0.60:
            return ProcessResult(ok=False, message="Taviz, the image is a bit blurry 😕 Could you send a clearer photo of the cover?")

        if 0.60 <= extraction.confidence < 0.85:
            title = extraction.title or "Unknown"
            author = extraction.authors[0] if extraction.authors else "Unknown"
            return ProcessResult(
                ok=False,
                message=(
                    f"I think this might be '{title}' by {author}, but I\'m not totally sure (confidence: {extraction.confidence:.0%}). "
                    "Could you confirm by sending: Add Book <title>, Taviz? 🙏"
                ),
            )

        if not extraction.title:
            return ProcessResult(ok=False, message="I couldn't make out the title, Taviz 😕 Try a clearer photo of the cover!")

        return None

    async def _upsert_book(self, metadata: ResolvedBookMetadata) -> ProcessResult:
        candidates = await self._notion.query_candidate_books(metadata.title)
        existing = find_matching_page(candidates, metadata.title, metadata.author)

        logger.info("Duplicado: %s (%d candidato(s))", "sí" if existing else "no", len(candidates))

        if existing:
            if self._dry_run:
                logger.info("[DRY RUN] El libro ya existe — se actualizarían campos")
                return ProcessResult(ok=True, message=f"[DRY RUN] '{metadata.title}' is already in your Notion, Taviz! I would update the missing fields.")

            changed = await self._notion.update_book_page_missing(existing, metadata)
            logger.info("Campos faltantes actualizados: %s", changed)
            if changed:
                return ProcessResult(ok=True, message=f"I found '{metadata.title}' already in your list, Taviz! I went ahead and filled in some missing details ✨")
            return ProcessResult(ok=True, message=f"'{metadata.title}' is already in your reading list, Taviz! Everything looks up to date 📚")

        if self._dry_run:
            logger.info("[DRY RUN] Se crearía el libro en Notion")
            return ProcessResult(ok=True, message=f"[DRY RUN] I would add '{metadata.title}' to your Notion list, Taviz!")

        page_id = await self._notion.create_book_page(metadata)
        logger.info("Libro creado en Notion [id=%s]", page_id)
        return ProcessResult(ok=True, message=f"Done, Taviz! ✨ I\'ve added '{metadata.title}' to your reading list 📚")
