from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from app.books.deduplication import find_matching_page, normalize_book_text
from app.books.metadata import BookCandidate, MetadataResolver, ResolvedBookMetadata, VisionBookExtraction
from app.notion.client import NotionClient
from app.gemini.enricher import GeminiEnricher
from app.gemini.vision import GeminiVisionClient, GeminiVisionQuotaError, GeminiVisionResponseError

logger = logging.getLogger(__name__)


class TelegramGateway(Protocol):
    async def download_file(self, file_id: str) -> tuple[bytes, str, str]: ...
    async def send_message(self, chat_id: int, text: str) -> None: ...


@dataclass
class ProcessResult:
    ok: bool
    message: str


class BookService:
    _SEARCH_CANDIDATE_LIMIT = 20

    def __init__(
        self,
        notion_client: NotionClient,
        telegram_client: TelegramGateway,
        vision_client: GeminiVisionClient,
        metadata_resolver: MetadataResolver,
        enricher: GeminiEnricher,
        dry_run: bool,
    ) -> None:
        self._notion = notion_client
        self._telegram = telegram_client
        self._vision = vision_client
        self._resolver = metadata_resolver
        self._enricher = enricher
        self._dry_run = dry_run
        self._pending: dict[int, list[BookCandidate]] = {}
        self._pending_input_mode: dict[int, str] = {}
        self._pending_search_title: dict[int, str] = {}

    def has_pending(self, chat_id: int) -> bool:
        return chat_id in self._pending

    def is_waiting_for_title(self, chat_id: int) -> bool:
        return self._pending_input_mode.get(chat_id) == "title"

    def is_waiting_for_photo(self, chat_id: int) -> bool:
        return self._pending_input_mode.get(chat_id) == "photo"

    def is_waiting_for_author(self, chat_id: int) -> bool:
        return self._pending_input_mode.get(chat_id) == "author"

    def start_addbook_mode(self, chat_id: int) -> None:
        self._pending_input_mode[chat_id] = "title"

    def start_scanbook_mode(self, chat_id: int) -> None:
        self._pending_input_mode[chat_id] = "photo"

    def start_author_mode(self, chat_id: int) -> None:
        self._pending_input_mode[chat_id] = "author"

    def clear_input_mode(self, chat_id: int) -> None:
        self._pending_input_mode.pop(chat_id, None)

    def _clear_pending_search(self, chat_id: int) -> None:
        self._pending_search_title.pop(chat_id, None)

    async def process_text_command(self, title: str, chat_id: int) -> ProcessResult:
        logger.info("Buscando '%s' en Open Library...", title)
        # Clear any previous pending selection for this chat
        self._pending.pop(chat_id, None)
        self.clear_input_mode(chat_id)
        self._clear_pending_search(chat_id)

        candidates = await self._resolver.search_candidates(title, limit=self._SEARCH_CANDIDATE_LIMIT)

        if not candidates:
            logger.info("Sin resultados en Open Library — usando título literal")
            resolved = ResolvedBookMetadata(title=title)
            return await self._upsert_book(resolved, chat_id=chat_id, requested_title=title)

        if len(candidates) == 1:
            logger.info("Un único resultado — procediendo directamente")
            resolved = await self._resolver.resolve_from_candidate(candidates[0])
            return await self._upsert_book(resolved, chat_id=chat_id, requested_title=title)

        self._pending[chat_id] = candidates
        self._pending_search_title[chat_id] = title
        self.start_author_mode(chat_id)
        logger.info("Pidiendo autor para afinar %d opciones [chat_id=%s]", len(candidates), chat_id)
        return ProcessResult(
            ok=True,
            message=(
                f"I found several matches for '{title}', Taviz. "
                "Send me the author name to narrow it down, or reply `skip` and I'll show the numbered list."
            ),
        )

    async def process_selection(self, chat_id: int, choice: int) -> ProcessResult:
        candidates = self._pending.pop(chat_id, None)
        self.clear_input_mode(chat_id)
        if not candidates or choice < 1 or choice > len(candidates):
            return ProcessResult(ok=False, message="Hmm, that doesn't match any of the options, Taviz. Try again with `/addbook <title>` 😉")

        selected = candidates[choice - 1]
        requested_title = self._pending_search_title.pop(chat_id, None)
        logger.info("Usuario seleccionó opción %d: '%s' — %s", choice, selected.title, selected.author)
        if requested_title and selected.author:
            resolved = await self._resolver.resolve(title=requested_title, author=selected.author)
        else:
            resolved = await self._resolver.resolve_from_candidate(selected)
        logger.info("Metadatos resueltos: '%s' — %s", resolved.title, resolved.author or "sin autor")
        logger.debug("Metadata completa:\n%s", "\n".join(f"  {k}: {v}" for k, v in resolved.model_dump().items()))
        return await self._upsert_book(resolved, chat_id=chat_id, requested_title=requested_title)

    async def process_author_refinement(self, chat_id: int, author_text: str) -> ProcessResult:
        candidates = self._pending.get(chat_id, [])
        if not candidates:
            self.clear_input_mode(chat_id)
            self._clear_pending_search(chat_id)
            return ProcessResult(ok=False, message="I lost the previous search, Taviz. Try `/addbook <title>` again.")

        if normalize_book_text(author_text) in {"skip", "omitir"}:
            self.clear_input_mode(chat_id)
            return ProcessResult(ok=True, message=self._format_candidate_options(candidates))

        narrowed = self._filter_candidates_by_author(candidates, author_text)
        logger.info(
            "Refinamiento por autor %r dejó %d opción(es) [chat_id=%s]",
            author_text,
            len(narrowed),
            chat_id,
        )

        if len(narrowed) == 1:
            self._pending.pop(chat_id, None)
            self.clear_input_mode(chat_id)
            selected = narrowed[0]
            requested_title = self._pending_search_title.pop(chat_id, None)
            if requested_title and selected.author:
                resolved = await self._resolver.resolve(title=requested_title, author=selected.author)
            else:
                resolved = await self._resolver.resolve_from_candidate(selected)
            logger.info("Autor refinó a una sola opción: '%s' — %s", resolved.title, resolved.author or "sin autor")
            return await self._upsert_book(resolved, chat_id=chat_id, requested_title=requested_title)

        if len(narrowed) > 1:
            self._pending[chat_id] = narrowed
            self.clear_input_mode(chat_id)
            return ProcessResult(ok=True, message=self._format_candidate_options(narrowed))

        return ProcessResult(
            ok=False,
            message=(
                "I couldn't narrow it down with that author, Taviz. "
                "Send another author name, reply `skip`, or choose from the full list with a number.\n\n"
                + self._format_candidate_options(candidates)
            ),
        )

    async def process_image_command(self, file_id: str, chat_id: int) -> ProcessResult:
        logger.info("Procesando imagen de portada")
        self.clear_input_mode(chat_id)
        image_bytes, mime_type, source_image_url = await self._telegram.download_file(file_id)
        logger.debug("Imagen descargada (%d bytes, %s)", len(image_bytes), mime_type)
        try:
            extraction = await self._vision.extract_book_data(image_bytes, mime_type)
        except GeminiVisionQuotaError:
            logger.warning("Gemini Vision sin cuota disponible — solicitando fallback manual por texto")
            return ProcessResult(
                ok=False,
                message=(
                    "Taviz, alcancé el limite de vision por ahora 😕 "
                    "Mientras se libera la cuota, puedes agregarlo con texto: `/addbook <titulo>`."
                ),
            )
        except GeminiVisionResponseError as exc:
            logger.warning("Gemini Vision devolvió una respuesta inválida — solicitando reintento: %s", exc)
            return ProcessResult(
                ok=False,
                message=(
                    "Taviz, no pude leer bien esa portada porque Gemini devolvió una respuesta incompleta 😕 "
                    "Prueba enviando la foto otra vez o usa `/addbook <titulo>`."
                ),
            )

        logger.info(
            "Gemini detectó: '%s' por %s (confianza %.0f%%)",
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
        if extraction.subtitle and not resolved.subtitle:
            resolved.subtitle = extraction.subtitle
        if extraction.series_or_edition and not resolved.series:
            resolved.series = extraction.series_or_edition
        if not resolved.cover_url:
            resolved.cover_url = source_image_url

        logger.info(
            "Metadatos resueltos: '%s' — %s",
            resolved.title,
            resolved.author or "sin autor",
        )
        logger.debug("Metadata completa:\n%s", "\n".join(f"  {k}: {v}" for k, v in resolved.model_dump().items()))
        return await self._upsert_book(resolved, chat_id=chat_id)

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
                    "Could you confirm by sending `/addbook <title>`, Taviz? 🙏"
                ),
            )

        if not extraction.title:
            return ProcessResult(ok=False, message="I couldn't make out the title, Taviz 😕 Try a clearer photo of the cover!")

        return None

    async def _upsert_book(
        self,
        metadata: ResolvedBookMetadata,
        *,
        chat_id: int,
        requested_title: str | None = None,
    ) -> ProcessResult:
        if requested_title:
            requested_norm = normalize_book_text(requested_title)
            metadata_norm = normalize_book_text(metadata.title)
            title_es_norm = normalize_book_text(metadata.title_es)
            if requested_norm and requested_norm != metadata_norm and title_es_norm in {"", metadata_norm}:
                metadata.title_es = requested_title

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

        await self._telegram.send_message(
            chat_id,
            f"I'm adding *{metadata.title_es or metadata.title}* to Notion now, Taviz. This can take a little bit ⏳",
        )
        metadata = await self._enricher.enrich(metadata)
        page_id = await self._notion.create_book_page(metadata)
        logger.info("Libro creado en Notion [id=%s]", page_id)
        return ProcessResult(ok=True, message=f"Done, Taviz! ✨\nI\'ve added *{metadata.title}* to your reading list 📚")

    @staticmethod
    def _filter_candidates_by_author(candidates: list[BookCandidate], author_text: str) -> list[BookCandidate]:
        target = normalize_book_text(author_text)
        if not target:
            return candidates

        matched: list[BookCandidate] = []
        for candidate in candidates:
            candidate_author = normalize_book_text(candidate.author)
            if candidate_author and (target in candidate_author or candidate_author in target):
                matched.append(candidate)
        return matched

    @staticmethod
    def _format_candidate_options(candidates: list[BookCandidate]) -> str:
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
        lines.append(f"\nWhich one is it? Reply with a number between 1 and {len(candidates)} 😊")
        return "\n".join(lines)
