from __future__ import annotations

import html
import logging
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from app.books.deduplication import find_matching_page, normalize_book_text
from app.books.metadata import BookCandidate, MetadataResolver, ResolvedBookMetadata, VisionBookExtraction, sanitize_series_name
from app.notion.client import NotionClient
from app.gemini.enricher import GeminiEnricher
from app.gemini.vision import GeminiVisionClient, GeminiVisionQuotaError, GeminiVisionResponseError

logger = logging.getLogger(__name__)


class TelegramGateway(Protocol):
    async def download_file(self, file_id: str) -> tuple[bytes, str, str]: ...
    async def send_message(self, chat_id: int, text: str) -> None: ...


@dataclass
class ProcessResult:
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
        contact_name: str,
    ) -> None:
        self._notion = notion_client
        self._telegram = telegram_client
        self._vision = vision_client
        self._resolver = metadata_resolver
        self._enricher = enricher
        self._dry_run = dry_run
        self._contact_name = contact_name
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
        await self._telegram.send_message(chat_id, self._search_started_message())
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
            message=(
                "🔎 "
                f"I found a few promising matches for {self._book(title)}.\n\n"
                "Send me the author name and I'll narrow it down for you, or send <code>skip</code> and I'll show you the numbered list."
            ),
        )

    async def process_selection(self, chat_id: int, choice: int) -> ProcessResult:
        candidates = self._pending.pop(chat_id, None)
        self.clear_input_mode(chat_id)
        if not candidates or choice < 1 or choice > len(candidates):
            return ProcessResult(
                message=(
                    f"🔢 {html.escape(self._contact_name)}, that number doesn't match any of the options I showed you.\n"
                    "Please try again with <code>/addbook &lt;title&gt;</code>."
                ),
            )

        selected = candidates[choice - 1]
        requested_title = self._pending_search_title.pop(chat_id, None)
        logger.info("Usuario seleccionó opción %d: '%s' — %s", choice, selected.title, selected.author)
        resolved = await self._resolve_candidate(selected, requested_title=requested_title)
        logger.info("Metadatos resueltos: '%s' — %s", resolved.title, resolved.author or "sin autor")
        logger.debug("Metadata completa:\n%s", "\n".join(f"  {k}: {v}" for k, v in resolved.model_dump().items()))
        return await self._upsert_book(resolved, chat_id=chat_id, requested_title=requested_title)

    async def process_author_refinement(self, chat_id: int, author_text: str) -> ProcessResult:
        candidates = self._pending.get(chat_id, [])
        if not candidates:
            self.clear_input_mode(chat_id)
            self._clear_pending_search(chat_id)
            return ProcessResult(
                message=(
                    f"🧭 {html.escape(self._contact_name)}, I can't find that previous search anymore.\n"
                    "Please try <code>/addbook &lt;title&gt;</code> again."
                ),
            )

        if normalize_book_text(author_text) in {"skip", "omitir"}:
            self.clear_input_mode(chat_id)
            return ProcessResult(message=self._format_candidate_options(candidates))

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
            resolved = await self._resolve_candidate(selected, requested_title=requested_title)
            logger.info("Autor refinó a una sola opción: '%s' — %s", resolved.title, resolved.author or "sin autor")
            return await self._upsert_book(resolved, chat_id=chat_id, requested_title=requested_title)

        if len(narrowed) > 1:
            self._pending[chat_id] = narrowed
            self.clear_input_mode(chat_id)
            return ProcessResult(message=self._format_candidate_options(narrowed))

        return ProcessResult(
            message=(
                f"🕵️ {html.escape(self._contact_name)}, I still couldn't narrow it down with that author.\n"
                "Send another author name, send <code>skip</code>, or choose from the full list with a number.\n\n"
                + self._format_candidate_options(candidates)
            ),
        )

    async def _resolve_candidate(
        self,
        candidate: BookCandidate,
        *,
        requested_title: str | None = None,
    ) -> ResolvedBookMetadata:
        if requested_title and candidate.author:
            return await self._resolver.resolve(title=requested_title, author=candidate.author)
        return await self._resolver.resolve_from_candidate(candidate)

    async def process_image_command(self, file_id: str, chat_id: int) -> ProcessResult:
        logger.info("Procesando imagen de portada")
        self.clear_input_mode(chat_id)
        await self._telegram.send_message(chat_id, self._scan_started_message())
        image_bytes, mime_type, source_image_url = await self._telegram.download_file(file_id)
        logger.debug("Imagen descargada (%d bytes, %s)", len(image_bytes), mime_type)
        try:
            extraction = await self._vision.extract_book_data(image_bytes, mime_type)
        except GeminiVisionQuotaError:
            logger.warning("Gemini Vision sin cuota disponible — solicitando fallback manual por texto")
            return ProcessResult(
                message=(
                    "👁️🚫 I've reached the vision limit for now.\n\n"
                    "You can still add the book with <code>/addbook &lt;title&gt;</code>."
                ),
            )
        except GeminiVisionResponseError as exc:
            logger.warning("Gemini Vision devolvió una respuesta inválida — solicitando reintento: %s", exc)
            return ProcessResult(
                message=(
                    f"📸 {html.escape(self._contact_name)}, I couldn't read that cover properly because Gemini returned an incomplete response.\n"
                    "Please send the photo again, or use <code>/addbook &lt;title&gt;</code>."
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
        if self._titles_match(title, resolved.title):
            resolved = resolved.model_copy(
                update={
                    "title": self._matching_title_display(title, resolved.title),
                    "author": author or resolved.author,
                }
            )
        else:
            if self._should_keep_resolved_original_title(extraction, resolved):
                logger.info(
                    "Open Library parece haber resuelto el título original; conservando original y usando la portada como título en español [%r -> %r]",
                    title,
                    resolved.title,
                )
                resolved = resolved.model_copy(
                    update={
                        "title_es": title,
                        "subtitle": extraction.subtitle or resolved.subtitle,
                        "author": author or resolved.author,
                        "series": sanitize_series_name(extraction.series_or_edition) or sanitize_series_name(resolved.series),
                        "language": extraction.language or resolved.language,
                    }
                )
            else:
                logger.warning(
                    "Open Library devolvió un título distinto al detectado por visión; usando fallback de portada [%r -> %r]",
                    title,
                    resolved.title,
                )
                resolved = resolved.model_copy(
                    update={
                        "title": title,
                        "subtitle": extraction.subtitle or resolved.subtitle,
                        "author": author or resolved.author,
                        "series": sanitize_series_name(extraction.series_or_edition) or sanitize_series_name(resolved.series),
                        "language": extraction.language or resolved.language,
                    }
                )
        if extraction.subtitle and not resolved.subtitle:
            resolved.subtitle = extraction.subtitle
        resolved.series = sanitize_series_name(resolved.series)
        if extraction.series_or_edition and not resolved.series:
            resolved.series = sanitize_series_name(extraction.series_or_edition)
        # For scanned books, the user's photo is the most reliable cover source.
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
            return ProcessResult(message=f"🖼️ {html.escape(self._contact_name)}, I couldn't add that image for you.\n{html.escape(reason)}")

        if extraction.confidence < 0.60:
            return ProcessResult(
                message=(
                    f"📸 {html.escape(self._contact_name)}, the photo looks a little blurry.\n"
                    "Please send a clearer photo of the cover."
                ),
            )

        if 0.60 <= extraction.confidence < 0.85:
            title = extraction.title or "Unknown"
            author = extraction.authors[0] if extraction.authors else "Unknown"
            return ProcessResult(
                message=(
                    f"📕 {html.escape(self._contact_name)}, I think this might be {self._book(title)} by {html.escape(author)}, "
                    f"but I'm not completely sure ({extraction.confidence:.0%} confidence).\n"
                    "Could you confirm it with <code>/addbook &lt;title&gt;</code>?"
                ),
            )

        if not extraction.title:
            return ProcessResult(
                message=(
                    f"📸 {html.escape(self._contact_name)}, I couldn't make out the title clearly.\n"
                    "Please try a clearer photo of the cover."
                ),
            )

        return None

    async def _upsert_book(
        self,
        metadata: ResolvedBookMetadata,
        *,
        chat_id: int,
        requested_title: str | None = None,
    ) -> ProcessResult:
        metadata.title = self._display_title(metadata.title)
        if metadata.title_es:
            metadata.title_es = self._display_title(metadata.title_es)

        if requested_title:
            requested_norm = normalize_book_text(requested_title)
            metadata_norm = normalize_book_text(metadata.title)
            title_es_norm = normalize_book_text(metadata.title_es)
            if requested_norm and requested_norm != metadata_norm and title_es_norm in {"", metadata_norm}:
                metadata.title_es = self._display_title(requested_title)

        candidates = await self._notion.query_candidate_books(metadata.title)
        existing = find_matching_page(candidates, metadata.title, metadata.author)

        logger.info("Duplicado: %s (%d candidato(s))", "sí" if existing else "no", len(candidates))

        if existing:
            if self._dry_run:
                logger.info("[DRY RUN] El libro ya existe — se actualizarían campos")
                return ProcessResult(
                    message=(
                        f"🧪 {html.escape(self._contact_name)}, [DRY RUN] {self._book(metadata.title)} is already in your reading list.\n"
                        "I'd fill in the missing details for you."
                    ),
                )

            metadata = await self._enricher.enrich(metadata)
            metadata.series = sanitize_series_name(metadata.series)
            changed = await self._notion.update_book_page_missing(existing, metadata)
            logger.info("Campos faltantes actualizados: %s", changed)
            if changed:
                return ProcessResult(
                    message=(
                        f"✅ Done!, {html.escape(self._contact_name)}\n\n"
                        f"📕 {self._book(metadata.title)} was already in your Reading List.\n"
                        "I've just updated the missing information."
                    ),
                )
            return ProcessResult(
                message=(
                    f"✅ Done!, {html.escape(self._contact_name)}\n\n"
                    f"📕{self._book(metadata.title)} is already up to date in your Reading List."
                ),
            )

        if self._dry_run:
            logger.info("[DRY RUN] Se crearía el libro en Notion")
            return ProcessResult(
                message=(
                    f"🧪 {html.escape(self._contact_name)}, [DRY RUN] I'd add {self._book(metadata.title)} to your reading list."
                ),
            )

        metadata = await self._enricher.enrich(metadata)
        metadata.series = sanitize_series_name(metadata.series)
        page_id = await self._notion.create_book_page(metadata)
        logger.info("Libro creado en Notion [id=%s]", page_id)
        return ProcessResult(
            message=(
                f"✅ Done!, {html.escape(self._contact_name)}\n\n"
                f"I've added 📕 {self._book(metadata.title)} to your reading list."
            ),
        )

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

    def _format_candidate_options(self, candidates: list[BookCandidate]) -> str:
        name = html.escape(self._contact_name)
        lines = ["🔎 here are the closest matches I found:\n"]
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
            lines.append(f"{i}. <b>{html.escape(c.title)}</b> — {html.escape(author)}{html.escape(extra)}")
        lines.append(f"\n🔢 Send me a number between 1 and {len(candidates)}.")
        return "\n".join(lines)

    @staticmethod
    def _book(title: str) -> str:
        return f"<b>{html.escape(title)}</b>"

    @staticmethod
    def _scan_started_message() -> str:
        return (
            "⏳ I'm analyzing that book for you now...\n\n"
            "Give me a moment while I scan the photo and prepare the book data."
        )

    @staticmethod
    def _search_started_message() -> str:
        return (
            "⏳ I'm looking for that book for you now...\n\n"
            "Give me a moment while I search for the best match and prepare the book data."
        )

    @staticmethod
    def _titles_match(detected_title: str | None, resolved_title: str | None) -> bool:
        left = BookService._comparison_title(detected_title)
        right = BookService._comparison_title(resolved_title)
        if not left or not right:
            return True
        if left == right:
            return True
        return left in right or right in left

    @staticmethod
    def _comparison_title(value: str | None) -> str:
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKD", value)
        without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        cleaned = without_accents.upper().strip()
        cleaned = cleaned.replace(":", " ").replace("-", " ")
        return " ".join(cleaned.split())

    @staticmethod
    def _display_title(value: str | None) -> str:
        if not value:
            return ""
        text = " ".join(value.strip().split())
        return text.title()

    @classmethod
    def _matching_title_display(cls, detected_title: str | None, resolved_title: str | None) -> str:
        if detected_title:
            return cls._display_title(detected_title)
        return cls._display_title(resolved_title)

    @staticmethod
    def _should_keep_resolved_original_title(
        extraction: VisionBookExtraction,
        resolved: ResolvedBookMetadata,
    ) -> bool:
        detected_title = normalize_book_text(extraction.title)
        resolved_title_es = normalize_book_text(resolved.title_es)
        detected_language = normalize_book_text(extraction.language)
        resolved_language = normalize_book_text(resolved.language)

        if not detected_title or not resolved_title_es or detected_title != resolved_title_es:
            return False

        # If the scanned cover clearly has its own subtitle but Open Library does not,
        # it is more likely that Open Library matched the wrong book.
        if extraction.subtitle and not resolved.subtitle:
            return False

        if detected_language and resolved_language and detected_language != resolved_language:
            return True

        return False
