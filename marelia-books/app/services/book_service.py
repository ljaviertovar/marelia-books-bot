from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from app.books.deduplication import find_matching_page
from app.books.metadata import MetadataResolver, ResolvedBookMetadata, VisionBookExtraction
from app.notion.client import NotionClient, get_page_author, get_page_title
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

    async def process_text_command(self, title: str) -> ProcessResult:
        resolved = await self._resolver.resolve(title=title)
        logger.info(
            "event=metadata_resolution title=%s author=%s categories=%s",
            resolved.title,
            resolved.author,
            ",".join(resolved.categories),
        )
        return await self._upsert_book(resolved)

    async def process_image_command(self, file_id: str) -> ProcessResult:
        image_bytes, mime_type = await self._telegram.download_file(file_id)
        extraction = await self._vision.extract_book_data(image_bytes, mime_type)

        logger.info(
            "event=extracted_vision title=%s author=%s confidence=%s",
            extraction.title,
            ",".join(extraction.authors),
            extraction.confidence,
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
            "event=metadata_resolution title=%s author=%s categories=%s",
            resolved.title,
            resolved.author,
            ",".join(resolved.categories),
        )
        return await self._upsert_book(resolved)

    def _validate_vision(self, extraction: VisionBookExtraction) -> ProcessResult | None:
        if not extraction.is_book_cover:
            reason = extraction.reason_if_not_book or "Image does not look like a book cover."
            return ProcessResult(ok=False, message=f"Could not add: {reason}")

        if extraction.confidence < 0.60:
            return ProcessResult(ok=False, message="Image confidence too low. Please send a clearer book cover photo.")

        if 0.60 <= extraction.confidence < 0.85:
            title = extraction.title or "Unknown"
            author = extraction.authors[0] if extraction.authors else "Unknown"
            return ProcessResult(
                ok=False,
                message=(
                    f"I detected '{title}' by {author} with medium confidence ({extraction.confidence:.2f}). "
                    "Please confirm by sending: Add Book <title>"
                ),
            )

        if not extraction.title:
            return ProcessResult(ok=False, message="Could not detect book title. Please send a clearer image.")

        return None

    async def _upsert_book(self, metadata: ResolvedBookMetadata) -> ProcessResult:
        candidates = await self._notion.query_candidate_books(metadata.title)
        existing = find_matching_page(
            candidates,
            metadata.title,
            metadata.author,
            get_title=get_page_title,
            get_author=get_page_author,
        )

        logger.info("event=duplicate_detection found=%s candidates=%s", bool(existing), len(candidates))

        if existing:
            if self._dry_run:
                logger.info("event=action_update dry_run=true")
                return ProcessResult(ok=True, message=f"[DRY_RUN] Book already exists. Missing fields would be updated: {metadata.title}")

            changed = await self._notion.update_book_page_missing(existing, metadata)
            logger.info("event=action_update changed=%s", changed)
            if changed:
                return ProcessResult(ok=True, message=f"Updated missing fields for: {metadata.title}")
            return ProcessResult(ok=True, message=f"Book already up to date: {metadata.title}")

        if self._dry_run:
            logger.info("event=action_create dry_run=true")
            return ProcessResult(ok=True, message=f"[DRY_RUN] Book would be created: {metadata.title}")

        page_id = await self._notion.create_book_page(metadata)
        logger.info("event=action_create page_id=%s", page_id)
        return ProcessResult(ok=True, message=f"Added: {metadata.title}")
