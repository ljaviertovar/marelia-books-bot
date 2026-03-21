import asyncio

from app.books.metadata import BookCandidate
from app.books.metadata import ResolvedBookMetadata, VisionBookExtraction
from app.services.book_service import BookService


class _Dummy:
    pass


class _CaptureService(BookService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_metadata = None

    async def _upsert_book(self, metadata, *, chat_id, requested_title=None):
        self.captured_metadata = metadata
        return type("Result", (), {"ok": True, "message": "ok"})()


def test_input_modes_switch_and_clear_per_chat():
    service = BookService(
        notion_client=_Dummy(),
        telegram_client=_Dummy(),
        vision_client=_Dummy(),
        metadata_resolver=_Dummy(),
        enricher=_Dummy(),
        dry_run=True,
        contact_name="Taviz",
    )

    service.start_addbook_mode(123)
    assert service.is_waiting_for_title(123) is True
    assert service.is_waiting_for_photo(123) is False

    service.start_scanbook_mode(123)
    assert service.is_waiting_for_title(123) is False
    assert service.is_waiting_for_photo(123) is True

    service.clear_input_mode(123)
    assert service.is_waiting_for_title(123) is False
    assert service.is_waiting_for_photo(123) is False


def test_author_mode_and_filtering():
    service = BookService(
        notion_client=_Dummy(),
        telegram_client=_Dummy(),
        vision_client=_Dummy(),
        metadata_resolver=_Dummy(),
        enricher=_Dummy(),
        dry_run=True,
        contact_name="Taviz",
    )

    service.start_author_mode(123)
    assert service.is_waiting_for_author(123) is True

    candidates = [
        BookCandidate(title="Artemisa", author="unknown author"),
        BookCandidate(title="Artemisa", author="Andy Weir"),
        BookCandidate(title="Artemisa", author="Carlos Sabat Ercasty"),
    ]

    filtered = service._filter_candidates_by_author(candidates, "Andy")

    assert len(filtered) == 1
    assert filtered[0].author == "Andy Weir"


def test_process_image_command_keeps_vision_title_when_openlibrary_mismatches():
    class _Telegram:
        async def download_file(self, file_id):
            return b"img", "image/jpeg", "https://example.com/cover.jpg"

    class _Vision:
        async def extract_book_data(self, image_bytes, mime_type):
            return VisionBookExtraction(
                is_book_cover=True,
                title="El monje que vendio su ferrari",
                subtitle=None,
                authors=["Robin S. Sharma"],
                series_or_edition=None,
                language="espanol",
                confidence=0.98,
                reason_if_not_book=None,
                raw_visible_text=None,
            )

    class _Resolver:
        async def resolve(self, *, title, author=None):
            return ResolvedBookMetadata(
                title="Discover Your Destiny",
                author="Robin S. Sharma",
                cover_url=None,
                categories=["Self-development"],
                link="https://openlibrary.org/works/OL123W",
            )

    service = _CaptureService(
        notion_client=_Dummy(),
        telegram_client=_Telegram(),
        vision_client=_Vision(),
        metadata_resolver=_Resolver(),
        enricher=_Dummy(),
        dry_run=True,
        contact_name="Taviz",
    )

    asyncio.run(service.process_image_command("file-123", 123))

    assert service.captured_metadata is not None
    assert service.captured_metadata.title == "El monje que vendio su ferrari"
    assert service.captured_metadata.author == "Robin S. Sharma"
    assert service.captured_metadata.cover_url == "https://example.com/cover.jpg"
    assert service.captured_metadata.categories == ["Self-development"]
    assert service.captured_metadata.link == "https://openlibrary.org/works/OL123W"


def test_process_image_command_prefers_uploaded_cover_over_openlibrary_cover():
    class _Telegram:
        async def download_file(self, file_id):
            return b"img", "image/jpeg", "https://example.com/uploaded-cover.jpg"

    class _Vision:
        async def extract_book_data(self, image_bytes, mime_type):
            return VisionBookExtraction(
                is_book_cover=True,
                title="Dune",
                subtitle=None,
                authors=["Frank Herbert"],
                series_or_edition=None,
                language="english",
                confidence=0.98,
                reason_if_not_book=None,
                raw_visible_text=None,
            )

    class _Resolver:
        async def resolve(self, *, title, author=None):
            return ResolvedBookMetadata(
                title="Dune",
                author="Frank Herbert",
                cover_url="https://example.com/openlibrary-cover.jpg",
            )

    service = _CaptureService(
        notion_client=_Dummy(),
        telegram_client=_Telegram(),
        vision_client=_Vision(),
        metadata_resolver=_Resolver(),
        enricher=_Dummy(),
        dry_run=True,
        contact_name="Taviz",
    )

    asyncio.run(service.process_image_command("file-456", 123))

    assert service.captured_metadata is not None
    assert service.captured_metadata.cover_url == "https://example.com/uploaded-cover.jpg"


def test_process_image_command_keeps_original_title_when_openlibrary_found_translation():
    class _Telegram:
        async def download_file(self, file_id):
            return b"img", "image/jpeg", "https://example.com/foundation-cover.jpg"

    class _Vision:
        async def extract_book_data(self, image_bytes, mime_type):
            return VisionBookExtraction(
                is_book_cover=True,
                title="Hacia la Fundacion",
                subtitle=None,
                authors=["Isaac Asimov"],
                series_or_edition=None,
                language="espanol",
                confidence=0.99,
                reason_if_not_book=None,
                raw_visible_text=None,
            )

    class _Resolver:
        async def resolve(self, *, title, author=None):
            return ResolvedBookMetadata(
                title="Forward the Foundation",
                title_es="Hacia la Fundacion",
                author="Isaac Asimov",
                language="ingles",
                cover_url="https://example.com/openlibrary-foundation.jpg",
                link="https://openlibrary.org/works/OL27448W",
                categories=["Sci-Fi"],
            )

    service = _CaptureService(
        notion_client=_Dummy(),
        telegram_client=_Telegram(),
        vision_client=_Vision(),
        metadata_resolver=_Resolver(),
        enricher=_Dummy(),
        dry_run=True,
        contact_name="Taviz",
    )

    asyncio.run(service.process_image_command("file-789", 123))

    assert service.captured_metadata is not None
    assert service.captured_metadata.title == "Forward the Foundation"
    assert service.captured_metadata.title_es == "Hacia la Fundacion"
    assert service.captured_metadata.categories == ["Sci-Fi"]
    assert service.captured_metadata.link == "https://openlibrary.org/works/OL27448W"
    assert service.captured_metadata.cover_url == "https://example.com/foundation-cover.jpg"


def test_existing_book_is_enriched_before_missing_data_update_and_sends_progress_message():
    class _Telegram:
        def __init__(self) -> None:
            self.messages: list[tuple[int, str]] = []

        async def send_message(self, chat_id, text):
            self.messages.append((chat_id, text))

    class _Notion:
        def __init__(self) -> None:
            self.updated_metadata = None

        async def query_candidate_books(self, title):
            return [
                type(
                    "Record",
                    (),
                    {
                        "page_id": "page-1",
                        "title": "Dune",
                        "author": "Frank Herbert",
                        "series": None,
                        "_raw_properties": {},
                    },
                )()
            ]

        async def update_book_page_missing(self, record, metadata):
            self.updated_metadata = metadata
            return True

    class _Enricher:
        async def enrich(self, metadata):
            return metadata.model_copy(update={"tagline": "Dune es una novela de ciencia ficcion de Frank Herbert."})

    notion = _Notion()
    telegram = _Telegram()
    service = BookService(
        notion_client=notion,
        telegram_client=telegram,
        vision_client=_Dummy(),
        metadata_resolver=_Dummy(),
        enricher=_Enricher(),
        dry_run=False,
        contact_name="Taviz",
    )

    result = asyncio.run(
        service._upsert_book(
            ResolvedBookMetadata(title="Dune", author="Frank Herbert"),
            chat_id=123,
        )
    )

    assert notion.updated_metadata is not None
    assert notion.updated_metadata.tagline == "Dune es una novela de ciencia ficcion de Frank Herbert."
    assert telegram.messages
    assert telegram.messages[0][0] == 123
    assert "Taviz" in telegram.messages[0][1]
    assert "is already in your reading list" in telegram.messages[0][1]
    assert "I'll update the missing information for you now." in telegram.messages[0][1]
    assert "I've updated the missing details for" in result.message
