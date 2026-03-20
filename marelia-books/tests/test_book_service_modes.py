from app.books.metadata import BookCandidate
from app.services.book_service import BookService


class _Dummy:
    pass


def test_input_modes_switch_and_clear_per_chat():
    service = BookService(
        notion_client=_Dummy(),
        telegram_client=_Dummy(),
        vision_client=_Dummy(),
        metadata_resolver=_Dummy(),
        enricher=_Dummy(),
        dry_run=True,
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
