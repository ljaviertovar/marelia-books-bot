from app.books.deduplication import find_matching_page, make_book_key, normalize_book_text
from app.notion.utils import NotionBookRecord


def test_normalize_book_text_rules():
    assert normalize_book_text("  The-Hobbit: There and Back Again  ") == "the hobbit there and back again"
    assert normalize_book_text("A   B") == "a b"


def test_make_book_key_and_duplicate_match():
    candidates = [
        NotionBookRecord(
            page_id="page_1",
            title="Dune",
            author="Frank Herbert",
            series=None,
            status=None,
        )
    ]

    match = find_matching_page(
        candidates,
        "dune",
        " frank-herbert ",
    )
    assert match is not None
    assert make_book_key("Dune", "Frank Herbert") == "dune|frank herbert"
