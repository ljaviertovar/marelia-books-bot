from app.books.deduplication import find_matching_page, make_book_key, normalize_book_text



def test_normalize_book_text_rules():
    assert normalize_book_text("  The-Hobbit: There and Back Again  ") == "the hobbit there and back again"
    assert normalize_book_text("A   B") == "a b"



def test_make_book_key_and_duplicate_match():
    candidates = [
        {
            "properties": {
                "Book Name": {"title": [{"plain_text": "Dune"}]},
                "Author": {"rich_text": [{"plain_text": "Frank Herbert"}]},
            }
        }
    ]

    match = find_matching_page(
        candidates,
        "dune",
        " frank-herbert ",
        get_title=lambda page: page["properties"]["Book Name"]["title"][0]["plain_text"],
        get_author=lambda page: page["properties"]["Author"]["rich_text"][0]["plain_text"],
    )
    assert match is not None
    assert make_book_key("Dune", "Frank Herbert") == "dune|frank herbert"
