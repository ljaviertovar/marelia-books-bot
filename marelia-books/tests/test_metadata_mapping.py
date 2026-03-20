from app.books.metadata import (
    _extract_edition_key,
    _extract_work_key,
    _lang_name_from_edition_payload,
    infer_reading_type,
    map_categories,
    resolve_openlibrary_cover_url,
)



def test_category_mapping_to_allowed_values_only():
    values = ["Epic Fantasy", "Science Fiction", "Self-Help", "Cooking"]
    assert map_categories(values) == ["Fantasy", "Sci-Fi", "Self-development"]



def test_reading_type_mapping():
    assert infer_reading_type("This is an audiobook edition") == "Audiobook"
    assert infer_reading_type("DRM-free ebook format") == "eBook"
    assert infer_reading_type("hardcover") == "Physical"


def test_cover_url_prefers_cover_edition_key_then_edition_key_then_cover_i():
    assert resolve_openlibrary_cover_url({"cover_edition_key": "OL57487091M", "cover_i": 1}) == (
        "https://covers.openlibrary.org/b/olid/OL57487091M-L.jpg"
    )
    assert resolve_openlibrary_cover_url({"edition_key": ["OL1111111M"], "cover_i": 2}) == (
        "https://covers.openlibrary.org/b/olid/OL1111111M-L.jpg"
    )
    assert resolve_openlibrary_cover_url({"cover_i": 8239821}) == (
        "https://covers.openlibrary.org/b/id/8239821-L.jpg"
    )


def test_extract_work_and_edition_keys():
    assert _extract_work_key({"key": "/works/OL21745884W"}) == "/works/OL21745884W"
    assert _extract_work_key({"works": [{"key": "/works/OL1W"}]}) == "/works/OL1W"
    assert _extract_edition_key({"cover_edition_key": "OL57487091M"}) == "OL57487091M"
    assert _extract_edition_key({"edition_key": ["OL123M"]}) == "OL123M"


def test_language_from_edition_payload():
    payload = {"languages": [{"key": "/languages/eng"}]}
    assert _lang_name_from_edition_payload(payload) == "inglés"
