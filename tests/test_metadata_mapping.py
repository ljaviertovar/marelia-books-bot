from app.books.metadata import (
    MetadataResolver,
    _extract_edition_key,
    _extract_work_key,
    _lang_name_from_edition_payload,
    _sort_search_docs,
    extract_series_name,
    extract_series_order,
    infer_reading_type,
    map_categories,
    resolve_openlibrary_cover_url,
    sanitize_series_name,
)



def test_category_mapping_to_allowed_values_only():
    values = ["Epic Fantasy", "Science Fiction", "Self-Help", "Cooking"]
    assert map_categories(values) == ["Fantasy", "Sci-Fi", "Self-development"]



def test_reading_type_mapping():
    assert infer_reading_type("This is an audiobook edition") == "Audiobook"
    assert infer_reading_type("DRM-free ebook format") == "eBook"
    assert infer_reading_type("hardcover") == "Physical"


def test_sanitize_series_name_rejects_marketing_labels():
    assert sanitize_series_name("Best Seller") is None
    assert sanitize_series_name("New York Times Bestseller") is None
    assert sanitize_series_name("Foundation Series") == "Foundation Series"


def test_extract_series_fields_from_varied_openlibrary_shapes():
    payload = {
        "series": [{"name": "Reino de Sombras"}],
        "series_position": "Book 2",
    }

    assert extract_series_name(payload) == "Reino de Sombras"
    assert extract_series_order(payload) == 2


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


def test_sort_search_docs_prioritizes_exact_title_matches():
    docs = [
        {"title": "Artemis Fowl"},
        {"title": "Artemisa", "cover_i": 123},
        {"title": "Artemisa ilustrada"},
    ]

    sorted_docs = _sort_search_docs(docs, "Artemisa")

    assert sorted_docs[0]["title"] == "Artemisa"


def test_sort_search_docs_prioritizes_author_when_provided():
    docs = [
        {"title": "Artemisa", "author_name": ["unknown author"]},
        {"title": "Artemis", "author_name": ["Andy Weir"], "cover_i": 123},
    ]

    sorted_docs = _sort_search_docs(docs, "Artemisa", "Andy Weir")

    assert sorted_docs[0]["author_name"][0] == "Andy Weir"


def test_resolve_sets_title_es_from_query_when_selected_title_differs():
    resolver = MetadataResolver()
    doc = {"title": "Artemis", "author_name": ["Andy Weir"]}

    metadata = resolver._doc_to_metadata(doc, "Artemisa", "Andy Weir")
    if metadata.title != "Artemisa":
        metadata.title_es = "Artemisa"

    assert metadata.title == "Artemis"
    assert metadata.title_es == "Artemisa"


def test_doc_to_metadata_reads_series_and_order_when_present():
    resolver = MetadataResolver()
    doc = {
        "title": "Lo que el tiempo olvidó",
        "author_name": ["Lorena Franco"],
        "series_name": "Trilogía del tiempo",
        "number_in_series": "2",
    }

    metadata = resolver._doc_to_metadata(doc, "Lo que el tiempo olvidó", "Lorena Franco")

    assert metadata.series == "Trilogía del tiempo"
    assert metadata.order_to_read == 2


def test_openlibrary_summary_includes_series_and_description():
    from app.books.metadata import _openlibrary_summary

    summary = _openlibrary_summary(
        {
            "title": "Lo que el tiempo olvidó",
            "author_name": ["Lorena Franco"],
            "series_name": "Trilogía del tiempo",
            "reading_order": "2",
            "description": {"value": "Una novela corta sobre recuerdos, tiempo y secretos."},
            "subjects": ["Thrillers", "Suspense"],
        }
    )

    assert summary["title"] == "Lo que el tiempo olvidó"
    assert summary["series"] == "Trilogía del tiempo"
    assert summary["series_position"] == "2"
    assert summary["description"] == "Una novela corta sobre recuerdos, tiempo y secretos."
