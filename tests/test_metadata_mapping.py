import asyncio

import httpx

from app.books.metadata import (
    BookCandidate,
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


def test_search_candidates_falls_back_to_google_books_when_openlibrary_is_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "openlibrary.org":
            return httpx.Response(200, json={"docs": []})
        if request.url.host == "www.googleapis.com":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "volumeInfo": {
                                "title": "Artemisa",
                                "authors": ["Andy Weir"],
                                "publisher": "Nova",
                                "publishedDate": "2017-11-14",
                                "language": "es",
                            }
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = MetadataResolver(client=client)

    candidates = asyncio.run(resolver.search_candidates("Artemisa", limit=3))

    assert len(candidates) == 1
    assert candidates[0].title == "Artemisa"
    assert candidates[0].author == "Andy Weir"
    assert candidates[0].publisher == "Nova"
    assert candidates[0].year == 2017
    assert candidates[0].raw_doc["_source"] == "google_books"
    asyncio.run(client.aclose())


def test_resolve_falls_back_to_google_books_when_openlibrary_has_no_match():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "openlibrary.org":
            return httpx.Response(200, json={"docs": []})
        if request.url.host == "www.googleapis.com":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "volumeInfo": {
                                "title": "The Pragmatic Programmer",
                                "authors": ["Andrew Hunt", "David Thomas"],
                                "description": "Un clasico moderno sobre practica profesional de software.",
                                "publisher": "Addison-Wesley",
                                "publishedDate": "1999-10-20",
                                "pageCount": 352,
                                "industryIdentifiers": [{"type": "ISBN_13", "identifier": "9780201616224"}],
                                "imageLinks": {"thumbnail": "http://example.com/pragmatic.jpg"},
                                "language": "en",
                                "categories": ["Technology"],
                                "infoLink": "https://books.google.com/books?id=abc",
                            }
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = MetadataResolver(client=client)

    metadata = asyncio.run(resolver.resolve(title="The Pragmatic Programmer", author="Andrew Hunt"))

    assert metadata.title == "The Pragmatic Programmer"
    assert metadata.author == "Andrew Hunt"
    assert metadata.publisher == "Addison-Wesley"
    assert metadata.pages == 352
    assert metadata.isbn == "9780201616224"
    assert metadata.cover_url == "https://example.com/pragmatic.jpg"
    assert metadata.language == "inglés"
    assert metadata.categories == ["Technology"]
    assert metadata.synopsis == "Un clasico moderno sobre practica profesional de software."
    assert metadata.link == "https://books.google.com/books?id=abc"
    asyncio.run(client.aclose())


def test_resolve_from_candidate_uses_google_books_to_fill_missing_openlibrary_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.googleapis.com":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "volumeInfo": {
                                "title": "Dune",
                                "authors": ["Frank Herbert"],
                                "description": "En Arrakis, la especia decide el destino del imperio.",
                                "publisher": "Ace",
                                "publishedDate": "1965-08-01",
                                "pageCount": 412,
                                "industryIdentifiers": [{"type": "ISBN_13", "identifier": "9780441172719"}],
                                "imageLinks": {"thumbnail": "http://example.com/dune.jpg"},
                                "language": "en",
                                "categories": ["Science Fiction"],
                                "infoLink": "https://books.google.com/books?id=dune",
                            }
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = MetadataResolver(client=client)
    candidate = BookCandidate(
        title="Dune",
        author="Frank Herbert",
        raw_doc={
            "title": "Dune",
            "author_name": ["Frank Herbert"],
        },
    )

    metadata = asyncio.run(resolver.resolve_from_candidate(candidate))

    assert metadata.title == "Dune"
    assert metadata.author == "Frank Herbert"
    assert metadata.publisher == "Ace"
    assert metadata.pages == 412
    assert metadata.isbn == "9780441172719"
    assert metadata.cover_url == "https://example.com/dune.jpg"
    assert metadata.language == "inglés"
    assert metadata.categories == ["Sci-Fi"]
    assert metadata.synopsis == "En Arrakis, la especia decide el destino del imperio."
    assert metadata.link == "https://books.google.com/books?id=dune"
    asyncio.run(client.aclose())
