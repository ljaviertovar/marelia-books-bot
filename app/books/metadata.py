from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable
import re
import unicodedata

import httpx
from pydantic import BaseModel, Field
logger = __import__("logging").getLogger(__name__)

# MARC/ISO 639-2 language codes → Spanish names (as returned by OpenLibrary)
_LANGUAGE_CODE_MAP: dict[str, str] = {
    "eng": "inglés",
    "spa": "español",
    "fre": "francés",
    "ger": "alemán",
    "ita": "italiano",
    "por": "portugués",
    "rus": "ruso",
    "jpn": "japonés",
    "chi": "chino",
    "zho": "chino",
    "ara": "árabe",
    "dut": "neerlandés",
    "swe": "sueco",
    "nor": "noruego",
    "dan": "danés",
    "pol": "polaco",
    "kor": "coreano",
    "lat": "latín",
    "tur": "turco",
    "gre": "griego",
    "heb": "hebreo",
    "cat": "catalán",
    "ukr": "ucraniano",
}

ALLOWED_CATEGORIES = {
    "Fantasy",
    "Sci-Fi",
    "Non-fiction",
    "Philosophy",
    "Technology",
    "Business",
    "History",
    "Psychology",
    "Biography",
    "Self-development",
}

_CATEGORY_MAP = {
    # English
    "fantasy": "Fantasy",
    "science fiction": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "nonfiction": "Non-fiction",
    "non-fiction": "Non-fiction",
    "philosophy": "Philosophy",
    "technology": "Technology",
    "business": "Business",
    "history": "History",
    "psychology": "Psychology",
    "biography": "Biography",
    "self-help": "Self-development",
    "self development": "Self-development",
    "self-development": "Self-development",
    # Spanish
    "fantasía": "Fantasy",
    "fantasia": "Fantasy",
    "ciencia ficción": "Sci-Fi",
    "ciencia ficcion": "Sci-Fi",
    "ficción científica": "Sci-Fi",
    "ficcion cientifica": "Sci-Fi",
    "no ficción": "Non-fiction",
    "no ficcion": "Non-fiction",
    "filosofía": "Philosophy",
    "filosofia": "Philosophy",
    "tecnología": "Technology",
    "tecnologia": "Technology",
    "negocios": "Business",
    "historia": "History",
    "psicología": "Psychology",
    "psicologia": "Psychology",
    "biografía": "Biography",
    "biografia": "Biography",
    "autobiografía": "Biography",
    "autobiografia": "Biography",
    "desarrollo personal": "Self-development",
    "autoayuda": "Self-development",
}

_GENERIC_SERIES_PATTERNS = (
    "best seller",
    "bestseller",
    "new york times",
    "award-winning",
    "award winning",
    "premio",
)


class VisionBookExtraction(BaseModel):
    is_book_cover: bool
    title: str | None
    subtitle: str | None
    authors: list[str] = Field(default_factory=list)
    series_or_edition: str | None
    language: str | None
    confidence: float
    reason_if_not_book: str | None
    raw_visible_text: str | None


class ResolvedBookMetadata(BaseModel):
    title: str
    subtitle: str | None = None
    author: str | None = None
    series: str | None = None
    order_to_read: int | None = None
    cover_url: str | None = None
    categories: list[str] = Field(default_factory=list)
    reading_type: str = "Physical"
    link: str | None = None
    type: str = "Book"
    # Extended fields (from OpenLibrary or Gemini enrichment)
    isbn: str | None = None
    pages: int | None = None
    year: int | None = None
    publisher: str | None = None
    language: str | None = None
    title_es: str | None = None
    genre_es: str | None = None
    synopsis: str | None = None
    tagline: str | None = None


@dataclass
class BookCandidate:
    title: str
    author: str | None
    year: int | None = None
    publisher: str | None = None
    language: str | None = None
    raw_doc: dict[str, Any] = dc_field(default_factory=dict)



def map_categories(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []

    mapped: list[str] = []
    seen = set()
    for raw in values:
        lowered = raw.strip().lower()
        for key, target in _CATEGORY_MAP.items():
            if key in lowered and target not in seen:
                seen.add(target)
                mapped.append(target)
                break
    return [name for name in mapped if name in ALLOWED_CATEGORIES]



def infer_reading_type(raw: str | None) -> str:
    lowered = (raw or "").lower()
    if "audiobook" in lowered:
        return "Audiobook"
    if "ebook" in lowered or "e-book" in lowered:
        return "eBook"
    return "Physical"


def sanitize_series_name(value: str | None) -> str | None:
    if not value:
        return None

    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if any(pattern in lowered for pattern in _GENERIC_SERIES_PATTERNS):
        return None

    return cleaned


def _extract_text_value(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = " ".join(value.strip().split())
        return cleaned or None

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        for item in value:
            extracted = _extract_text_value(item)
            if extracted:
                return extracted
        return None

    if isinstance(value, dict):
        for key in ("name", "title", "value", "label", "text"):
            extracted = _extract_text_value(value.get(key))
            if extracted:
                return extracted
    return None


def _extract_first_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value) if value.is_integer() else None

    if isinstance(value, list):
        for item in value:
            extracted = _extract_first_int(item)
            if extracted is not None:
                return extracted
        return None

    if isinstance(value, dict):
        for key in ("value", "number", "position", "sequence", "order"):
            extracted = _extract_first_int(value.get(key))
            if extracted is not None:
                return extracted
        return None

    text = _extract_text_value(value)
    if not text:
        return None

    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


def extract_series_name(payload: dict[str, Any]) -> str | None:
    for key in ("series", "series_name", "book_series", "saga"):
        extracted = _extract_text_value(payload.get(key))
        cleaned = sanitize_series_name(extracted)
        if cleaned:
            return cleaned
    return None


def extract_series_order(payload: dict[str, Any]) -> int | None:
    for key in ("series_position", "number_in_series", "order_in_series", "reading_order", "sequence"):
        extracted = _extract_first_int(payload.get(key))
        if extracted is not None:
            return extracted
    return None


def resolve_openlibrary_cover_url(doc: dict[str, Any]) -> str | None:
    """Pick the most edition-specific OpenLibrary cover URL available in a search doc."""
    cover_edition_key = (doc.get("cover_edition_key") or "").strip()
    if cover_edition_key:
        return f"https://covers.openlibrary.org/b/olid/{cover_edition_key}-L.jpg"

    edition_keys = doc.get("edition_key") or []
    if edition_keys:
        first_edition_key = str(edition_keys[0]).strip()
        if first_edition_key:
            return f"https://covers.openlibrary.org/b/olid/{first_edition_key}-L.jpg"

    cover_i = doc.get("cover_i")
    if cover_i:
        return f"https://covers.openlibrary.org/b/id/{cover_i}-L.jpg"

    return None


def _extract_work_key(doc: dict[str, Any]) -> str | None:
    key = str(doc.get("key") or "").strip()
    if key.startswith("/works/"):
        return key

    works = doc.get("works") or []
    if works:
        first = works[0]
        if isinstance(first, dict):
            work_key = str(first.get("key") or "").strip()
            if work_key.startswith("/works/"):
                return work_key
    return None


def _extract_edition_key(doc: dict[str, Any]) -> str | None:
    cover_edition_key = str(doc.get("cover_edition_key") or "").strip()
    if cover_edition_key:
        return cover_edition_key
    edition_keys = doc.get("edition_key") or []
    if edition_keys:
        first = str(edition_keys[0]).strip()
        if first:
            return first
    return None


def _lang_name_from_edition_payload(payload: dict[str, Any]) -> str | None:
    languages = payload.get("languages") or []
    for item in languages:
        if isinstance(item, dict):
            key = str(item.get("key") or "").strip()
            if key:
                code = key.rsplit("/", 1)[-1].lower()
                return _LANGUAGE_CODE_MAP.get(code, code)
    return None


def _normalize_search_text(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", (value or "").strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace(",", " ")
    return " ".join(text.split())


def _sort_search_docs(docs: list[dict[str, Any]], title: str, author: str | None = None) -> list[dict[str, Any]]:
    target_title = _normalize_search_text(title)
    target_author = _normalize_search_text(author)

    def sort_key(doc: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
        doc_title = _normalize_search_text(doc.get("title"))
        doc_author = _normalize_search_text(((doc.get("author_name") or [None])[0]))

        exact_title = int(doc_title == target_title)
        prefix_title = int(bool(target_title) and doc_title.startswith(target_title))
        exact_author = int(bool(target_author) and doc_author == target_author)
        partial_author = int(bool(target_author) and target_author in doc_author)
        has_cover = int(bool(doc.get("cover_edition_key") or doc.get("edition_key") or doc.get("cover_i")))

        if target_author:
            return (-exact_author, -partial_author, -exact_title, -prefix_title, -has_cover, doc_title)
        return (-exact_title, -prefix_title, -exact_author, -partial_author, -has_cover, doc_title)

    return sorted(docs, key=sort_key)


def _openlibrary_summary(payload: dict[str, Any]) -> dict[str, Any]:
    description = payload.get("description")
    if isinstance(description, dict):
        description = description.get("value")
    if isinstance(description, str):
        description = " ".join(description.strip().split())[:160]
    else:
        description = None

    return {
        "title": payload.get("title"),
        "author_name": (payload.get("author_name") or [None])[0],
        "series": payload.get("series") or payload.get("series_name") or payload.get("book_series"),
        "series_position": payload.get("series_position") or payload.get("number_in_series") or payload.get("reading_order"),
        "subjects": (payload.get("subjects") or payload.get("subject") or [])[:5],
        "description": description,
        "key": payload.get("key"),
    }


def _log_openlibrary_result(label: str, payload: dict[str, Any]) -> None:
    summary = _openlibrary_summary(payload)
    logger.info("━" * 60)
    logger.info("📚 OPEN LIBRARY %s", label.upper())
    logger.info("  title         : %r", summary["title"])
    logger.info("  author        : %r", summary["author_name"])
    logger.info("  series        : %r", summary["series"])
    logger.info("  series_order  : %r", summary["series_position"])
    logger.info("  subjects      : %s", summary["subjects"])
    logger.info("  synopsis      : %r", summary["description"])
    logger.info("  key           : %r", summary["key"])
    logger.info("━" * 60)


def _has_additional_openlibrary_details(base_payload: dict[str, Any], candidate_payload: dict[str, Any]) -> bool:
    base = _openlibrary_summary(base_payload)
    candidate = _openlibrary_summary(candidate_payload)

    detail_keys = ("series", "series_position", "subjects", "description")
    for key in detail_keys:
        base_value = base.get(key)
        candidate_value = candidate.get(key)
        if candidate_value and candidate_value != base_value:
            return True
    return False


class MetadataResolver:
    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def search_candidates(self, title: str, limit: int = 3) -> list[BookCandidate]:
        params = {"q": title, "limit": max(limit * 5, 10)}
        response = await self._client.get("https://openlibrary.org/search.json", params=params)
        response.raise_for_status()
        docs = _sort_search_docs(response.json().get("docs", []), title)[:limit]
        if docs:
            _log_openlibrary_result(f"search top candidate for {title!r}", docs[0])
        candidates = []
        for doc in docs:
            t = (doc.get("title") or title).strip()
            a = (doc.get("author_name") or [None])[0]
            year = doc.get("first_publish_year") or None
            publisher = (doc.get("publisher") or [None])[0]
            langs = doc.get("language") or []
            language = langs[0] if langs else None
            candidates.append(BookCandidate(title=t, author=a, year=year, publisher=publisher, language=language, raw_doc=doc))
        return candidates

    async def resolve_from_candidate(self, candidate: BookCandidate) -> ResolvedBookMetadata:
        metadata = self._doc_to_metadata(candidate.raw_doc, candidate.title, candidate.author)
        return await self._enrich_with_openlibrary_details(metadata, candidate.raw_doc)

    async def resolve(self, *, title: str, author: str | None = None) -> ResolvedBookMetadata:
        params = {"q": title, "limit": 25}
        if author:
            params["author"] = author

        response = await self._client.get("https://openlibrary.org/search.json", params=params)
        response.raise_for_status()
        docs = _sort_search_docs(response.json().get("docs", []), title, author)

        if not docs:
            return ResolvedBookMetadata(title=title, author=author)

        _log_openlibrary_result(f"resolved top doc for title={title!r} author={author!r}", docs[0])

        metadata = self._doc_to_metadata(docs[0], title, author)
        if author and title and _normalize_search_text(metadata.title) != _normalize_search_text(title):
            metadata.title_es = title
        return await self._enrich_with_openlibrary_details(metadata, docs[0])

    async def _enrich_with_openlibrary_details(self, metadata: ResolvedBookMetadata, doc: dict[str, Any]) -> ResolvedBookMetadata:
        """Get additional data from OpenLibrary edition/work endpoints before Gemini."""
        updated = metadata.model_copy()
        work_key = _extract_work_key(doc)
        edition_key = _extract_edition_key(doc)

        if edition_key:
            try:
                edition_resp = await self._client.get(f"https://openlibrary.org/books/{edition_key}.json")
                edition_resp.raise_for_status()
                edition = edition_resp.json()
                _log_openlibrary_result(f"edition {edition_key}", edition)

                if not updated.pages:
                    pages_raw = edition.get("number_of_pages")
                    if pages_raw is not None:
                        updated.pages = int(pages_raw)

                if not updated.publisher:
                    publishers = edition.get("publishers") or []
                    if publishers:
                        updated.publisher = str(publishers[0]).strip() or None

                if not updated.isbn:
                    isbn_list = edition.get("isbn_13") or edition.get("isbn_10") or []
                    if isbn_list:
                        updated.isbn = str(isbn_list[0]).strip() or None

                if not updated.language:
                    updated.language = _lang_name_from_edition_payload(edition)

                if not updated.cover_url:
                    covers = edition.get("covers") or []
                    if covers:
                        updated.cover_url = f"https://covers.openlibrary.org/b/id/{covers[0]}-L.jpg"

                if not updated.series:
                    updated.series = extract_series_name(edition)

                if updated.order_to_read is None:
                    updated.order_to_read = extract_series_order(edition)

                if not work_key:
                    works = edition.get("works") or []
                    if works:
                        first = works[0]
                        if isinstance(first, dict):
                            wk = str(first.get("key") or "").strip()
                            if wk.startswith("/works/"):
                                work_key = wk
            except Exception as exc:
                logger.debug("No se pudo enriquecer desde edition %s: %s", edition_key, exc)

        if work_key:
            try:
                work_resp = await self._client.get(f"https://openlibrary.org{work_key}.json")
                work_resp.raise_for_status()
                work = work_resp.json()
                if _has_additional_openlibrary_details(doc, work):
                    _log_openlibrary_result(f"work {work_key}", work)

                if not updated.categories:
                    subjects = work.get("subjects") or []
                    updated.categories = map_categories(subjects)

                if not updated.synopsis:
                    description = work.get("description")
                    if isinstance(description, str):
                        updated.synopsis = description.strip() or None
                    elif isinstance(description, dict):
                        updated.synopsis = str(description.get("value") or "").strip() or None

                if not updated.cover_url:
                    covers = work.get("covers") or []
                    if covers:
                        updated.cover_url = f"https://covers.openlibrary.org/b/id/{covers[0]}-L.jpg"

                if not updated.series:
                    updated.series = extract_series_name(work)

                if updated.order_to_read is None:
                    updated.order_to_read = extract_series_order(work)
            except Exception as exc:
                logger.debug("No se pudo enriquecer desde work %s: %s", work_key, exc)

        return updated

    def _doc_to_metadata(self, doc: dict[str, Any], title: str, hint_author: str | None) -> ResolvedBookMetadata:
        resolved_title = (doc.get("title") or title).strip()
        resolved_author = (doc.get("author_name") or [hint_author])[0]
        cover_url = resolve_openlibrary_cover_url(doc)
        subjects = doc.get("subject") or []
        logger.debug("Subjects crudos de Open Library: %s", subjects[:20])
        categories = map_categories(subjects)
        reading_type = infer_reading_type(" ".join(doc.get("format") or []))
        series = extract_series_name(doc)
        order_to_read = extract_series_order(doc)
        key = (doc.get("key") or "").strip()
        link = f"https://openlibrary.org{key}" if key else None

        # Extended fields from OpenLibrary
        isbn_list = doc.get("isbn_13") or doc.get("isbn_10") or []
        isbn = isbn_list[0] if isbn_list else None
        pages_raw = doc.get("number_of_pages_median")
        pages = int(pages_raw) if pages_raw is not None else None
        year = doc.get("first_publish_year") or None
        publisher = (doc.get("publisher") or [None])[0]
        lang_list = doc.get("language") or []
        lang_code = lang_list[0] if lang_list else None
        language = _LANGUAGE_CODE_MAP.get(lang_code, lang_code) if lang_code else None

        return ResolvedBookMetadata(
            title=resolved_title,
            author=(resolved_author or hint_author),
            series=series,
            order_to_read=order_to_read,
            cover_url=cover_url,
            categories=categories,
            reading_type=reading_type,
            link=link,
            isbn=isbn,
            pages=pages,
            year=year,
            publisher=publisher,
            language=language,
        )
