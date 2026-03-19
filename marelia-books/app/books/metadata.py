from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable

import httpx
from pydantic import BaseModel, Field

logger = __import__("logging").getLogger(__name__)

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
    author: str | None = None
    series: str | None = None
    cover_url: str | None = None
    categories: list[str] = Field(default_factory=list)
    reading_type: str = "Physical"
    link: str | None = None
    type: str = "Book"


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


class MetadataResolver:
    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def search_candidates(self, title: str, limit: int = 3) -> list[BookCandidate]:
        params = {"title": title, "limit": limit}
        response = await self._client.get("https://openlibrary.org/search.json", params=params)
        response.raise_for_status()
        docs = response.json().get("docs", [])[:limit]
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
        return self._doc_to_metadata(candidate.raw_doc, candidate.title, candidate.author)

    async def resolve(self, *, title: str, author: str | None = None) -> ResolvedBookMetadata:
        params = {"title": title, "limit": 5}
        if author:
            params["author"] = author

        response = await self._client.get("https://openlibrary.org/search.json", params=params)
        response.raise_for_status()
        docs = response.json().get("docs", [])

        if not docs:
            return ResolvedBookMetadata(title=title, author=author)

        return self._doc_to_metadata(docs[0], title, author)

    def _doc_to_metadata(self, doc: dict[str, Any], title: str, hint_author: str | None) -> ResolvedBookMetadata:
        resolved_title = (doc.get("title") or title).strip()
        resolved_author = (doc.get("author_name") or [hint_author])[0]
        cover_i = doc.get("cover_i")
        cover_url = f"https://covers.openlibrary.org/b/id/{cover_i}-L.jpg" if cover_i else None
        subjects = doc.get("subject") or []
        logger.debug("Subjects crudos de Open Library: %s", subjects[:20])
        categories = map_categories(subjects)
        reading_type = infer_reading_type(" ".join(doc.get("format") or []))
        series = (doc.get("series") or [None])[0]
        key = (doc.get("key") or "").strip()
        link = f"https://openlibrary.org{key}" if key else None

        return ResolvedBookMetadata(
            title=resolved_title,
            author=(resolved_author or hint_author),
            series=series,
            cover_url=cover_url,
            categories=categories,
            reading_type=reading_type,
            link=link,
        )
