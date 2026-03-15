from __future__ import annotations

from typing import Iterable

import httpx
from pydantic import BaseModel, Field

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

    async def resolve(self, *, title: str, author: str | None = None) -> ResolvedBookMetadata:
        params = {"title": title, "limit": 5}
        if author:
            params["author"] = author

        response = await self._client.get("https://openlibrary.org/search.json", params=params)
        response.raise_for_status()
        docs = response.json().get("docs", [])

        if not docs:
            return ResolvedBookMetadata(title=title, author=author)

        best = docs[0]
        resolved_title = (best.get("title") or title).strip()
        resolved_author = (best.get("author_name") or [author])[0]
        cover_i = best.get("cover_i")
        cover_url = f"https://covers.openlibrary.org/b/id/{cover_i}-L.jpg" if cover_i else None
        subjects = best.get("subject") or []
        categories = map_categories(subjects)
        reading_type = infer_reading_type(" ".join(best.get("format") or []))
        series = (best.get("series") or [None])[0]
        key = (best.get("key") or "").strip()
        link = f"https://openlibrary.org{key}" if key else None

        return ResolvedBookMetadata(
            title=resolved_title,
            author=(resolved_author or author),
            series=series,
            cover_url=cover_url,
            categories=categories,
            reading_type=reading_type,
            link=link,
        )
