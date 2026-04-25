from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NotionBookRecord:
    """Flat representation of a Notion page for matching and update operations."""

    page_id: str
    title: str | None
    author: str | None
    series: str | None
    page_cover_url: str | None = field(default=None, repr=False, compare=False)
    # Raw properties kept internally for update operations (not for matching logic)
    _raw_properties: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def flatten_notion_pages(pages: list[dict[str, Any]]) -> list[NotionBookRecord]:
    return [_flatten_page(p) for p in pages]


def _flatten_page(page: dict[str, Any]) -> NotionBookRecord:
    props = page.get("properties") or {}
    cover = page.get("cover") or {}
    page_cover_url: str | None = None
    if cover.get("type") == "external":
        page_cover_url = (cover.get("external") or {}).get("url") or None
    elif cover.get("type") == "file":
        page_cover_url = (cover.get("file") or {}).get("url") or None
    return NotionBookRecord(
        page_id=page["id"],
        title=_extract_title(props.get("Book Name")),
        author=_extract_rich_text(props.get("Author")),
        series=_extract_rich_text(props.get("Book Series")),
        page_cover_url=page_cover_url,
        _raw_properties=props,
    )


def _extract_title(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    items = value.get("title") or []
    if not items:
        return None
    return (items[0].get("plain_text") or "").strip() or None


def _extract_rich_text(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    items = value.get("rich_text") or []
    if not items:
        return None
    return (items[0].get("plain_text") or "").strip() or None
