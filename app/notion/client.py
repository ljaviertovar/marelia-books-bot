from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.books.metadata import ResolvedBookMetadata
from app.notion.utils import NotionBookRecord, flatten_notion_pages

logger = logging.getLogger(__name__)

NOTION_VERSION = "2022-06-28"
_HEADING_BLOCK_TYPES = frozenset({"heading_1", "heading_2", "heading_3"})
_REQUIRED_TEMPLATE_SECTIONS = ("notes", "synopsis", "references")
_SECTION_DEFAULT_HEADINGS = {
    "notes": "Book Notes",
    "synopsis": "Synopsis",
    "references": "References / Links",
}
_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "notes": (
        "book notes",
        "notes",
        "book details",
        "details",
        "overview",
        "about the book",
        "info",
        "general",
    ),
    "synopsis": ("synopsis", "sinopsis", "summary", "resumen"),
    "references": ("references", "reference", "links", "link", "enlaces", "enlace", "sources", "source"),
}


@dataclass(frozen=True)
class SectionAppendPlan:
    section: str
    after_block_id: str
    children: list[dict[str, Any]]


@dataclass(frozen=True)
class BlockUpdatePlan:
    block_id: str
    payload: dict[str, Any]


def _title_prop(text: str) -> dict[str, Any]:
    return {"title": [{"text": {"content": text}}]}


def _rich_text_prop(text: str) -> dict[str, Any]:
    return {"rich_text": [{"text": {"content": text}}]}


def _url_prop(url: str) -> dict[str, Any]:
    return {"url": url}


def _number_prop(value: int) -> dict[str, Any]:
    return {"number": value}


def _file_prop(url: str) -> dict[str, Any]:
    return {"files": [{"type": "external", "name": "Cover", "external": {"url": url}}]}


def _select_prop(name: str) -> dict[str, Any]:
    return {"select": {"name": name}}


def _status_prop(name: str) -> dict[str, Any]:
    return {"status": {"name": name}}


def _multi_select_prop(values: list[str]) -> dict[str, Any]:
    return {"multi_select": [{"name": item} for item in values]}


def _normalize_categories(categories: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for item in categories:
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def _infer_categories_from_genre(genre_es: str | None) -> list[str]:
    if not genre_es:
        return []

    genre = genre_es.lower()
    inferred: list[str] = []
    if "ciencia ficcion" in genre or "ciencia ficción" in genre or "sci-fi" in genre or "science fiction" in genre:
        inferred.append("Sci-Fi")
    if "fantas" in genre:
        inferred.append("Fantasy")
    if "filosof" in genre or "philosophy" in genre:
        inferred.append("Philosophy")
    if "tecnolog" in genre or "technology" in genre:
        inferred.append("Technology")
    if "historia" in genre or "history" in genre:
        inferred.append("History")
    if "psicolog" in genre or "psychology" in genre:
        inferred.append("Psychology")
    if "biograf" in genre or "biography" in genre:
        inferred.append("Biography")
    if "negocio" in genre or "business" in genre:
        inferred.append("Business")
    if "desarrollo personal" in genre or "self-development" in genre or "self development" in genre:
        inferred.append("Self-development")
    if "no ficcion" in genre or "no ficción" in genre or "non-fiction" in genre:
        inferred.append("Non-fiction")
    return _normalize_categories(inferred)


def _effective_categories(metadata: ResolvedBookMetadata) -> list[str]:
    if metadata.categories:
        return _normalize_categories(metadata.categories)
    inferred = _infer_categories_from_genre(metadata.genre_es)
    if inferred:
        return inferred
    if metadata.genre_es:
        return _normalize_categories([metadata.genre_es])
    return []


def _metadata_category_display(metadata: ResolvedBookMetadata) -> str | None:
    categories = _effective_categories(metadata)
    if categories:
        return ", ".join(categories)
    return metadata.genre_es


def _publication_display(metadata: ResolvedBookMetadata) -> str | None:
    if metadata.year and metadata.language:
        return f"{metadata.year} ({metadata.language})"
    if metadata.year:
        return str(metadata.year)
    return metadata.language


def _append_standard_note_lines(lines: list[str], metadata: ResolvedBookMetadata) -> None:
    if metadata.title:
        lines.append(f"Title: {metadata.title}")
    if metadata.subtitle:
        lines.append(f"Subtitle: {metadata.subtitle}")
    if metadata.title_es:
        lines.append(f"Title (ES): {metadata.title_es}")
    category_display = _metadata_category_display(metadata)
    if category_display:
        lines.append(f"Genre: {category_display}")
    publication = _publication_display(metadata)
    if publication:
        lines.append(f"Original publication (language): {publication}")
    if metadata.publisher:
        lines.append(f"Editorial: {metadata.publisher}")
    if metadata.isbn:
        lines.append(f"ISBN: {metadata.isbn}")
    if metadata.pages:
        lines.append(f"Pages: {metadata.pages}")


def _append_standard_note_values(values: dict[str, str], metadata: ResolvedBookMetadata) -> None:
    if metadata.title:
        values["Title:"] = metadata.title
    if metadata.subtitle:
        values["Subtitle:"] = metadata.subtitle
    if metadata.title_es:
        values["Title (ES):"] = metadata.title_es
    category_display = _metadata_category_display(metadata)
    if category_display:
        values["Genre:"] = category_display
    publication = _publication_display(metadata)
    if publication:
        values["Original publication (language):"] = publication
    if metadata.publisher:
        values["Editorial:"] = metadata.publisher
    if metadata.isbn:
        values["ISBN:"] = metadata.isbn
    if metadata.pages:
        values["Pages:"] = str(metadata.pages)


def _rich_text_segments(text: str, *, url: str | None = None) -> list[dict[str, Any]]:
    segment: dict[str, Any] = {"type": "text", "text": {"content": text}}
    if url:
        segment["text"]["link"] = {"url": url}
    return [segment]


def _clone_rich_text_segments(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for item in items or []:
        if item.get("type") != "text":
            continue
        text = item.get("text", {}) or {}
        segment: dict[str, Any] = {
            "type": "text",
            "text": {
                "content": text.get("content", "") or item.get("plain_text", "") or "",
            },
        }
        link = text.get("link")
        if link and link.get("url"):
            segment["text"]["link"] = {"url": link["url"]}
        annotations = item.get("annotations")
        if annotations:
            segment["annotations"] = annotations
        if item.get("href"):
            segment["href"] = item["href"]
        segments.append(segment)
    return segments


def _paragraph_block(text: str) -> dict[str, Any]:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rich_text_segments(text)}}


def _heading_block(text: str) -> dict[str, Any]:
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": _rich_text_segments(text)}}


def _bulleted_item_block(text: str, *, url: str | None = None) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": _rich_text_segments(text, url=url)},
    }


def _callout_block(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": _rich_text_segments(text),
            "icon": {"type": "emoji", "emoji": "📖"},
            "color": "blue_background",
        },
    }


def _image_block(image_url: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "image",
        "image": {
            "type": "external",
            "external": {"url": image_url},
            "caption": [],
        },
    }


def _clone_block_for_append(block: dict[str, Any], cover_url: str | None = None) -> dict[str, Any] | None:
    block_type = block.get("type")
    if not block_type:
        return None

    payload: dict[str, Any] = {"object": "block", "type": block_type}
    source = dict(block.get(block_type, {}) or {})

    if block_type in {"paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "quote", "to_do", "toggle", "callout"}:
        cloned = {k: v for k, v in source.items() if v is not None}
        if "rich_text" in cloned:
            cloned["rich_text"] = _clone_rich_text_segments(cloned.get("rich_text", []))
        if block_type == "callout" and "icon" in cloned:
            cloned["icon"] = cloned["icon"]
        if block_type == "to_do":
            cloned["checked"] = bool(cloned.get("checked"))
        if block.get("has_children"):
            children = _clone_children_for_append(block.get("children", []))
            if children:
                cloned["children"] = children
        payload[block_type] = cloned
        return payload

    if block_type == "image":
        image = source
        image_type = image.get("type")
        if image_type == "external" and image.get("external", {}).get("url"):
            payload["image"] = {
                "type": "external",
                "external": {"url": image["external"]["url"]},
                "caption": _clone_rich_text_segments(image.get("caption", [])),
            }
            return payload
        # Placeholder image block (not yet filled) — replace with cover if available
        if cover_url:
            payload["image"] = {"type": "external", "external": {"url": cover_url}, "caption": []}
            return payload
        return None

    if block_type == "divider":
        payload["divider"] = {}
        return payload

    return None


def _clone_children_for_append(blocks: list[dict[str, Any]], cover_url: str | None = None) -> list[dict[str, Any]]:
    remaining_cover_url = cover_url
    cloned: list[dict[str, Any]] = []
    for block in blocks:
        payload = _clone_block_for_append(block, remaining_cover_url)
        if payload:
            # Consume cover_url after the first image block so it's only placed once
            if payload.get("type") == "image" and remaining_cover_url:
                remaining_cover_url = None
            cloned.append(payload)
    return cloned


def _extract_plain_text(rich_text: list[dict[str, Any]]) -> str:
    return "".join(seg.get("plain_text", "") or seg.get("text", {}).get("content", "") for seg in rich_text)


def _match_section_name(block: dict[str, Any]) -> str | None:
    block_type = block.get("type")
    if block_type not in _HEADING_BLOCK_TYPES:
        return None

    heading_text = _extract_plain_text(block.get(block_type, {}).get("rich_text", [])).strip().lower()
    if not heading_text:
        return None

    for section, aliases in _SECTION_ALIASES.items():
        if any(alias in heading_text for alias in aliases):
            return section
    return None


def _is_empty_property(prop_name: str, prop_value: dict[str, Any] | None) -> bool:
    if not prop_value:
        return True

    if prop_name in {"Author", "Book Series"}:
        return not (prop_value.get("rich_text") or [])
    if prop_name == "Cover":
        return not (prop_value.get("files") or [])
    if prop_name == "Genre":
        return not (prop_value.get("multi_select") or [])
    if prop_name == "Order to Read":
        return prop_value.get("number") is None
    if prop_name in {"Reading Type", "Book Type"}:
        return not (prop_value.get("select") or {}).get("name")
    if prop_name == "Link":
        return not prop_value.get("url")

    return False


def _metadata_property_values(metadata: ResolvedBookMetadata) -> dict[str, dict[str, Any]]:
    categories = _effective_categories(metadata)
    properties: dict[str, dict[str, Any]] = {}

    if metadata.author:
        properties["Author"] = _rich_text_prop(metadata.author)
    if metadata.series:
        properties["Book Series"] = _rich_text_prop(metadata.series)
    if metadata.order_to_read is not None:
        properties["Order to Read"] = _number_prop(metadata.order_to_read)
    if metadata.cover_url:
        properties["Cover"] = _file_prop(metadata.cover_url)
    if categories:
        properties["Genre"] = _multi_select_prop(categories)
    if metadata.reading_type:
        properties["Reading Type"] = _select_prop(metadata.reading_type)
    if metadata.type:
        properties["Book Type"] = _select_prop(metadata.type)
    if metadata.link:
        properties["Link"] = _url_prop(metadata.link)

    return properties


def build_create_properties(metadata: ResolvedBookMetadata) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "Book Name": _title_prop(metadata.title),
        "Status": _status_prop("Wishlist"),
        "Book Type": _select_prop("TBD"),
        "Reading Type": _select_prop("TBD"),
    }
    properties.update({
        name: value
        for name, value in _metadata_property_values(metadata).items()
        if name not in {"Reading Type", "Book Type"}
    })
    return properties


def build_missing_update_properties(raw_properties: dict[str, Any], metadata: ResolvedBookMetadata) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for name, value in _metadata_property_values(metadata).items():
        if _is_empty_property(name, raw_properties.get(name)):
            updates[name] = value
    return updates


def _build_note_blocks(metadata: ResolvedBookMetadata) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    if metadata.tagline:
        blocks.append(_callout_block(metadata.tagline))

    detail_lines: list[str] = []
    _append_standard_note_lines(detail_lines, metadata)
    if metadata.genre_es:
        detail_lines.append(f"Genero: {metadata.genre_es}")
    if metadata.series:
        detail_lines.append(f"Serie: {metadata.series}")
    if metadata.pages:
        detail_lines.append(f"Paginas: {metadata.pages}")

    blocks.extend(_bulleted_item_block(line) for line in detail_lines)
    return blocks


def build_section_content(metadata: ResolvedBookMetadata) -> dict[str, list[dict[str, Any]]]:
    content: dict[str, list[dict[str, Any]]] = {}

    notes_blocks = _build_note_blocks(metadata)
    if notes_blocks:
        content["notes"] = notes_blocks

    if metadata.synopsis:
        content["synopsis"] = [_paragraph_block(metadata.synopsis[:1900])]

    return content


def _make_label_value_rich_text(label: str, value: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": {"content": f"{label} "},
            "annotations": {"bold": True},
        },
        {"type": "text", "text": {"content": value}},
    ]


def _extract_block_rich_text(block: dict[str, Any]) -> list[dict[str, Any]]:
    block_type = block.get("type")
    if block_type not in {"bulleted_list_item", "numbered_list_item", "paragraph", "heading_1", "heading_2", "heading_3", "callout"}:
        return []
    return block.get(block_type, {}).get("rich_text", []) or []


def _replace_block_rich_text_payload(block: dict[str, Any], rich_text: list[dict[str, Any]]) -> dict[str, Any] | None:
    block_type = block.get("type")
    if block_type not in {"bulleted_list_item", "numbered_list_item", "paragraph", "callout"}:
        return None

    type_data = {k: v for k, v in block.get(block_type, {}).items() if v is not None}
    type_data["rich_text"] = rich_text
    payload: dict[str, Any] = {block_type: type_data}
    payload["type"] = block_type
    return payload


def _replace_block_text_payload(block: dict[str, Any], text: str, *, url: str | None = None) -> dict[str, Any] | None:
    return _replace_block_rich_text_payload(block, _rich_text_segments(text, url=url))


def _replace_image_block_payload(block: dict[str, Any], image_url: str) -> dict[str, Any] | None:
    if block.get("type") != "image":
        return None
    caption = (block.get("image", {}) or {}).get("caption", [])
    payload: dict[str, Any] = {
        "type": "image",
        "image": {
            "external": {"url": image_url},
            "caption": caption,
        },
    }
    return payload


def _notes_label_values(metadata: ResolvedBookMetadata) -> dict[str, str]:
    values: dict[str, str] = {}
    _append_standard_note_values(values, metadata)
    return values


def plan_template_block_updates(page_blocks: list[dict[str, Any]], metadata: ResolvedBookMetadata) -> tuple[list[BlockUpdatePlan], set[str], set[str]]:
    updates: list[BlockUpdatePlan] = []
    filled_sections: set[str] = set()
    placeholder_sections: set[str] = set()

    notes_values = _notes_label_values(metadata)
    used_note_labels: set[str] = set()
    current_section: str | None = None
    synopsis_placeholder: dict[str, Any] | None = None
    tagline_placeholder: dict[str, Any] | None = None
    cover_placeholder: dict[str, Any] | None = None
    note_placeholders_detected = False

    for block in page_blocks:
        block_type = block.get("type")
        plain = _extract_plain_text(_extract_block_rich_text(block)).strip()

        if metadata.tagline and tagline_placeholder is None and block_type == "callout":
            lowered = plain.lower()
            if not plain or lowered.startswith("notion tip:") or "tagline" in lowered:
                tagline_placeholder = block

        if metadata.cover_url and cover_placeholder is None and block_type == "image":
            cover_placeholder = block

        matched_section = _match_section_name(block)
        if matched_section is not None:
            current_section = matched_section
            continue

        if current_section == "notes":
            if block_type in {"bulleted_list_item", "numbered_list_item", "paragraph"}:
                for label, value in notes_values.items():
                    if label in used_note_labels:
                        continue
                    if plain.startswith(label):
                        note_placeholders_detected = True
                        payload = _replace_block_rich_text_payload(block, _make_label_value_rich_text(label, value))
                        block_id = block.get("id")
                        if payload and block_id:
                            updates.append(BlockUpdatePlan(block_id=block_id, payload=payload))
                            filled_sections.add("notes")
                            used_note_labels.add(label)
                        break

        if current_section == "synopsis":
            if block_type in {"bulleted_list_item", "numbered_list_item", "paragraph"}:
                if not plain:
                    if synopsis_placeholder is None:
                        synopsis_placeholder = block
                else:
                    # Already has content — mark filled to prevent duplicate appends
                    filled_sections.add("synopsis")
    if metadata.synopsis and synopsis_placeholder:
        payload = _replace_block_text_payload(synopsis_placeholder, metadata.synopsis[:1900])
        block_id = synopsis_placeholder.get("id")
        if payload and block_id:
            updates.append(BlockUpdatePlan(block_id=block_id, payload=payload))
            filled_sections.add("synopsis")
    if synopsis_placeholder:
        placeholder_sections.add("synopsis")

    if metadata.tagline and tagline_placeholder:
        payload = _replace_block_text_payload(tagline_placeholder, metadata.tagline)
        block_id = tagline_placeholder.get("id")
        if payload and block_id:
            updates.append(BlockUpdatePlan(block_id=block_id, payload=payload))
            filled_sections.add("notes")

    if metadata.cover_url and cover_placeholder:
        payload = _replace_image_block_payload(cover_placeholder, metadata.cover_url)
        block_id = cover_placeholder.get("id")
        if payload and block_id:
            updates.append(BlockUpdatePlan(block_id=block_id, payload=payload))

    if note_placeholders_detected:
        placeholder_sections.add("notes")

    return updates, filled_sections, placeholder_sections


def plan_section_appends(page_blocks: list[dict[str, Any]], metadata: ResolvedBookMetadata) -> list[SectionAppendPlan]:
    section_content = build_section_content(metadata)
    if not section_content:
        return []

    sections_in_order: list[tuple[str, str]] = []
    current_section: str | None = None
    current_after_id: str | None = None
    seen_sections: set[str] = set()

    for block in page_blocks:
        block_id = block.get("id")
        if not block_id:
            continue

        matched_section = _match_section_name(block)
        if matched_section is not None:
            if current_section and current_after_id and current_section not in seen_sections:
                sections_in_order.append((current_section, current_after_id))
                seen_sections.add(current_section)
            current_section = matched_section
            current_after_id = block_id
            continue

        if current_section is not None:
            current_after_id = block_id

    if current_section and current_after_id and current_section not in seen_sections:
        sections_in_order.append((current_section, current_after_id))

    return [
        SectionAppendPlan(section=section, after_block_id=after_block_id, children=section_content[section])
        for section, after_block_id in sections_in_order
        if section in section_content
    ]


def build_missing_template_blocks(page_blocks: list[dict[str, Any]], metadata: ResolvedBookMetadata) -> list[dict[str, Any]]:
    present_sections = {
        section
        for block in page_blocks
        if (section := _match_section_name(block)) is not None
    }
    section_content = build_section_content(metadata)
    blocks: list[dict[str, Any]] = []

    if metadata.cover_url and not any(block.get("type") == "image" for block in page_blocks):
        blocks.append(_image_block(metadata.cover_url))

    for section in _REQUIRED_TEMPLATE_SECTIONS:
        if section in present_sections:
            continue

        blocks.append(_heading_block(_SECTION_DEFAULT_HEADINGS[section]))
        children = section_content.get(section)
        if children:
            blocks.extend(children)
        else:
            blocks.append(_paragraph_block(""))

    return blocks


def _has_template_structure(blocks: list[dict[str, Any]]) -> bool:
    return any(_match_section_name(block) for block in blocks)


def _find_pre_section_insert_anchor(blocks: list[dict[str, Any]]) -> str | None:
    """Return the ID of the last block before the first section heading.
    Used to insert the cover image at the right position when it's missing."""
    anchor: str | None = None
    for block in blocks:
        if _match_section_name(block) is not None:
            break
        block_id = block.get("id")
        if block_id:
            anchor = block_id
    return anchor


class NotionClient:
    def __init__(self, api_key: str, database_id: str, template_id: str, timeout_seconds: float = 20.0) -> None:
        self._database_id = database_id
        self._template_id = template_id
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def query_candidate_books(self, title: str) -> list[NotionBookRecord]:
        logger.info("Buscando en Notion: %r", title[:50])
        payload: dict[str, Any] = {
            "page_size": 50,
            "filter": {
                "property": "Book Name",
                "title": {"contains": title[:50]},
            },
        }
        response = await self._request_with_retry(
            "POST", f"https://api.notion.com/v1/databases/{self._database_id}/query", json=payload
        )
        raw_results = response.json().get("results", [])

        active_results = []
        for page in raw_results:
            if page.get("archived") or page.get("in_trash"):
                title_items = (page.get("properties", {}).get("Book Name", {}).get("title") or [])
                page_title = title_items[0].get("plain_text", "?") if title_items else "?"
                logger.warning(
                    "Ignorando página archivada/en papelera: %r [id=%s]", page_title, page.get("id", "?")
                )
            else:
                active_results.append(page)

        records = flatten_notion_pages(active_results)
        logger.info("Notion: %d resultado(s) activo(s) para %r", len(records), title[:50])
        logger.debug(
            "Notion records:\n%s",
            "\n".join(f"title={r.title!r} author={r.author!r} series={r.series!r}" for r in records),
        )
        return records

    async def create_book_page(self, metadata: ResolvedBookMetadata) -> str:
        logger.info("Creando página en Notion: '%s' de %s", metadata.title, metadata.author or "autor desconocido")

        payload = {
            "parent": {"database_id": self._database_id},
            "properties": build_create_properties(metadata),
            "template": {"type": "template_id", "template_id": self._template_id},
        }

        logger.debug("Payload para Notion:\n%s", json.dumps(payload, indent=2))

        response = await self._request_with_retry("POST", "https://api.notion.com/v1/pages", json=payload)
        page_id = response.json()["id"]
        await self._populate_created_page(page_id, metadata)
        logger.info("Página creada exitosamente en Notion [id=%s]", page_id)
        return page_id

    async def update_book_page_missing(self, record: NotionBookRecord, metadata: ResolvedBookMetadata) -> bool:
        content_changed = await self._ensure_page_content(record.page_id, metadata, seed_missing_sections=True)
        updates = build_missing_update_properties(record._raw_properties, metadata)
        if not updates:
            logger.debug("Sin campos que actualizar en Notion [page_id=%s]", record.page_id)
            return content_changed

        logger.info("Actualizando campos en Notion: %s [page_id=%s]", list(updates.keys()), record.page_id)
        await self._request_with_retry(
            "PATCH",
            f"https://api.notion.com/v1/pages/{record.page_id}",
            json={"properties": updates},
        )
        logger.info("Campos actualizados en Notion [page_id=%s]", record.page_id)
        return True

    async def _populate_created_page(self, page_id: str, metadata: ResolvedBookMetadata) -> None:
        blocks = await self._wait_for_page_content(page_id)
        if not _has_template_structure(blocks):
            logger.warning(
                "El template no apareció a tiempo; intentando aplicar/clonar el template manualmente [page_id=%s]",
                page_id,
            )
            # Intentar aplicar/clonar la estructura del template un par de veces
            seeded = False
            attempts = 3
            for attempt in range(1, attempts + 1):
                try:
                    new_blocks, seeded = await self._ensure_template_structure(page_id, metadata, blocks)
                    if seeded:
                        blocks = new_blocks
                        logger.info("Template aplicado/clonado correctamente en el intento %s [page_id=%s]", attempt, page_id)
                        break
                except Exception as exc:  # pragma: no cover - best-effort retry
                    logger.warning("Error intentando clonar el template (intento %s): %s", attempt, exc)
                if attempt < attempts:
                    await asyncio.sleep(1.0)

            if not seeded:
                # Último intento: esperar un poco más por el template que pueda venir desde Notion
                blocks = await self._wait_for_page_content(page_id, max_attempts=4, delay_seconds=1.0)
                if not _has_template_structure(blocks):
                    logger.warning(
                        "El template no estuvo disponible tras los reintentos; la página se conservará sin template [page_id=%s]",
                        page_id,
                    )
                    return

        await self._ensure_page_content(page_id, metadata, blocks=blocks)

    async def _ensure_page_content(
        self,
        page_id: str,
        metadata: ResolvedBookMetadata,
        *,
        seed_missing_sections: bool = False,
        blocks: list[dict[str, Any]] | None = None,
    ) -> bool:
        changed = False
        if blocks is None:
            blocks = await self._list_block_children(page_id)

        if seed_missing_sections:
            blocks, seeded = await self._ensure_template_structure(page_id, metadata, blocks)
            changed = changed or seeded

        all_blocks = await self._list_block_children_recursive(page_id)
        block_updates, filled_sections, placeholder_sections = plan_template_block_updates(all_blocks, metadata)
        for update in block_updates:
            logger.debug("Actualizando bloque del template [page_id=%s block=%s]", page_id, update.block_id)
            await self._request_with_retry(
                "PATCH",
                f"https://api.notion.com/v1/blocks/{update.block_id}",
                json=update.payload,
            )
            changed = True

        # Insert cover image only if no image block exists anywhere in the page (including nested blocks)
        if metadata.cover_url and not any(b.get("type") == "image" for b in all_blocks):
            after_block_id = _find_pre_section_insert_anchor(blocks)
            image_payload: dict[str, Any] = {"children": [_image_block(metadata.cover_url)]}
            if after_block_id:
                image_payload["after"] = after_block_id
            logger.info(
                "Imagen de portada ausente en la página; insertando [page_id=%s after=%s]",
                page_id,
                after_block_id,
            )
            await self._request_with_retry(
                "PATCH",
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                json=image_payload,
            )
            changed = True

        append_plans = [
            plan
            for plan in plan_section_appends(blocks, metadata)
            if plan.section not in filled_sections and plan.section not in placeholder_sections
        ]
        if not append_plans and not block_updates:
            logger.info("No hay contenido enriquecido o secciones reconocidas para completar [page_id=%s]", page_id)
            return changed

        for plan in append_plans:
            logger.debug(
                "Insertando %d bloque(s) en la sección %s [page_id=%s after=%s]",
                len(plan.children),
                plan.section,
                page_id,
                plan.after_block_id,
            )
            await self._request_with_retry(
                "PATCH",
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                json={"children": plan.children, "after": plan.after_block_id},
            )
            changed = True
        return changed

    async def _ensure_template_structure(
        self,
        page_id: str,
        metadata: ResolvedBookMetadata,
        blocks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        if _has_template_structure(blocks):
            return blocks, False

        # Intentar clonar el template real varias veces antes de caer al fallback
        attempts = 3
        for attempt in range(1, attempts + 1):
            cloned_template_blocks = await self._load_template_blocks_for_append(metadata.cover_url)
            if cloned_template_blocks:
                if metadata.cover_url and not any(b.get("type") == "image" for b in cloned_template_blocks):
                    cloned_template_blocks = [_image_block(metadata.cover_url)] + cloned_template_blocks
                logger.info(
                    "La página no tenía template; clonando la estructura real del template configurado [page_id=%s blocks=%s attempt=%s]",
                    page_id,
                    len(cloned_template_blocks),
                    attempt,
                )
                await self._request_with_retry(
                    "PATCH",
                    f"https://api.notion.com/v1/blocks/{page_id}/children",
                    json={"children": cloned_template_blocks},
                )
                return await self._list_block_children(page_id), True

            logger.debug("No se pudo clonar el template (intento %s) — reintentando", attempt)
            if attempt < attempts:
                await asyncio.sleep(1.0)

        # Si no se pudo clonar el template real tras varios intentos, usar estructura mínima de fallback
        missing_blocks = build_missing_template_blocks(blocks, metadata)
        if missing_blocks:
            logger.info(
                "No se pudo clonar el template real tras reintentos; agregando estructura mínima de fallback [page_id=%s blocks=%s]",
                page_id,
                len(missing_blocks),
            )
            await self._request_with_retry(
                "PATCH",
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                json={"children": missing_blocks},
            )
            return await self._list_block_children(page_id), True

        return blocks, False

    async def _load_template_blocks_for_append(self, cover_url: str | None = None) -> list[dict[str, Any]]:
        try:
            blocks = await self._list_block_children_tree(self._template_id)
        except Exception as exc:
            logger.warning("No se pudo leer el template configurado para clonarlo: %s", exc)
            return []

        cloned = _clone_children_for_append(blocks, cover_url)
        if not cloned:
            logger.warning("El template configurado no devolvió bloques clonables")
        return cloned

    async def _wait_for_page_content(
        self,
        page_id: str,
        *,
        max_attempts: int = 8,
        delay_seconds: float = 1.0,
    ) -> list[dict[str, Any]]:
        last_blocks: list[dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            blocks = await self._list_block_children(page_id)
            last_blocks = blocks
            if any(_match_section_name(block) for block in blocks):
                logger.debug(
                    "Template aplicado en la nueva página [page_id=%s attempt=%s blocks=%s]",
                    page_id,
                    attempt,
                    len(blocks),
                )
                return blocks
            if attempt < max_attempts:
                await asyncio.sleep(delay_seconds)

        logger.warning(
            "Timeout esperando el contenido del template en la nueva página [page_id=%s blocks=%s]",
            page_id,
            len(last_blocks),
        )
        return last_blocks

    async def _list_block_children(self, block_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        start_cursor: str | None = None

        while True:
            url = f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100"
            if start_cursor:
                url = f"{url}&start_cursor={start_cursor}"

            response = await self._request_with_retry("GET", url)
            payload = response.json()
            results.extend(payload.get("results", []))

            if not payload.get("has_more"):
                return results
            start_cursor = payload.get("next_cursor")

    async def _list_block_children_recursive(self, block_id: str) -> list[dict[str, Any]]:
        root_blocks = await self._list_block_children(block_id)
        all_blocks: list[dict[str, Any]] = []

        async def _walk(blocks: list[dict[str, Any]]) -> None:
            for block in blocks:
                all_blocks.append(block)
                if block.get("has_children"):
                    child_blocks = await self._list_block_children(block["id"])
                    await _walk(child_blocks)

        await _walk(root_blocks)
        return all_blocks

    async def _list_block_children_tree(self, block_id: str) -> list[dict[str, Any]]:
        root_blocks = await self._list_block_children(block_id)

        async def _walk(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
            hydrated: list[dict[str, Any]] = []
            for block in blocks:
                current = dict(block)
                if current.get("has_children"):
                    current["children"] = await _walk(await self._list_block_children(current["id"]))
                hydrated.append(current)
            return hydrated

        return await _walk(root_blocks)

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        max_attempts: int = 4,
    ) -> httpx.Response:
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.request(method, url, json=json)
                if response.status_code in (429, 500, 502, 503, 504):
                    retry_after = float(response.headers.get("Retry-After", delay))
                    logger.warning(
                        "Notion respondió %s — reintentando (intento %s, esperando %ss)",
                        response.status_code,
                        attempt,
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    delay = max(delay * 2, retry_after)
                    continue

                if response.is_error:
                    logger.error("Notion error %s — body: %s", response.status_code, response.text)
                    if 400 <= response.status_code < 500:
                        raise RuntimeError(f"Notion client error {response.status_code}: {response.text}")
                response.raise_for_status()
                return response
            except RuntimeError:
                raise
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(f"Notion request failed after retries: {last_error}")
