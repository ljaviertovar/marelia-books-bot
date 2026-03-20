from __future__ import annotations

import logging
from typing import Any, Callable

from app.books.metadata import ResolvedBookMetadata

logger = logging.getLogger(__name__)

# Read-only fields returned by the Notion API that must be stripped before creating blocks
_READONLY_FIELDS = frozenset({
    "id",
    "object",
    "created_time",
    "created_by",
    "last_edited_time",
    "last_edited_by",
    "parent",
    "archived",
    "in_trash",
    "has_children",
    "_fetched_children",  # internal tracking field
})

# Maps a label prefix (as it appears at the start of a rich_text block) to the metadata field value.
# Labels must match exactly what the Notion template contains (use DEBUG logs to verify).
_FIELD_INJECTORS: list[tuple[str, Callable[[ResolvedBookMetadata], str | None]]] = [
    ("Title:", lambda m: m.title),
    ("Subtitle:", lambda m: m.subtitle),
    ("Title (ES):", lambda m: m.title_es),
    ("Serie:", lambda m: m.series),
    ("Genre:", lambda m: m.genre_es),
    (
        "Original publication (language):",
        lambda m: (
            f"{m.year} ({m.language})" if m.year and m.language
            else str(m.year) if m.year
            else m.language
        ),
    ),
    ("Editorial:", lambda m: m.publisher),
    ("ISBN:", lambda m: m.isbn),
    ("Pages:", lambda m: str(m.pages) if m.pages else None),
]


def build_blocks_from_template(
    template_blocks: list[dict[str, Any]],
    metadata: ResolvedBookMetadata,
) -> list[dict[str, Any]]:
    """Convert template blocks fetched from Notion into blocks suitable for page creation.

    1. Strips read-only fields.
    2. Recursively nests children.
    3. Populates known label placeholders with metadata values.
    """
    _in_synopsis_section = [False]
    _tagline_injected = [False]  # replace the first callout in the template
    result: list[dict[str, Any]] = []

    for block in template_blocks:
        built = _build_block(block, metadata, _in_synopsis_section, _tagline_injected)
        if built is not None:
            result.append(built)
    return result


def _build_block(
    block: dict[str, Any],
    metadata: ResolvedBookMetadata,
    in_synopsis_section: list[bool],
    tagline_injected: list[bool] | None = None,
) -> dict[str, Any] | None:
    block_type: str = block.get("type", "")
    if not block_type:
        return None

    # Strip read-only fields, keeping only type and the type-specific content
    clean: dict[str, Any] = {"type": block_type}

    type_data: dict[str, Any] = dict(block.get(block_type, {}))

    # Detect section headings and toggle flags
    if block_type in ("heading_1", "heading_2", "heading_3"):
        heading_text = _extract_plain_text(type_data.get("rich_text", [])).lower()
        in_synopsis_section[0] = "synopsis" in heading_text or "sinopsis" in heading_text

    # Replace the first callout block with the tagline
    if block_type == "callout" and metadata.tagline and tagline_injected is not None and not tagline_injected[0]:
        type_data["rich_text"] = [{"type": "text", "text": {"content": metadata.tagline}}]
        type_data["icon"] = {"type": "emoji", "emoji": "📖"}
        type_data["color"] = "blue_background"
        tagline_injected[0] = True
        logger.debug("Template populate callout: tagline=%r", metadata.tagline)

    # Populate rich_text labels for list items / paragraphs
    if block_type in ("bulleted_list_item", "numbered_list_item", "paragraph"):
        rich_text = type_data.get("rich_text", [])
        type_data["rich_text"] = _populate_rich_text(rich_text, metadata, in_synopsis_section, block_type)

    # Populate image blocks with cover_url; drop the block if no URL is available
    if block_type == "image":
        external = type_data.get("external", {})
        if not external.get("url"):
            if metadata.cover_url:
                type_data["type"] = "external"
                type_data["external"] = {"url": metadata.cover_url}
                logger.debug("Template populate image: cover_url=%r", metadata.cover_url)
            else:
                logger.debug("Template: bloque image descartado — sin cover_url")
                return None

    # Handle children: take from _fetched_children, nest inside the type data
    fetched_children: list[dict[str, Any]] = block.get("_fetched_children", [])
    if fetched_children:
        child_syn = [in_synopsis_section[0]]
        built_children = [
            c for raw_child in fetched_children
            if (c := _build_block(raw_child, metadata, child_syn, tagline_injected)) is not None
        ]
        if built_children:
            type_data["children"] = built_children

    clean[block_type] = type_data
    return clean


def _extract_plain_text(rich_text: list[dict[str, Any]]) -> str:
    return "".join(seg.get("plain_text", "") or seg.get("text", {}).get("content", "") for seg in rich_text)


def _populate_rich_text(
    rich_text: list[dict[str, Any]],
    metadata: ResolvedBookMetadata,
    in_synopsis_section: list[bool],
    block_type: str = "bulleted_list_item",
) -> list[dict[str, Any]]:
    plain = _extract_plain_text(rich_text)

    # Synopsis block: empty list item/paragraph inside the synopsis section
    if in_synopsis_section[0] and not plain.strip() and metadata.synopsis:
        return [{"type": "text", "text": {"content": metadata.synopsis}}]

    # Check each known label prefix
    for label, getter in _FIELD_INJECTORS:
        if plain.startswith(label):
            value = getter(metadata)
            if value is None:
                return rich_text  # leave block as-is, no data to inject
            # Preserve bold label segment if already there, else recreate
            label_segments = [seg for seg in rich_text if _extract_plain_text([seg]).startswith(label)]
            label_segment: dict[str, Any]
            if label_segments:
                label_segment = dict(label_segments[0])
            else:
                label_segment = {
                    "type": "text",
                    "text": {"content": f"{label} "},
                    "annotations": {"bold": True},
                }
            # Ensure bold on label
            if "annotations" not in label_segment:
                label_segment["annotations"] = {}
            label_segment["annotations"]["bold"] = True
            # Strip the label text from the label segment (keep it clean)
            label_segment["text"] = {"content": f"{label} "}

            value_segment: dict[str, Any] = {"type": "text", "text": {"content": value}}
            logger.debug("Template populate: %r → %r", label, value)
            return [label_segment, value_segment]

    return rich_text
