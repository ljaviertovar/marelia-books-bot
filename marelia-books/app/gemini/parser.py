from __future__ import annotations

import json

from app.books.metadata import VisionBookExtraction


def _extract_json_object(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Gemini sometimes puts literal newlines inside string values (invalid JSON).
        # Collapse them to spaces and retry once.
        sanitized = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        return json.loads(sanitized)


def parse_vision_json(raw_text: str) -> VisionBookExtraction:
    parsed = _extract_json_object(raw_text)
    return VisionBookExtraction.model_validate(parsed)


def parse_enrichment_json(raw_text: str) -> dict[str, str | None]:
    """Parse Gemini enrichment JSON response.

    Returns a dict with keys: title_es, genre_es, synopsis, publisher_url.
    Missing or null values are returned as None.
    """
    parsed = _extract_json_object(raw_text)
    result: dict[str, str | None] = {}
    for key in ("title_es", "genre_es", "synopsis", "publisher_url", "tagline"):
        val = parsed.get(key)
        result[key] = str(val).strip() if val else None
    return result
