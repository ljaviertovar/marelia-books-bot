from __future__ import annotations

import json
import re

from app.books.metadata import VisionBookExtraction


class GeminiJSONParseError(ValueError):
    """Raised when Gemini returns malformed or incomplete JSON."""


def _extract_json_object(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]
    if not text:
        raise GeminiJSONParseError("Gemini returned an empty response")

    errors: list[json.JSONDecodeError] = []
    for candidate in (text, text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(exc)

    preview = text[:160].replace("\n", "\\n")
    detail = errors[-1].msg if errors else "unknown JSON parse error"
    raise GeminiJSONParseError(f"{detail}. Raw preview: {preview!r}")


def parse_vision_json(raw_text: str) -> VisionBookExtraction:
    parsed = _extract_json_object(raw_text)
    return VisionBookExtraction.model_validate(parsed)


def parse_enrichment_json(raw_text: str) -> dict[str, str | int | None]:
    """Parse Gemini enrichment JSON response.

    Returns a dict with keys: title_es, genre_es, synopsis, tagline, isbn, pages, series, order_to_read.
    Missing or null values are returned as None.
    """
    parsed = _extract_json_object(raw_text)
    result: dict[str, str | int | None] = {}
    text_aliases = {
        "title_es": ("title_es", "spanish_title", "titulo_es", "titulo"),
        "genre_es": ("genre_es", "genre", "genero_es", "genero"),
        "synopsis": ("synopsis", "summary", "descripcion", "description"),
        "tagline": ("tagline", "blurb", "short_description"),
        "series": ("series", "series_name", "saga", "book_series"),
    }
    for key, aliases in text_aliases.items():
        result[key] = _first_text_value(parsed, *aliases)

    isbn = _first_text_value(parsed, "isbn", "isbn_13", "isbn_10")
    result["isbn"] = str(isbn).strip() if isbn else None
    pages = _first_present_value(parsed, "pages", "page_count")
    result["pages"] = _parse_int_like(pages)
    order_to_read = _first_present_value(parsed, "order_to_read", "series_position", "number_in_series", "reading_order")
    result["order_to_read"] = _parse_int_like(order_to_read)
    return result


def _parse_int_like(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return None


def _first_present_value(payload: dict, *keys: str) -> object | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _first_text_value(payload: dict, *keys: str) -> str | None:
    value = _first_present_value(payload, *keys)
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            if item:
                return str(item).strip() or None
        return None
    text = str(value).strip()
    return text or None
