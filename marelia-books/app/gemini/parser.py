from __future__ import annotations

import json

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
    for key in ("title_es", "genre_es", "synopsis", "tagline", "series"):
        val = parsed.get(key)
        result[key] = str(val).strip() if val else None
    isbn = parsed.get("isbn")
    result["isbn"] = str(isbn).strip() if isbn else None
    pages = parsed.get("pages")
    if isinstance(pages, int):
        result["pages"] = pages
    elif isinstance(pages, str) and pages.strip().isdigit():
        result["pages"] = int(pages.strip())
    else:
        result["pages"] = None
    order_to_read = parsed.get("order_to_read")
    if isinstance(order_to_read, int):
        result["order_to_read"] = order_to_read
    elif isinstance(order_to_read, str) and order_to_read.strip().isdigit():
        result["order_to_read"] = int(order_to_read.strip())
    else:
        result["order_to_read"] = None
    return result
