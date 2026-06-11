from __future__ import annotations

import base64
import logging
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from app.books.metadata import VisionBookExtraction
from app.gemini.parser import GeminiJSONParseError, parse_vision_json

logger = logging.getLogger(__name__)

_MODEL = "gemini-3-flash-preview"
_PROMPT = (
    "You are a strict JSON extractor for book covers. "
    "Return JSON only, no markdown, no explanation. "
    "Keep every field concise. "
    "Use at most 2 authors. "
    "Keep subtitle under 120 characters. "
    "Keep series_or_edition under 80 characters. "
    "Keep reason_if_not_book under 120 characters. "
    "Keep raw_visible_text under 120 characters. "
    "If the cover mentions a saga, trilogy, series name, installment number, or wording like "
    "'book 2', 'tomo 3', 'volumen 1', include that clue in series_or_edition. "
    "Prefer the real series name over marketing labels. "
    "Schema: "
    '{"is_book_cover": true, '
    '"title": "string | null", '
    '"subtitle": "string | null", '
    '"authors": ["string"], '
    '"series_or_edition": "string | null", '
    '"language": "string | null", '
    '"confidence": 0.0, '
    '"reason_if_not_book": "string | null", '
    '"raw_visible_text": "string | null"}'
)


class GeminiVisionQuotaError(RuntimeError):
    """Raised when Gemini Vision quota is exhausted (HTTP 429)."""


class GeminiVisionResponseError(RuntimeError):
    """Raised when Gemini Vision returns an unusable response."""


class GeminiVisionClient:
    def __init__(self, api_key: str, timeout_seconds: float = 30.0) -> None:
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    async def close(self) -> None:
        pass  # SDK manages its own connection pool

    async def extract_book_data(
        self, image_bytes: bytes, mime_type: str
    ) -> VisionBookExtraction:
        logger.info("Enviando imagen a Gemini (%d bytes)", len(image_bytes))
        extraction: VisionBookExtraction | None = None
        last_error: Exception | None = None

        for max_output_tokens in (1200, 2000):
            contents = [
                types.Content(
                    parts=[
                        types.Part(text=_PROMPT),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type, data=image_bytes
                            )
                        ),
                    ]
                )
            ]
            config = types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                max_output_tokens=max_output_tokens,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            )
            try:
                response = await self._client.aio.models.generate_content(
                    model=_MODEL,
                    contents=contents,
                    config=config,
                )
            except ClientError as exc:
                if exc.code == 429 or "RESOURCE_EXHAUSTED" in str(exc):
                    logger.warning(
                        "Gemini Vision respondió 429 — cuota agotada, se omite visión"
                    )
                    raise GeminiVisionQuotaError(
                        "HTTP 429 — Gemini Vision quota exhausted"
                    ) from exc
                logger.error("Gemini respondió error de cliente: %s", exc)
                raise RuntimeError(f"Gemini client error: {exc}") from exc
            except ServerError as exc:
                logger.error("Gemini respondió error de servidor: %s", exc)
                raise RuntimeError(f"Gemini server error: {exc}") from exc

            output_text = response.text or ""
            finish_reason = None
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
            logger.debug("Respuesta cruda de Gemini: %r", output_text[:300])

            try:
                extraction = parse_vision_json(output_text)
                break
            except GeminiJSONParseError as exc:
                last_error = exc
                logger.warning(
                    "Gemini devolvió JSON inválido o incompleto (finish_reason=%s, max_tokens=%s): %s",
                    finish_reason or "unknown",
                    max_output_tokens,
                    exc,
                )
                if max_output_tokens == 2000:
                    break

        if extraction is None:
            raise GeminiVisionResponseError(
                f"Gemini returned an unusable vision response: {last_error}"
            )

        logger.info("━" * 60)
        logger.info("📖 GEMINI EXTRACTION RESULT")
        logger.info("  is_book_cover : %s", extraction.is_book_cover)
        logger.info("  title         : %r", extraction.title)
        logger.info("  subtitle      : %r", extraction.subtitle)
        logger.info("  authors       : %s", extraction.authors)
        logger.info("  series        : %r", extraction.series_or_edition)
        logger.info("  language      : %r", extraction.language)
        logger.info("  confidence    : %.0f%%", extraction.confidence * 100)
        if not extraction.is_book_cover:
            logger.info("  reason        : %r", extraction.reason_if_not_book)
        logger.info("━" * 60)
        return extraction

    @staticmethod
    def _build_payload(
        encoded_image: str, mime_type: str, *, max_output_tokens: int
    ) -> dict[str, Any]:
        return {
            "contents": [
                {
                    "parts": [
                        {"text": _PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded_image,
                            },
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "maxOutputTokens": max_output_tokens,
                "mediaResolution": "MEDIA_RESOLUTION_HIGH",
            },
        }

    @staticmethod
    def _extract_text_output(response: dict[str, Any]) -> str:
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return "".join(
                    str(part.get("text", "")) for part in parts if part.get("text")
                )
        return ""

    @staticmethod
    def _extract_finish_reason(response: dict[str, Any]) -> str | None:
        candidates = response.get("candidates", [])
        if candidates:
            return candidates[0].get("finishReason")
        return None
