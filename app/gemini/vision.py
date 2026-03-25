from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from app.books.metadata import VisionBookExtraction
from app.gemini.parser import GeminiJSONParseError, parse_vision_json

logger = logging.getLogger(__name__)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"


class GeminiVisionQuotaError(RuntimeError):
    """Raised when Gemini Vision quota is exhausted (HTTP 429)."""


class GeminiVisionResponseError(RuntimeError):
    """Raised when Gemini Vision returns an unusable response."""


class GeminiVisionClient:
    def __init__(self, api_key: str, timeout_seconds: float = 30.0) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Content-Type": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def extract_book_data(self, image_bytes: bytes, mime_type: str) -> VisionBookExtraction:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        logger.info("Enviando imagen a Gemini (%d bytes)", len(image_bytes))
        extraction: VisionBookExtraction | None = None
        last_error: Exception | None = None

        for max_output_tokens in (1200, 2000):
            payload = self._build_payload(encoded, mime_type, max_output_tokens=max_output_tokens)
            url = f"{_GEMINI_URL}?key={self._api_key}"
            response = await self._request_with_retry(url, payload)
            data = response.json()
            output_text = self._extract_text_output(data)
            finish_reason = self._extract_finish_reason(data)
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
            raise GeminiVisionResponseError(f"Gemini returned an unusable vision response: {last_error}")

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
    def _build_payload(encoded_image: str, mime_type: str, *, max_output_tokens: int) -> dict[str, Any]:
        return {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
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
                                "{\"is_book_cover\": true, "
                                "\"title\": \"string | null\", "
                                "\"subtitle\": \"string | null\", "
                                "\"authors\": [\"string\"], "
                                "\"series_or_edition\": \"string | null\", "
                                "\"language\": \"string | null\", "
                                "\"confidence\": 0.0, "
                                "\"reason_if_not_book\": \"string | null\", "
                                "\"raw_visible_text\": \"string | null\"}"
                            ),
                        },
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
                "media_resolution": "MEDIA_RESOLUTION_HIGH",
            },
        }

    async def _request_with_retry(self, url: str, payload: dict[str, Any], max_attempts: int = 4) -> httpx.Response:
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.post(url, json=payload)
                if response.status_code == 429:
                    logger.warning("Gemini Vision respondió 429 — cuota agotada, se omite visión")
                    raise GeminiVisionQuotaError("HTTP 429 — Gemini Vision quota exhausted")

                if response.status_code in (500, 502, 503, 504):
                    retry_after = float(response.headers.get("Retry-After", delay))
                    logger.warning(
                        "Gemini respondió %s — reintentando (intento %s, esperando %ss)",
                        response.status_code,
                        attempt,
                        retry_after,
                    )
                    last_error = RuntimeError(f"HTTP {response.status_code} después de {attempt} intento(s)")
                    if attempt == max_attempts:
                        break
                    await self._sleep(retry_after)
                    delay = max(delay * 2, retry_after)
                    continue

                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                await self._sleep(delay)
                delay *= 2

        raise RuntimeError(f"Gemini request failed after retries: {last_error}")

    @staticmethod
    async def _sleep(seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)

    @staticmethod
    def _extract_text_output(response: dict[str, Any]) -> str:
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return "".join(str(part.get("text", "")) for part in parts if part.get("text"))
        return ""

    @staticmethod
    def _extract_finish_reason(response: dict[str, Any]) -> str | None:
        candidates = response.get("candidates", [])
        if candidates:
            return candidates[0].get("finishReason")
        return None
