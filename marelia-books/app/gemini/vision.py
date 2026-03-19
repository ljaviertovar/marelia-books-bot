from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from app.books.metadata import VisionBookExtraction
from app.gemini.parser import parse_vision_json

logger = logging.getLogger(__name__)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


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

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are a strict JSON extractor for book covers. "
                                "Return JSON only, no markdown, no explanation. "
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
                                "data": encoded,
                            },
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 500,
            },
        }

        url = f"{_GEMINI_URL}?key={self._api_key}"
        logger.info("Enviando imagen a Gemini (%d bytes)", len(image_bytes))
        response = await self._request_with_retry(url, payload)
        data = response.json()
        output_text = self._extract_text_output(data)
        logger.debug("Respuesta cruda de Gemini: %r", output_text[:300])
        extraction = parse_vision_json(output_text)
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

    async def _request_with_retry(self, url: str, payload: dict[str, Any], max_attempts: int = 4) -> httpx.Response:
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.post(url, json=payload)
                if response.status_code in (429, 500, 502, 503, 504):
                    retry_after = float(response.headers.get("Retry-After", delay))
                    logger.warning(
                        "Gemini respondió %s — reintentando (intento %s, esperando %ss)",
                        response.status_code,
                        attempt,
                        retry_after,
                    )
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
                return parts[0].get("text", "")
        return ""
