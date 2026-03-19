from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from app.books.metadata import VisionBookExtraction
from app.openai.parser import parse_vision_json

logger = logging.getLogger(__name__)


class OpenAIVisionClient:
    def __init__(self, api_key: str, timeout_seconds: float = 30.0) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def extract_book_data(self, image_bytes: bytes, mime_type: str) -> VisionBookExtraction:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"

        payload = {
            "model": "gpt-4.1-mini",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
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
                            "type": "input_image",
                            "image_url": data_url,
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_output_tokens": 500,
        }

        logger.info("Enviando imagen a OpenAI (%d bytes)", len(image_bytes))
        response = await self._request_with_retry("https://api.openai.com/v1/responses", payload)
        data = response.json()
        output_text = self._extract_text_output(data)
        logger.debug("Respuesta cruda de OpenAI: %r", output_text[:300])
        extraction = parse_vision_json(output_text)
        logger.info("━" * 60)
        logger.info("📖 OPENAI EXTRACTION RESULT")
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
                        "OpenAI respondió %s — reintentando (intento %s, esperando %ss)",
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

        raise RuntimeError(f"OpenAI request failed after retries: {last_error}")

    @staticmethod
    async def _sleep(seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)

    @staticmethod
    def _extract_text_output(response: dict[str, Any]) -> str:
        outputs = response.get("output", [])
        for item in outputs:
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
        return response.get("output_text", "")
