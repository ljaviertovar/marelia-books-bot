from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.books.metadata import ResolvedBookMetadata, sanitize_series_name
from app.gemini.parser import parse_enrichment_json

logger = logging.getLogger(__name__)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"


class GeminiEnricher:
    """Enriches book metadata using Gemini when OpenLibrary data is incomplete."""

    def __init__(self, api_key: str, timeout_seconds: float = 30.0) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Content-Type": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def enrich(self, metadata: ResolvedBookMetadata) -> ResolvedBookMetadata:
        """Call Gemini to fill in text fields missing from OpenLibrary."""
        missing = [
            k for k in ("title_es", "genre_es", "synopsis", "publisher_url", "tagline", "isbn", "pages", "order_to_read")
            if getattr(metadata, k) is None
        ]
        if sanitize_series_name(metadata.series) is None:
            missing.append("series")
        if not missing:
            logger.debug("Todos los campos de enriquecimiento ya están presentes — se omite Gemini")
            return metadata

        prompt = self._build_prompt(metadata, missing)
        logger.info("Enriqueciendo con Gemini: %s", missing)

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }

        url = f"{_GEMINI_URL}?key={self._api_key}"
        try:
            response = await self._request_with_retry(url, payload)
            data = response.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            logger.debug("Respuesta cruda de Gemini (enricher):\n%s", raw_text)
            enriched = parse_enrichment_json(raw_text)
            updates = {k: v for k, v in enriched.items() if v is not None}
            if "series" in updates:
                updates["series"] = sanitize_series_name(updates["series"])
            updated = metadata.model_copy(update=updates)
            updated = self._ensure_series_in_tagline(updated)
        except Exception as exc:
            logger.warning("Gemini enricher falló — el libro se creará sin enriquecimiento: %s", exc)
            return metadata

        logger.info(
            "Enriquecimiento completado: title_es=%r genre_es=%r synopsis_len=%s publisher_url=%r tagline=%r isbn=%r pages=%r series=%r order_to_read=%r",
            updated.title_es,
            updated.genre_es,
            len(updated.synopsis) if updated.synopsis else 0,
            updated.publisher_url,
            updated.tagline,
            updated.isbn,
            updated.pages,
            updated.series,
            updated.order_to_read,
        )
        return updated

    @staticmethod
    def _ensure_series_in_tagline(metadata: ResolvedBookMetadata) -> ResolvedBookMetadata:
        if not metadata.series or not metadata.tagline:
            return metadata

        tagline_lower = metadata.tagline.lower()
        series_lower = metadata.series.lower()
        if series_lower in tagline_lower:
            return metadata

        connector = " It belongs to the "
        if metadata.tagline.endswith((".", "!", "?")):
            updated_tagline = metadata.tagline + connector + metadata.series + "."
        else:
            updated_tagline = metadata.tagline + "." + connector + metadata.series + "."

        return metadata.model_copy(update={"tagline": updated_tagline})

    def _build_prompt(self, metadata: ResolvedBookMetadata, missing: list[str]) -> str:
        known_parts = [
            f'- Title: "{metadata.title}"',
        ]
        if metadata.author:
            known_parts.append(f'- Author: "{metadata.author}"')
        if metadata.year:
            known_parts.append(f'- First published: {metadata.year}')
        if metadata.publisher:
            known_parts.append(f'- Publisher: "{metadata.publisher}"')
        if metadata.language:
            known_parts.append(f'- Original language: "{metadata.language}"')
        if metadata.isbn:
            known_parts.append(f'- ISBN: {metadata.isbn}')
        if metadata.pages:
            known_parts.append(f'- Pages: {metadata.pages}')
        if metadata.categories:
            known_parts.append(f'- Categories: {", ".join(metadata.categories)}')
        if metadata.series:
            known_parts.append(f'- Existing series hint: "{metadata.series}"')
        if metadata.order_to_read:
            known_parts.append(f'- Existing order in series: {metadata.order_to_read}')

        fields_desc: list[str] = []
        if "title_es" in missing:
            fields_desc.append(
                '"title_es": "Título del libro en español. Si el título es en inglés, tradúcelo al español de manera natural. '
                'Si ya existe un título oficial en español, úsalo. Si no, traduce literalmente."'
            )
        if "genre_es" in missing:
            fields_desc.append(
                '"genre_es": "Género del libro en español, en formato \'Género (subtipo)\' si aplica. '
                'Ejemplo: \'Ciencia ficción (antología)\', \'Fantasía épica\', \'No ficción (ensayo)\'."'
            )
        if "synopsis" in missing:
            fields_desc.append(
                '"synopsis": "Sinopsis del libro en español, sin spoilers, máximo 100 palabras. '
                'Describe la premisa y el tono sin revelar el desenlace. '
                'Escríbela en una sola línea, sin saltos de línea."'
            )
        if "publisher_url" in missing:
            fields_desc.append(
                '"publisher_url": "URL de la página oficial del libro en el sitio web de la editorial. '
                'Si no existe una URL conocida, devuelve null."'
            )
        if "tagline" in missing:
            fields_desc.append(
                '"tagline": "Una descripción breve del libro en español, de 1 a 2 oraciones naturales. '
                'Debe mencionar el título, el género/tipo de obra y el autor o editor. '
                'Si el libro pertenece a una serie o saga, menciónala de forma natural en el texto. '
                'Ejemplo: \'Dune es una novela de ciencia ficción escrita por Frank Herbert.\' '
                'No uses comillas en el texto de salida."'
            )
        if "isbn" in missing:
            fields_desc.append(
                '"isbn": "ISBN principal del libro o de una edición ampliamente disponible. Si no hay uno confiable, devuelve null."'
            )
        if "pages" in missing:
            fields_desc.append(
                '"pages": "Número aproximado de páginas de una edición común del libro. Devuelve un entero o null si no es confiable."'
            )
        if "series" in missing:
            fields_desc.append(
                '"series": "Nombre de la serie o saga a la que pertenece el libro, si aplica. '
                'Devuelve solo el nombre real de la serie, por ejemplo \'Foundation series\'. '
                'No devuelvas etiquetas comerciales como \'Best Seller\', \'Bestseller\', '
                '\'New York Times Bestseller\', premios, ediciones o slogans. '
                'Si el libro no pertenece a una serie clara, devuelve null."'
            )
        if "order_to_read" in missing:
            fields_desc.append(
                '"order_to_read": "Número de lectura del libro dentro de su serie o saga, por ejemplo 1, 2, 3. '
                'Devuelve un entero si es claro y confiable. Si no pertenece a una serie clara o el orden no es confiable, devuelve null."'
            )

        fields_json = ",\n  ".join(fields_desc)

        return (
            "You are a book data research assistant. "
            "Given the following known book metadata, fill in the missing fields. "
            "Return ONLY valid JSON, no markdown, no explanation.\n\n"
            "Known book data:\n" + "\n".join(known_parts) + "\n\n"
            "Return exactly this JSON schema:\n"
            "{\n  " + fields_json + "\n}"
        )

    async def _request_with_retry(self, url: str, payload: dict[str, Any], max_attempts: int = 4) -> httpx.Response:
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.post(url, json=payload)

                if response.status_code == 429:
                    # Quota exhausted — retrying won't help within the same minute
                    logger.warning("Gemini (enricher) respondió 429 — cuota agotada, se omite el enriquecimiento")
                    raise RuntimeError("HTTP 429 — cuota de Gemini agotada")

                if response.status_code in (500, 502, 503, 504):
                    retry_after = float(response.headers.get("Retry-After", delay))
                    logger.warning(
                        "Gemini (enricher) respondió %s — reintentando (intento %s, esperando %ss)",
                        response.status_code,
                        attempt,
                        retry_after,
                    )
                    last_error = RuntimeError(f"HTTP {response.status_code} después de {attempt} intento(s)")
                    if attempt == max_attempts:
                        break
                    await asyncio.sleep(retry_after)
                    delay = max(delay * 2, retry_after)
                    continue

                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(f"Gemini enricher request failed after retries: {last_error}")
