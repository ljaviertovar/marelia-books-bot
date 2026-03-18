from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.books.metadata import ResolvedBookMetadata

logger = logging.getLogger(__name__)

NOTION_VERSION = "2022-06-28"



def _title_prop(text: str) -> dict[str, Any]:
    return {"title": [{"text": {"content": text}}]}



def _rich_text_prop(text: str) -> dict[str, Any]:
    return {"rich_text": [{"text": {"content": text}}]}



def _url_prop(url: str) -> dict[str, Any]:
    return {"url": url}



def _select_prop(name: str) -> dict[str, Any]:
    return {"select": {"name": name}}



def _status_prop(name: str) -> dict[str, Any]:
    return {"status": {"name": name}}



def _multi_select_prop(values: list[str]) -> dict[str, Any]:
    return {"multi_select": [{"name": item} for item in values]}



def _extract_title(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    items = value.get("title") or []
    if not items:
        return None
    return (items[0].get("plain_text") or "").strip() or None



def _extract_rich_text(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    items = value.get("rich_text") or []
    if not items:
        return None
    return (items[0].get("plain_text") or "").strip() or None



def get_page_title(page: dict[str, Any]) -> str | None:
    return _extract_title((page.get("properties") or {}).get("Book Name"))



def get_page_author(page: dict[str, Any]) -> str | None:
    return _extract_rich_text((page.get("properties") or {}).get("Author"))



def _is_empty_property(prop_name: str, prop_value: dict[str, Any] | None) -> bool:
    if not prop_value:
        return True

    if prop_name in {"Author", "Book Series"}:
        return not (prop_value.get("rich_text") or [])
    if prop_name == "Cover":
        return not prop_value.get("url")
    if prop_name == "Category":
        return not (prop_value.get("multi_select") or [])
    if prop_name in {"Reading Type", "Type"}:
        return not (prop_value.get("select") or {}).get("name")
    if prop_name == "Link":
        return not prop_value.get("url")

    return False



def build_create_properties(metadata: ResolvedBookMetadata) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "Book Name": _title_prop(metadata.title),
        "Status": _status_prop("Not started"),
        "Type": _select_prop("Book"),
    }

    if metadata.author:
        properties["Author"] = _rich_text_prop(metadata.author)
    if metadata.series:
        properties["Book Series"] = _rich_text_prop(metadata.series)
    if metadata.cover_url:
        properties["Cover"] = _url_prop(metadata.cover_url)
    if metadata.categories:
        properties["Category"] = _multi_select_prop(metadata.categories)
    if metadata.reading_type:
        properties["Reading Type"] = _select_prop(metadata.reading_type)
    if metadata.link:
        properties["Link"] = _url_prop(metadata.link)

    return properties



def build_missing_update_properties(existing_page: dict[str, Any], metadata: ResolvedBookMetadata) -> dict[str, Any]:
    existing = existing_page.get("properties") or {}
    updates: dict[str, Any] = {}

    if metadata.author and _is_empty_property("Author", existing.get("Author")):
        updates["Author"] = _rich_text_prop(metadata.author)
    if metadata.series and _is_empty_property("Book Series", existing.get("Book Series")):
        updates["Book Series"] = _rich_text_prop(metadata.series)
    if metadata.cover_url and _is_empty_property("Cover", existing.get("Cover")):
        updates["Cover"] = _url_prop(metadata.cover_url)
    if metadata.categories and _is_empty_property("Category", existing.get("Category")):
        updates["Category"] = _multi_select_prop(metadata.categories)
    if metadata.reading_type and _is_empty_property("Reading Type", existing.get("Reading Type")):
        updates["Reading Type"] = _select_prop(metadata.reading_type)
    if metadata.type and _is_empty_property("Type", existing.get("Type")):
        updates["Type"] = _select_prop(metadata.type)
    if metadata.link and _is_empty_property("Link", existing.get("Link")):
        updates["Link"] = _url_prop(metadata.link)

    return updates


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

    async def query_candidate_books(self, title: str) -> list[dict[str, Any]]:
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
        results = response.json().get("results", [])
        logger.info("Notion: %d resultado(s) para %r", len(results), title[:50])
        return results

    async def create_book_page(self, metadata: ResolvedBookMetadata) -> str:
        logger.info("Creando página en Notion: '%s' de %s", metadata.title, metadata.author or "autor desconocido")
        payload = {
            "parent": {"database_id": self._database_id},
            "template": {
                "type": "template_id",
                "template_id": self._template_id,
            },
            "properties": build_create_properties(metadata),
        }
        response = await self._request_with_retry("POST", "https://api.notion.com/v1/pages", json=payload)
        page_id = response.json()["id"]
        logger.info("Página creada exitosamente en Notion [id=%s]", page_id)
        return page_id

    async def update_book_page_missing(self, page: dict[str, Any], metadata: ResolvedBookMetadata) -> bool:
        updates = build_missing_update_properties(page, metadata)
        if not updates:
            logger.debug("Sin campos que actualizar en Notion [page_id=%s]", page.get("id"))
            return False

        page_id = page["id"]
        logger.info("Actualizando campos en Notion: %s [page_id=%s]", list(updates.keys()), page_id)
        await self._request_with_retry(
            "PATCH",
            f"https://api.notion.com/v1/pages/{page_id}",
            json={"properties": updates},
        )
        logger.info("Campos actualizados en Notion [page_id=%s]", page_id)
        return True

    async def _request_with_retry(self, method: str, url: str, *, json: dict[str, Any], max_attempts: int = 4) -> httpx.Response:
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

                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(f"Notion request failed after retries: {last_error}")
