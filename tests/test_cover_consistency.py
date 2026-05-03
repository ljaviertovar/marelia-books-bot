"""
Validates that whenever a book is created or updated, the three cover surfaces
are all set to the same URL:

  1. Page-level cover  (Notion page `cover` field)
  2. `Cover` file property  (database property that stores the cover image)
  3. Image block inside the page body  (first `image` block in the template)
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.books.metadata import ResolvedBookMetadata
from app.notion.client import NotionClient
from app.notion.utils import NotionBookRecord

COVER_URL = "https://covers.example/dune.jpg"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _MockResponse:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.status_code = 200
        self.is_error = False

    def json(self) -> dict:
        return self._data


def _heading(block_id: str, text: str) -> dict:
    return {
        "id": block_id,
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _image_block_dict(block_id: str, url: str = "") -> dict:
    return {
        "id": block_id,
        "type": "image",
        "image": {"type": "external", "external": {"url": url}, "caption": []},
    }


# Template blocks that include an image placeholder and the required section headings
_TEMPLATE_WITH_IMAGE_PLACEHOLDER = [
    _image_block_dict("img-placeholder", ""),
    _heading("notes-heading", "Book Notes"),
    _heading("synopsis-heading", "Synopsis"),
    _heading("refs-heading", "References / Links"),
]

# Template blocks with section headings but NO image block
_TEMPLATE_NO_IMAGE = [
    _heading("notes-heading", "Book Notes"),
    _heading("synopsis-heading", "Synopsis"),
    _heading("refs-heading", "References / Links"),
]


# ---------------------------------------------------------------------------
# Test 1 – create_book_page sets all three cover surfaces to the same URL
# ---------------------------------------------------------------------------

class _CreateCoverClient(NotionClient):
    """Minimal NotionClient stub for create_book_page tests."""

    def __init__(self) -> None:
        self.post_calls: list[tuple[str, dict]] = []
        self.patch_calls: list[tuple[str, dict]] = []
        self._template_id = "template-id"
        self._database_id = "database-id"

    async def _request_with_retry(
        self, method: str, url: str, *, json: dict | None = None, max_attempts: int = 4
    ) -> _MockResponse:
        if method == "POST":
            self.post_calls.append((url, json or {}))
            return _MockResponse({"id": "new-page-id"})
        if method == "PATCH":
            self.patch_calls.append((url, json or {}))
            return _MockResponse({})
        return _MockResponse({})

    async def _list_block_children(self, block_id: str) -> list[dict]:
        # Return template blocks with an image placeholder so the template
        # structure is detected immediately (no polling delay needed).
        return list(_TEMPLATE_WITH_IMAGE_PLACEHOLDER)

    async def _list_block_children_recursive(self, block_id: str) -> list[dict]:
        return list(_TEMPLATE_WITH_IMAGE_PLACEHOLDER)

    async def _list_block_children_tree(self, block_id: str) -> list[dict]:
        return list(_TEMPLATE_WITH_IMAGE_PLACEHOLDER)


def test_create_book_page_sets_page_cover_and_property_and_image_block_to_same_url() -> None:
    """All three cover surfaces in create_book_page must point to the same URL."""
    client = _CreateCoverClient()
    metadata = ResolvedBookMetadata(
        title="Dune",
        author="Frank Herbert",
        cover_url=COVER_URL,
    )

    page_id = asyncio.run(client.create_book_page(metadata))
    assert page_id == "new-page-id"

    # --- Surface 1: page-level cover field sent in the POST payload ----------
    assert client.post_calls, "expected a POST call to create the page"
    post_url, post_payload = client.post_calls[0]
    assert post_url.endswith("/pages"), f"unexpected POST url: {post_url}"
    page_cover = post_payload.get("cover", {})
    assert page_cover.get("type") == "external", "page cover must be external"
    assert page_cover["external"]["url"] == COVER_URL, (
        f"page cover URL mismatch: {page_cover['external']['url']!r} != {COVER_URL!r}"
    )

    # --- Surface 2: Cover file property sent in the same POST payload --------
    cover_prop = post_payload.get("properties", {}).get("Cover", {})
    assert cover_prop, "Cover property must be present in the POST payload"
    files = cover_prop.get("files", [])
    assert files, "Cover property must have at least one file entry"
    assert files[0]["external"]["url"] == COVER_URL, (
        f"Cover property URL mismatch: {files[0]['external']['url']!r} != {COVER_URL!r}"
    )

    # --- Surface 3: image block inside the page body updated via PATCH --------
    image_patch = next(
        (payload for url, payload in client.patch_calls if url.endswith("/blocks/img-placeholder")),
        None,
    )
    assert image_patch is not None, (
        "expected a PATCH to the image placeholder block; "
        f"PATCH calls were: {[url for url, _ in client.patch_calls]}"
    )
    assert image_patch.get("type") == "image"
    assert image_patch["image"]["external"]["url"] == COVER_URL, (
        f"image block URL mismatch: {image_patch['image']['external']['url']!r} != {COVER_URL!r}"
    )

    # --- All three must be identical -----------------------------------------
    assert (
        page_cover["external"]["url"]
        == files[0]["external"]["url"]
        == image_patch["image"]["external"]["url"]
        == COVER_URL
    ), "all three cover surfaces must carry the same URL"


# ---------------------------------------------------------------------------
# Test 2 – update_book_page_missing fills all three cover surfaces when absent
# ---------------------------------------------------------------------------

class _UpdateCoverClient(NotionClient):
    """Stub for update_book_page_missing tests, no existing cover anywhere."""

    def __init__(self) -> None:
        self.patch_calls: list[tuple[str, dict]] = []
        self._template_id = "template-id"
        self._database_id = "database-id"

    async def _request_with_retry(
        self, method: str, url: str, *, json: dict | None = None, max_attempts: int = 4
    ) -> _MockResponse:
        if method == "PATCH":
            self.patch_calls.append((url, json or {}))
        return _MockResponse({})

    async def _list_block_children(self, block_id: str) -> list[dict]:
        # Template structure present but no image block → triggers image insertion
        return list(_TEMPLATE_NO_IMAGE)

    async def _list_block_children_recursive(self, block_id: str) -> list[dict]:
        return list(_TEMPLATE_NO_IMAGE)

    async def _list_block_children_tree(self, block_id: str) -> list[dict]:
        return list(_TEMPLATE_NO_IMAGE)


def test_update_book_page_missing_fills_all_three_cover_surfaces_when_absent() -> None:
    """When no cover exists yet, update must set page cover, Cover property, and image block."""
    client = _UpdateCoverClient()

    record = NotionBookRecord(
        page_id="page-123",
        title="Dune",
        author="Frank Herbert",
        series=None,
        page_cover_url=None,  # no page cover
        _raw_properties={"Cover": {"files": []}},  # empty Cover property
    )
    metadata = ResolvedBookMetadata(
        title="Dune",
        author="Frank Herbert",
        cover_url=COVER_URL,
    )

    changed = asyncio.run(client.update_book_page_missing(record, metadata))
    assert changed is True

    # --- Surface 1 & 2: page cover + Cover property in the page PATCH --------
    page_patch = next(
        (payload for url, payload in client.patch_calls if url.endswith(f"/pages/{record.page_id}")),
        None,
    )
    assert page_patch is not None, (
        "expected a PATCH to the page; "
        f"PATCH calls: {[url for url, _ in client.patch_calls]}"
    )

    # Surface 1 – page-level cover
    page_cover = page_patch.get("cover", {})
    assert page_cover.get("type") == "external", "page cover must be external"
    assert page_cover["external"]["url"] == COVER_URL

    # Surface 2 – Cover property
    cover_prop = page_patch.get("properties", {}).get("Cover", {})
    assert cover_prop, "Cover property must be present in the page PATCH"
    files = cover_prop.get("files", [])
    assert files, "Cover property must have at least one file entry"
    assert files[0]["external"]["url"] == COVER_URL

    # --- Surface 3: image block inserted via children PATCH ------------------
    children_patches = [
        payload
        for url, payload in client.patch_calls
        if url.endswith(f"/blocks/{record.page_id}/children")
    ]
    assert children_patches, (
        "expected a PATCH to insert an image block; "
        f"PATCH calls: {[url for url, _ in client.patch_calls]}"
    )
    image_children = [
        child
        for payload in children_patches
        for child in payload.get("children", [])
        if child.get("type") == "image"
    ]
    assert image_children, "at least one image block must be inserted in children PATCH"
    image_url = image_children[0]["image"]["external"]["url"]
    assert image_url == COVER_URL, f"image block URL mismatch: {image_url!r} != {COVER_URL!r}"

    # --- All three must be identical -----------------------------------------
    assert (
        page_cover["external"]["url"]
        == files[0]["external"]["url"]
        == image_url
        == COVER_URL
    ), "all three cover surfaces must carry the same URL"


# ---------------------------------------------------------------------------
# Test 3 – update always overwrites all three cover surfaces, even when set
# ---------------------------------------------------------------------------

def test_update_book_page_missing_always_syncs_all_three_cover_surfaces() -> None:
    """When a new cover_url arrives during an update, ALL three cover surfaces must
    be overwritten — even if the page already had a cover or property value."""
    client = _UpdateCoverClient()

    record = NotionBookRecord(
        page_id="page-456",
        title="Dune",
        author="Frank Herbert",
        series=None,
        page_cover_url="https://existing.example/old-cover.jpg",  # already set
        _raw_properties={"Cover": {"files": [{"type": "external", "name": "Cover", "external": {"url": "https://old.example/cover.jpg"}}]}},
    )
    metadata = ResolvedBookMetadata(
        title="Dune",
        author="Frank Herbert",
        cover_url=COVER_URL,
    )

    changed = asyncio.run(client.update_book_page_missing(record, metadata))
    assert changed is True

    page_patch = next(
        (payload for url, payload in client.patch_calls if url.endswith(f"/pages/{record.page_id}")),
        None,
    )
    assert page_patch is not None, "expected a PATCH to the page"

    # Surface 1 – page cover must be updated to the new URL
    page_cover = page_patch.get("cover", {})
    assert page_cover.get("type") == "external", "page cover must be set"
    assert page_cover["external"]["url"] == COVER_URL, (
        f"page cover must be overwritten with the new URL, got {page_cover['external']['url']!r}"
    )

    # Surface 2 – Cover property must be updated to the new URL
    cover_prop = page_patch.get("properties", {}).get("Cover", {})
    assert cover_prop, "Cover property must always be synced when cover_url is provided"
    files = cover_prop.get("files", [])
    assert files[0]["external"]["url"] == COVER_URL, (
        f"Cover property must be overwritten with the new URL, got {files[0]['external']['url']!r}"
    )

    # Surface 3 – image block must still be inserted (no existing image in _TEMPLATE_NO_IMAGE)
    children_patches = [
        payload
        for url, payload in client.patch_calls
        if url.endswith(f"/blocks/{record.page_id}/children")
    ]
    assert children_patches, "image block must be inserted"
    image_children = [
        child
        for payload in children_patches
        for child in payload.get("children", [])
        if child.get("type") == "image"
    ]
    assert image_children, "at least one image block must be inserted"
    assert image_children[0]["image"]["external"]["url"] == COVER_URL

    # All three agree
    assert (
        page_cover["external"]["url"]
        == files[0]["external"]["url"]
        == image_children[0]["image"]["external"]["url"]
        == COVER_URL
    )
