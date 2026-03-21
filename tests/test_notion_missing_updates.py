import asyncio

from app.books.metadata import ResolvedBookMetadata
from app.notion.client import NotionClient, build_missing_update_properties
from app.notion.utils import NotionBookRecord



def test_update_only_missing_fields_logic():
    existing_properties = {
        "Author": {"rich_text": [{"plain_text": "Existing Author"}]},
        "Book Series": {"rich_text": []},
        "Order to Read": {"number": None},
        "Cover": {"url": None},
        "Genre": {"multi_select": []},
        "Reading Type": {"select": {"name": None}},
        "Type": {"select": {"name": "Book"}},
        "Link": {"url": None},
        "Status": {"status": {"name": "In progress"}},
        "Score": {"select": {"name": "5"}},
    }

    metadata = ResolvedBookMetadata(
        title="Example",
        author="New Author",
        series="Series A",
        order_to_read=2,
        cover_url="https://example.com/cover.jpg",
        categories=["Fantasy"],
        reading_type="Physical",
        link="https://example.com/book",
        type="Book",
    )

    updates = build_missing_update_properties(existing_properties, metadata)

    assert "Author" not in updates
    assert "Type" not in updates
    assert updates["Book Series"]["rich_text"][0]["text"]["content"] == "Series A"
    assert updates["Order to Read"]["number"] == 2
    assert updates["Cover"]["files"][0]["external"]["url"] == "https://example.com/cover.jpg"
    assert updates["Genre"]["multi_select"][0]["name"] == "Fantasy"
    assert updates["Reading Type"]["select"]["name"] == "Physical"
    assert updates["Link"]["url"] == "https://example.com/book"
    assert "Status" not in updates
    assert "Score" not in updates


def test_update_existing_book_adds_template_structure_before_filling_missing_data():
    class _TestNotionClient(NotionClient):
        def __init__(self) -> None:
            self.patch_calls: list[tuple[str, dict]] = []
            self._children_calls = 0

        async def _request_with_retry(self, method: str, url: str, *, json: dict | None = None, max_attempts: int = 4):
            if method == "PATCH":
                self.patch_calls.append((url, json or {}))
            return None

        async def _list_block_children(self, block_id: str) -> list[dict]:
            self._children_calls += 1
            if self._children_calls == 1:
                return []
            return [
                {
                    "id": "notes-heading",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Book Notes"}}]},
                },
                {
                    "id": "synopsis-heading",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Synopsis"}}]},
                },
                {
                    "id": "refs-heading",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": "References / Links"}}]},
                },
            ]

        async def _list_block_children_recursive(self, block_id: str) -> list[dict]:
            return await self._list_block_children(block_id)

    client = _TestNotionClient()
    record = NotionBookRecord(
        page_id="page-123",
        title="Dune",
        author="Frank Herbert",
        series=None,
        _raw_properties={},
    )
    metadata = ResolvedBookMetadata(
        title="Dune",
        author="Frank Herbert",
        tagline="Dune es una novela de ciencia ficcion de Frank Herbert.",
        synopsis="Sin spoilers.",
    )

    changed = asyncio.run(client.update_book_page_missing(record, metadata))

    assert changed is True
    assert len(client.patch_calls) >= 1
    url, payload = client.patch_calls[0]
    assert url.endswith("/blocks/page-123/children")
    children = payload["children"]
    assert children[0]["heading_2"]["rich_text"][0]["text"]["content"] == "Book Notes"
    assert children[3]["heading_2"]["rich_text"][0]["text"]["content"] == "Synopsis"
