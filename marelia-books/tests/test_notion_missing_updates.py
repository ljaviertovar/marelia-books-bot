from app.books.metadata import ResolvedBookMetadata
from app.notion.client import build_missing_update_properties



def test_update_only_missing_fields_logic():
    existing_properties = {
        "Author": {"rich_text": [{"plain_text": "Existing Author"}]},
        "Book Series": {"rich_text": []},
        "Cover": {"url": None},
        "Category": {"multi_select": []},
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
    assert updates["Cover"]["files"][0]["external"]["url"] == "https://example.com/cover.jpg"
    assert updates["Category"]["multi_select"][0]["name"] == "Fantasy"
    assert updates["Reading Type"]["select"]["name"] == "Physical"
    assert updates["Link"]["url"] == "https://example.com/book"
    assert "Status" not in updates
    assert "Score" not in updates
