from app.books.metadata import ResolvedBookMetadata
from app.notion.client import (
    build_create_properties,
    build_section_content,
    plan_section_appends,
    plan_template_block_updates,
)


def _heading(block_id: str, text: str) -> dict:
    return {
        "id": block_id,
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _paragraph(block_id: str, text: str) -> dict:
    return {
        "id": block_id,
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _callout(block_id: str, text: str) -> dict:
    return {
        "id": block_id,
        "type": "callout",
        "callout": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _image(block_id: str, url: str = "") -> dict:
    return {
        "id": block_id,
        "type": "image",
        "image": {"type": "external", "external": {"url": url}, "caption": []},
    }


def test_build_section_content_uses_richer_page_blocks():
    metadata = ResolvedBookMetadata(
        title="Dune",
        author="Frank Herbert",
        series="Dune",
        title_es="Duna",
        genre_es="Ciencia ficcion",
        synopsis="Un heredero queda atrapado en el centro de una lucha por poder y especia.",
        publisher_url="https://publisher.example/dune",
        tagline="Dune es una novela de ciencia ficcion de Frank Herbert.",
        year=1965,
        language="ingles",
        publisher="Chilton Books",
        isbn="9780441172719",
        pages=412,
    )

    content = build_section_content(metadata)

    assert set(content) == {"notes", "synopsis", "references"}
    assert content["notes"][0]["type"] == "callout"
    assert content["notes"][1]["bulleted_list_item"]["rich_text"][0]["text"]["content"] == "Titulo en espanol: Duna"
    assert content["synopsis"][0]["paragraph"]["rich_text"][0]["text"]["content"] == metadata.synopsis
    ref_text = content["references"][0]["bulleted_list_item"]["rich_text"][0]["text"]
    assert ref_text["content"] == metadata.publisher_url
    assert ref_text["link"]["url"] == metadata.publisher_url


def test_build_create_properties_infers_category_from_genre_when_missing():
    metadata = ResolvedBookMetadata(
        title="Proyecto Hail Mary",
        categories=[],
        genre_es="Ciencia ficcion (Dura)",
    )

    props = build_create_properties(metadata)

    assert props["Category"]["multi_select"][0]["name"] == "Sci-Fi"


def test_plan_section_appends_uses_new_page_headings_not_template_source():
    page_blocks = [
        _heading("h1", "Book Notes"),
        _paragraph("p1", "Existing note placeholder"),
        _heading("h2", "Synopsis"),
        _paragraph("p2", ""),
        _heading("h3", "Links"),
    ]
    metadata = ResolvedBookMetadata(
        title="Dune",
        synopsis="Sin spoilers.",
        publisher_url="https://publisher.example/dune",
        tagline="A classic sci-fi novel.",
    )

    plans = plan_section_appends(page_blocks, metadata)

    assert [plan.section for plan in plans] == ["notes", "synopsis", "references"]
    assert plans[0].after_block_id == "p1"
    assert plans[1].after_block_id == "p2"
    assert plans[2].after_block_id == "h3"
    assert plans[0].children[0]["type"] == "callout"


def test_plan_section_appends_skips_missing_sections():
    page_blocks = [
        _heading("h1", "Synopsis"),
        _paragraph("p1", ""),
    ]
    metadata = ResolvedBookMetadata(
        title="Dune",
        tagline="A classic sci-fi novel.",
        synopsis="Sin spoilers.",
        publisher_url="https://publisher.example/dune",
    )

    plans = plan_section_appends(page_blocks, metadata)

    assert [plan.section for plan in plans] == ["synopsis"]


def test_plan_template_block_updates_fills_existing_placeholders_in_place():
    page_blocks = [
        _heading("h1", "Notes"),
        _paragraph("n1", "Title (ES):"),
        _paragraph("n2", "Original Title:"),
        _paragraph("n3", "Category:"),
        _heading("h2", "Synopsis (no spoilers)"),
        _paragraph("s1", ""),
        _heading("h3", "References / Links"),
        _paragraph("r1", ""),
    ]
    metadata = ResolvedBookMetadata(
        title="Project Hail Mary",
        title_es="Proyecto Hail Mary",
        genre_es="Ciencia ficcion",
        synopsis="Sinopsis breve.",
        publisher_url="https://example.com/hail-mary",
    )

    updates, filled_sections, placeholder_sections = plan_template_block_updates(page_blocks, metadata)

    ids = {u.block_id for u in updates}
    assert ids == {"n1", "n2", "n3", "s1", "r1"}
    assert filled_sections == {"notes", "synopsis", "references"}
    assert placeholder_sections == {"notes", "synopsis", "references"}


def test_plan_template_block_updates_includes_tagline_and_cover_updates():
    page_blocks = [
        _callout("c1", "Notion Tip: Use this page..."),
        _image("img1"),
        _heading("h1", "Notes"),
        _paragraph("n1", "Title (ES):"),
    ]
    metadata = ResolvedBookMetadata(
        title="Project Hail Mary",
        title_es="Proyecto Hail Mary",
        tagline="Proyecto Hail Mary es una novela de ciencia ficcion escrita por Andy Weir.",
        cover_url="https://covers.example/project-hail-mary.jpg",
    )

    updates, _, _ = plan_template_block_updates(page_blocks, metadata)
    by_id = {u.block_id: u.payload for u in updates}

    assert by_id["c1"]["type"] == "callout"
    assert by_id["c1"]["callout"]["rich_text"][0]["text"]["content"] == metadata.tagline
    assert by_id["img1"]["type"] == "image"
    assert by_id["img1"]["image"]["external"]["url"] == metadata.cover_url
    assert "type" not in by_id["img1"]["image"]


def test_plan_template_block_updates_replaces_first_image_even_with_placeholder_url():
    page_blocks = [
        _image("img1", "https://notion.so/placeholder-image.png"),
    ]
    metadata = ResolvedBookMetadata(
        title="Project Hail Mary",
        cover_url="https://covers.example/project-hail-mary.jpg",
    )

    updates, _, _ = plan_template_block_updates(page_blocks, metadata)
    by_id = {u.block_id: u.payload for u in updates}

    assert by_id["img1"]["image"]["external"]["url"] == metadata.cover_url
    assert "type" not in by_id["img1"]["image"]
