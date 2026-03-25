from app.books.metadata import ResolvedBookMetadata
from app.gemini.enricher import GeminiEnricher


def test_ensure_series_in_tagline_mentions_series_and_order_naturally():
    metadata = ResolvedBookMetadata(
        title="Lo que el tiempo olvidó",
        author="Lorena Franco",
        series="Trilogía del tiempo",
        order_to_read=2,
        tagline="Lo que el tiempo olvidó es un thriller psicológico escrito por Lorena Franco.",
    )

    updated = GeminiEnricher._ensure_series_in_tagline(metadata)

    assert updated.tagline is not None
    assert "Trilogía del tiempo" in updated.tagline
    assert "libro 2" in updated.tagline


def test_build_prompt_explicitly_prioritizes_series_research():
    enricher = GeminiEnricher("test-key")
    metadata = ResolvedBookMetadata(
        title="Lo que el tiempo olvidó",
        author="Lorena Franco",
    )

    prompt = enricher._build_prompt(metadata, ["series", "order_to_read", "tagline"])

    assert "belongs to a series or saga" in prompt
    assert "Do not skip the series field" in prompt
    assert '"series":' in prompt
    assert '"order_to_read":' in prompt


def test_should_enrich_synopsis_when_openlibrary_text_is_too_short():
    assert GeminiEnricher._should_enrich_synopsis("Thriller psicológico escrito por Lorena Franco.") is True
    assert GeminiEnricher._should_enrich_synopsis("Una mujer descubre una verdad enterrada por años mientras su pasado y presente colisionan en una trama de secretos, tensión y consecuencias emocionales.") is False


def test_build_prompt_asks_to_improve_weak_existing_synopsis():
    enricher = GeminiEnricher("test-key")
    metadata = ResolvedBookMetadata(
        title="Lo que el tiempo olvidó",
        author="Lorena Franco",
        synopsis="Thriller psicológico escrito por Lorena Franco.",
    )

    prompt = enricher._build_prompt(metadata, ["synopsis"])

    assert 'If the existing synopsis is poor' in prompt or 'Si la sinopsis existente es pobre' in prompt
    assert "No inventes personajes" in prompt
    assert "No mezcles información de otros libros" in prompt


def test_build_prompt_warns_against_cross_book_synopsis_mixups():
    enricher = GeminiEnricher("test-key")
    metadata = ResolvedBookMetadata(
        title="Lo que el tiempo olvidó",
        author="Lorena Franco",
    )

    prompt = enricher._build_prompt(metadata, ["synopsis"])

    assert "Do not invent or borrow synopsis details from a different book" in prompt
    assert "Be strict about title-author consistency" in prompt
