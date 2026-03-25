from app.gemini.parser import GeminiJSONParseError, parse_enrichment_json, parse_vision_json
from app.gemini.vision import GeminiVisionClient


def test_parse_vision_json_raises_clear_error_on_truncated_json():
    raw_text = '{\n  "is_'

    try:
        parse_vision_json(raw_text)
    except GeminiJSONParseError as exc:
        assert "Raw preview" in str(exc)
    else:
        raise AssertionError("Expected GeminiJSONParseError for truncated JSON")


def test_extract_text_output_concatenates_all_text_parts():
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": '{\n  "is_book_cover": true,'},
                        {"text": '\n  "title": "Dune",\n  "confidence": 0.98\n}'},
                    ]
                }
            }
        ]
    }

    combined = GeminiVisionClient._extract_text_output(response)

    assert '"title": "Dune"' in combined
    assert combined.startswith("{")
    assert combined.endswith("}")


def test_parse_enrichment_json_accepts_textual_numbers():
    parsed = parse_enrichment_json(
        """{
            "series": "Saga del tiempo",
            "order_to_read": "Libro 2",
            "pages": "384 paginas"
        }"""
    )

    assert parsed["series"] == "Saga del tiempo"
    assert parsed["order_to_read"] == 2
    assert parsed["pages"] == 384


def test_parse_enrichment_json_accepts_common_alias_keys():
    parsed = parse_enrichment_json(
        """{
            "genre": "Thriller psicológico",
            "series_name": "Trilogía del tiempo",
            "reading_order": "2"
        }"""
    )

    assert parsed["genre_es"] == "Thriller psicológico"
    assert parsed["series"] == "Trilogía del tiempo"
    assert parsed["order_to_read"] == 2


def test_vision_prompt_explicitly_requests_series_clues_from_cover():
    payload = GeminiVisionClient._build_payload("ZmFrZQ==", "image/jpeg", max_output_tokens=1200)
    prompt = payload["contents"][0]["parts"][0]["text"]

    assert "series_or_edition" in prompt
    assert "If the cover mentions a saga, trilogy, series name" in prompt
    assert "Prefer the real series name over marketing labels" in prompt
