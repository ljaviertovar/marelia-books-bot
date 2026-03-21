from app.gemini.parser import GeminiJSONParseError, parse_vision_json
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
