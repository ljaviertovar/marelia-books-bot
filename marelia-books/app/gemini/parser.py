from __future__ import annotations

import json

from app.books.metadata import VisionBookExtraction



def parse_vision_json(raw_text: str) -> VisionBookExtraction:
    text = raw_text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

    parsed = json.loads(text)
    return VisionBookExtraction.model_validate(parsed)
