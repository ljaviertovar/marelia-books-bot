from __future__ import annotations

import re
from typing import Iterable


_SPACE_RE = re.compile(r"\s+")



def normalize_book_text(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.lower().strip()
    cleaned = cleaned.replace(":", " ").replace("-", " ")
    cleaned = _SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()



def make_book_key(title: str | None, author: str | None) -> str:
    return f"{normalize_book_text(title)}|{normalize_book_text(author)}"



def find_matching_page(
    candidates: Iterable[dict],
    incoming_title: str | None,
    incoming_author: str | None,
    *,
    get_title,
    get_author,
) -> dict | None:
    incoming_key = make_book_key(incoming_title, incoming_author)
    if incoming_key == "|":
        return None

    for page in candidates:
        candidate_key = make_book_key(get_title(page), get_author(page))
        if candidate_key == incoming_key:
            return page
    return None
