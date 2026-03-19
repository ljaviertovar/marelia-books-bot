from __future__ import annotations

import re
from typing import Any, Iterable


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
    candidates: Iterable[Any],
    incoming_title: str | None,
    incoming_author: str | None,
) -> Any | None:
    """Find a matching record among candidates.

    Candidates must expose ``.title`` and ``.author`` attributes
    (e.g. ``NotionBookRecord``).
    """
    incoming_key = make_book_key(incoming_title, incoming_author)
    if incoming_key == "|":
        return None

    for record in candidates:
        if make_book_key(record.title, record.author) == incoming_key:
            return record
    return None
