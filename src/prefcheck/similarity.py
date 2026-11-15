"""Text normalisation and similarity helpers (standard library only)."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lower-case, drop punctuation and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return _NON_WORD.sub(" ", text).strip()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def similarity(a: str, b: str) -> float:
    """1.0 when the texts are equal after normalisation, else the Jaccard score of their character 3-grams."""
    na, nb = normalize_text(a), normalize_text(b)
    if na == nb:
        return 1.0
    ga, gb = char_ngrams(na), char_ngrams(nb)
    if not ga or not gb:
        return 0.0
    return jaccard(ga, gb)


def is_near_identical(a: str, b: str, threshold: float) -> bool:
    """True when two texts are equal after normalisation or share >= threshold of their 3-grams."""
    return similarity(a, b) >= threshold


def digest(text: str) -> str:
    """Short stable key so the dataset-level maps do not keep the original text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
