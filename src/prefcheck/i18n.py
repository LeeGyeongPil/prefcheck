"""Screen-language handling: language resolution and message lookup (en / ko)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

SUPPORTED_LANGS = ("en", "ko")
DEFAULT_LANG = "en"
ENV_VAR = "PREFCHECK_LANG"
_LOCALE_DIR = Path(__file__).parent / "locales"


def _match(value: str | None) -> str | None:
    """Map 'ko', 'ko_KR.UTF-8', 'en_US' ... to a supported language, or None if unknown/empty."""
    if not value:
        return None
    v = value.strip().lower()
    for lang in SUPPORTED_LANGS:
        if v == lang or v.startswith((lang + "_", lang + "-", lang + ".")):
            return lang
    return None


def resolve_lang(cli_value: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """Pick the screen language: ``--lang`` > ``PREFCHECK_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``.

    Anything that is not Korean falls back to English.
    """
    import os

    env = os.environ if env is None else env
    chosen = _match(cli_value)
    if chosen:
        return chosen
    chosen = _match(env.get(ENV_VAR))
    if chosen:
        return chosen
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = env.get(name, "")
        if value.strip():  # the first non-empty locale variable decides, as in POSIX
            return _match(value) or DEFAULT_LANG
    return DEFAULT_LANG


@lru_cache(maxsize=None)
def messages(lang: str) -> dict[str, str]:
    path = _LOCALE_DIR / f"{lang}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def t(lang: str, key: str, **params: Any) -> str:
    """Look up a message and fill its ``{placeholders}``. Raises KeyError for unknown keys."""
    template = messages(lang)[key]
    return template.format(**params)


def has(lang: str, key: str) -> bool:
    return key in messages(lang)
