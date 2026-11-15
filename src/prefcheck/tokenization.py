"""Tokenizer adapters.

``ApproxTokenizer`` needs nothing but the standard library (practice / mock mode).
``HFTokenizer`` wraps a ``transformers`` tokenizer and is the only place that calls
``apply_chat_template``, so transformers API changes stay in one spot.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .errors import TokenizerLoadError

Message = dict[str, Any]


@runtime_checkable
class Tokenizer(Protocol):
    """What the TRL-rule engine needs from a tokenizer."""

    name: str
    eos_token: str

    def encode(self, text: str) -> list[Any]: ...

    def apply_chat_template(
        self, messages: Sequence[Message], *, add_generation_prompt: bool = False, **kwargs: Any
    ) -> list[Any]: ...

    def decode(self, ids: Sequence[Any]) -> str: ...


_APPROX_TOKEN = re.compile(r"<\|[^|>\s]+\|>|</s>|\w+|[^\w\s]", re.UNICODE)


class ApproxTokenizer:
    """Regex word/punctuation tokenizer with a built-in ChatML-style template.

    Token counts only approximate a real model's, but every other part of the TRL
    simulation (drop filter, truncation, identical-after-truncation) runs for real.
    Tokens are plain strings. ``tools`` and ``chat_template_kwargs`` are ignored.
    """

    name = "approx"
    eos_token = "</s>"
    approximate = True

    def encode(self, text: str) -> list[str]:
        return _APPROX_TOKEN.findall(text)

    def render_chat(self, messages: Sequence[Message], add_generation_prompt: bool) -> str:
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def apply_chat_template(
        self, messages: Sequence[Message], *, add_generation_prompt: bool = False, **kwargs: Any
    ) -> list[str]:
        return self.encode(self.render_chat(messages, add_generation_prompt))

    def decode(self, ids: Sequence[Any]) -> str:
        return " ".join(str(i) for i in ids)


class HFTokenizer:
    """Adapter around a ``transformers`` tokenizer (tokenizer only, no torch needed)."""

    approximate = False

    def __init__(self, tokenizer: Any, name: str) -> None:
        self._tok = tokenizer
        self.name = name
        eos = getattr(tokenizer, "eos_token", None)
        self.eos_token = eos if isinstance(eos, str) else ""

    def encode(self, text: str) -> list[int]:
        return _flat_ids(self._tok(text=text)["input_ids"])

    def apply_chat_template(
        self, messages: Sequence[Message], *, add_generation_prompt: bool = False, **kwargs: Any
    ) -> list[int]:
        out = self._tok.apply_chat_template(
            list(messages), tokenize=True, return_dict=True, add_generation_prompt=add_generation_prompt, **kwargs
        )
        return _flat_ids(out["input_ids"])

    def decode(self, ids: Sequence[Any]) -> str:
        return str(self._tok.decode(list(ids)))


def _flat_ids(ids: Any) -> list[int]:
    ids = list(ids)
    if ids and isinstance(ids[0], (list, tuple)):  # some versions keep a batch dimension
        ids = list(ids[0])
    return [int(i) for i in ids]


def _env_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def load_tokenizer(name_or_path: str, *, offline: bool | None = None, revision: str | None = None) -> HFTokenizer:
    """Load a Hugging Face tokenizer from a Hub id or a local folder.

    ``offline=True`` (or ``HF_HUB_OFFLINE=1`` when ``offline`` is None) reads only local
    files and fails immediately instead of retrying the network.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenizerLoadError(
            'Real tokenizer mode needs the optional dependencies. Install them with: pip install "prefcheck[hf]"',
            key="error.hf_missing",
        ) from exc

    local_only = _env_offline() if offline is None else offline
    kwargs: dict[str, Any] = {"local_files_only": local_only}
    if revision:
        kwargs["revision"] = revision
    try:
        tok = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    except Exception as exc:  # transformers raises OSError, ValueError, ... depending on the cause
        hint = (
            "offline mode is on: pass a local folder or populate the cache first"
            if local_only
            else "check the model id, your network, or pass a local folder"
        )
        reason = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}".strip()
        raise TokenizerLoadError(
            f"Could not load tokenizer {name_or_path!r} ({reason}); {hint}.",
            key="error.tokenizer_load_offline" if local_only else "error.tokenizer_load",
            name=name_or_path, reason=reason,
        ) from exc
    # a local folder is reported by its folder name so reports never carry absolute paths
    return HFTokenizer(tok, Path(name_or_path).name if os.path.isdir(name_or_path) else name_or_path)
