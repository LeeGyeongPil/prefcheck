"""Row validation, format detection and TRL's ``extract_prompt``.

Supported (TRL preference formats): standard (strings) or conversational (message lists),
each with an explicit ``prompt`` column or an implicit prompt shared by chosen/rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ROLES = frozenset({"system", "user", "assistant", "tool", "developer"})


class RowSchemaError(Exception):
    """A row cannot be used. ``reason`` is a code that maps to a ``schema.<reason>`` message."""

    def __init__(self, reason: str, **detail: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ParsedRow:
    conversational: bool
    explicit: bool
    prompt: str | list[dict[str, Any]]
    chosen: str | list[dict[str, Any]]
    rejected: str | list[dict[str, Any]]
    tools: Any = None
    chat_template_kwargs: dict[str, Any] | None = None
    split_warning: str | None = None  # "no_divergence" | "prompt_ends_with_assistant" (implicit prompts only)

    @property
    def kind(self) -> str:
        return "conversational" if self.conversational else "standard"


def extract_prompt(chosen: Any, rejected: Any) -> tuple[Any, Any, Any, bool]:
    """Re-implementation of TRL 1.18 ``extract_prompt`` (longest common prefix of messages/characters).

    Returns ``(prompt, chosen_rest, rejected_rest, diverged)``. ``diverged`` is False when one
    side is a prefix of the other; TRL then splits off only the last element. For strings TRL
    moves one trailing space from the prompt into the completion. Unlike TRL, the space step
    is skipped at index 0 (TRL would look at ``chosen[-1]`` there).
    """
    n = min(len(chosen), len(rejected))
    idx = 0
    diverged = False
    for idx in range(n):
        if chosen[idx] != rejected[idx]:
            diverged = True
            if idx > 0 and chosen[idx - 1] == " ":
                idx -= 1
            break
    return chosen[:idx], chosen[idx:], rejected[idx:], diverged


def _check_messages(field: str, value: Any, *, allow_empty_list: bool = False) -> None:
    if not isinstance(value, list):
        raise RowSchemaError("bad_type", field=field, expected="list of messages")
    if not value and not allow_empty_list:
        raise RowSchemaError("empty_messages", field=field)
    for i, msg in enumerate(value):
        where = f"{field}[{i}]"
        if not isinstance(msg, dict):
            raise RowSchemaError("bad_type", field=where, expected="message object")
        if "role" not in msg:
            raise RowSchemaError("missing_field", field=f"{where}.role")
        if msg["role"] not in ROLES:
            raise RowSchemaError("bad_role", field=where, role=str(msg["role"]))
        if "content" not in msg:
            raise RowSchemaError("missing_field", field=f"{where}.content")
        content = msg["content"]
        if not isinstance(content, str):
            raise RowSchemaError("bad_type", field=f"{where}.content", expected="string")
        if not content.strip() and not msg.get("tool_calls"):
            raise RowSchemaError("empty_content", field=where)


def _check_text(field: str, value: Any) -> None:
    if not isinstance(value, str):
        raise RowSchemaError("bad_type", field=field, expected="string")
    if not value.strip():
        raise RowSchemaError("empty_content", field=field)


def parse_row(data: Any) -> ParsedRow:
    """Validate one record and return it in explicit-prompt form. Raises ``RowSchemaError``."""
    if not isinstance(data, dict):
        raise RowSchemaError("not_an_object")
    for key in ("chosen", "rejected"):
        if key not in data:
            raise RowSchemaError("missing_field", field=key)
    chosen, rejected = data["chosen"], data["rejected"]
    explicit = "prompt" in data
    prompt = data.get("prompt")

    conversational = isinstance(chosen, list)
    for key, value in (("chosen", chosen), ("rejected", rejected)) + ((("prompt", prompt),) if explicit else ()):
        if isinstance(value, list) != conversational:
            raise RowSchemaError("mixed_types", field=key)

    tools = data.get("tools")
    if tools is not None and not isinstance(tools, (list, str)):
        raise RowSchemaError("bad_type", field="tools", expected="list or JSON string")
    kwargs = data.get("chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise RowSchemaError("bad_type", field="chat_template_kwargs", expected="object")
    if not conversational and (tools is not None or kwargs is not None):
        tools = kwargs = None  # TRL only forwards them for conversational rows

    split_warning = None
    if conversational:
        _check_messages("chosen", chosen)
        _check_messages("rejected", rejected)
        if explicit:
            _check_messages("prompt", prompt)
            if prompt[-1]["role"] == "assistant":
                raise RowSchemaError("prompt_ends_with_assistant")
        else:
            prompt, chosen, rejected, diverged = extract_prompt(chosen, rejected)
            if not diverged:
                split_warning = "no_divergence"
            elif prompt and prompt[-1]["role"] == "assistant":
                split_warning = "prompt_ends_with_assistant"
        if split_warning != "no_divergence":
            for key, value in (("chosen", chosen), ("rejected", rejected)):
                if value[0]["role"] != "assistant":
                    raise RowSchemaError("completion_not_assistant", field=key, role=value[0]["role"])
    else:
        _check_text("chosen", chosen)
        _check_text("rejected", rejected)
        if explicit:
            _check_text("prompt", prompt)
        else:
            prompt, chosen, rejected, diverged = extract_prompt(chosen, rejected)
            if not diverged:
                split_warning = "no_divergence"

    return ParsedRow(conversational, explicit, prompt, chosen, rejected, tools, kwargs, split_warning)


def message_text(value: Any) -> str:
    """Plain text of a prompt/completion (message contents joined by newlines)."""
    if isinstance(value, str):
        return value
    return "\n".join(f"{m['role']}: {m['content']}" for m in value)


def completion_text(value: Any) -> str:
    """Text of a completion without role labels (used for similarity)."""
    if isinstance(value, str):
        return value
    return "\n".join(m["content"] for m in value)
