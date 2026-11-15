"""Exception hierarchy. The CLI turns every PrefcheckError into exit code 2."""

from __future__ import annotations

from typing import Any


class PrefcheckError(Exception):
    """Base class for all errors raised on purpose by prefcheck.

    ``key`` and ``params`` let the CLI print the message in the user's language;
    ``str(error)`` is always the English message.
    """

    def __init__(self, message: str, *, key: str | None = None, **params: Any) -> None:
        super().__init__(message)
        self.key = key
        self.params = params


class InputFormatError(PrefcheckError):
    """The input file is missing, unreadable, malformed, or not a preference dataset."""


class TokenizerLoadError(PrefcheckError):
    """A tokenizer could not be loaded (not installed, not found, or offline)."""


class ConfigError(PrefcheckError):
    """The configuration (arguments or prefcheck.toml) is invalid."""
