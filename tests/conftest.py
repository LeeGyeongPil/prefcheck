"""Shared test setup: no network, fixture paths and an in-process CLI runner."""

from __future__ import annotations

import io
import os
from collections.abc import Callable
from pathlib import Path

# Must be set before transformers is imported anywhere: block every Hub call.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest  # noqa: E402

from prefcheck.cli import main  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = ROOT / "examples"


@pytest.fixture
def tiny_tokenizer_path() -> str:
    return str(FIXTURES / "tiny-tokenizer")


@pytest.fixture
def wrap_tokenizer_path() -> str:
    return str(FIXTURES / "tiny-tokenizer-wrap")


@pytest.fixture
def sample_path() -> Path:
    return EXAMPLES / "conversational.jsonl"


@pytest.fixture
def implicit_path() -> Path:
    return EXAMPLES / "standard_implicit.jsonl"


CliResult = tuple[int, str, str]


@pytest.fixture
def run_cli() -> Callable[..., CliResult]:
    """Run the CLI in-process and return (exit_code, stdout, stderr). English unless env says otherwise."""

    def run(*args: str, env: dict[str, str] | None = None) -> CliResult:
        out, err = io.StringIO(), io.StringIO()
        code = main(list(args), env=env if env is not None else {"LANG": "en_US.UTF-8"}, stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()

    return run


def conv_row(prompt: str, chosen: str, rejected: str) -> dict:
    """A conversational explicit-prompt row."""
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }
