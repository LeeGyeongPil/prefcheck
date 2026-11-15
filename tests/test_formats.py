"""Prompt extraction and boundaries (AC-4)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("transformers")

from prefcheck import CheckConfig, check_file, check_rows, load_tokenizer  # noqa: E402
from prefcheck.formats import RowSchemaError, extract_prompt, parse_row  # noqa: E402

from conftest import conv_row  # noqa: E402


def test_extract_prompt_and_boundaries(implicit_path, wrap_tokenizer_path, tmp_path):
    rows = [json.loads(line) for line in implicit_path.read_text(encoding="utf-8").splitlines()]

    # TRL extract_prompt: common prefix, one space moved from the prompt into the completion
    assert extract_prompt(rows[0]["chosen"], rows[0]["rejected"]) == ("The sky is", " blue.", " green.", True)
    assert extract_prompt(rows[2]["chosen"], rows[2]["rejected"]) == ("2 plus 2 is", " 4.", " 5.", True)
    # one side is a prefix of the other: no divergence, TRL keeps only the last element as completion
    prompt, chosen, rejected, diverged = extract_prompt(rows[1]["chosen"], rows[1]["rejected"])
    assert (prompt, chosen, rejected, diverged) == ("Water boils at 100 degree", "s", "s Celsius.", False)
    # conversational: common prefix of messages
    u = {"role": "user", "content": "hi"}
    assert extract_prompt([u, {"role": "assistant", "content": "a"}], [u, {"role": "assistant", "content": "b"}])[0] == [u]

    report = check_file(implicit_path, CheckConfig(max_length=64))
    by_row: dict = {}
    for f in report.findings:
        by_row.setdefault(f.row, []).append(f.rule)
    assert "implicit_prompt_split" in by_row[1]
    assert not {"implicit_prompt_split"} & set(by_row.get(0, []) + by_row.get(2, []) + by_row.get(3, []))
    assert not [f for f in report.findings if f.rule == "mixed_format"]

    # a template that renders the prompt differently when alone: every row loses the prompt tail
    wrap = load_tokenizer(wrap_tokenizer_path, offline=True)
    wrap_rows = [conv_row("what color is the sky", "the sky is blue", "the sky is green"), conv_row("what is a", "a b c", "a b d")]
    wrap_report = check_rows(wrap_rows, CheckConfig(max_length=None), tokenizer=wrap)
    hits = [f for f in wrap_report.findings if f.rule == "prompt_not_prefix"]
    assert sorted(f.row for f in hits) == [0, 1]
    assert all(f.detail["dropped_tokens"] > 0 for f in hits)

    # formats mixed in one file
    mixed = [conv_row("what color is the sky", "the sky is blue", "the sky is green"), {"prompt": "the sky is", "chosen": " blue", "rejected": " green"}]
    mixed_report = check_rows(mixed)
    assert [f.rule for f in mixed_report.findings if f.row is None] == ["mixed_format"]
    assert mixed_report.has_errors


def test_schema_validation_reasons():
    def reason(data):
        with pytest.raises(RowSchemaError) as exc:
            parse_row(data)
        return exc.value.reason

    assert reason([1, 2]) == "not_an_object"
    assert reason({"chosen": "a"}) == "missing_field"
    assert reason({"prompt": "p", "chosen": "a", "rejected": ["x"]}) == "mixed_types"
    assert reason({"prompt": "p", "chosen": " ", "rejected": "b"}) == "empty_content"
    bad_role = conv_row("q", "a", "b")
    bad_role["chosen"][0]["role"] = "robot"
    assert reason(bad_role) == "bad_role"
    user_first = conv_row("q", "a", "b")
    user_first["chosen"][0]["role"] = "user"
    assert reason(user_first) == "completion_not_assistant"
    ends_assistant = conv_row("q", "a", "b")
    ends_assistant["prompt"][0]["role"] = "assistant"
    assert reason(ends_assistant) == "prompt_ends_with_assistant"
    assert reason({"prompt": [], "chosen": [{"role": "assistant", "content": "a"}], "rejected": [{"role": "assistant", "content": "b"}]}) == "empty_messages"


def test_implicit_conversational_prompt():
    u = {"role": "user", "content": "What color is the sky?"}
    row = parse_row({"chosen": [u, {"role": "assistant", "content": "Blue."}], "rejected": [u, {"role": "assistant", "content": "Green."}]})
    assert not row.explicit and row.prompt == [u] and row.split_warning is None
