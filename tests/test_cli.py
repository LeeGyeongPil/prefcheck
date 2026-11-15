"""CLI behaviour: planted defects, error paths and exit codes (AC-1, AC-6)."""

from __future__ import annotations

import json
import time

import pytest

from conftest import conv_row

EXPECTED = {
    1: "identical_pair",
    2: "dropped_by_trl",
    3: "identical_after_truncation",
    5: "contradictory_label",
    7: "near_identical_pair",
    8: "extreme_length_ratio",
    9: "schema_error",
}


def test_sample_planted_defects_approx(run_cli, sample_path, tmp_path):
    json_path = tmp_path / "r.json"
    code, out, _ = run_cli(str(sample_path), "--max-length", "64", "--json", str(json_path))
    assert code == 1
    assert "errors: 5" in out and "warnings: 2" in out
    assert "approximate mode" in out  # honest banner

    data = json.loads(json_path.read_text(encoding="utf-8"))
    by_row: dict = {}
    for f in data["findings"]:
        by_row.setdefault(f["row"], []).append(f["rule"])
    # exactly the planted defects, one finding per row, nothing on healthy rows or the dataset
    assert by_row == {row: [rule] for row, rule in EXPECTED.items()}
    assert None not in by_row
    contra = next(f for f in data["findings"] if f["rule"] == "contradictory_label")
    assert contra["detail"]["other_row"] == 4
    assert data["meta"]["mode"] == "approx"


def test_error_paths_and_exit_codes(run_cli, sample_path, tmp_path):
    # missing file
    code, _, err = run_cli(str(tmp_path / "nope.jsonl"))
    assert code == 2 and "File not found" in err

    # broken JSON names the line
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(conv_row("a b", "c", "d")) + "\n" + '{"chosen": [\n', encoding="utf-8")
    code, _, err = run_cli(str(bad))
    assert code == 2 and "line 2" in err

    # unknown format
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text('{"foo": 1}\n{"bar": 2}\n', encoding="utf-8")
    code, _, err = run_cli(str(unknown))
    assert code == 2 and "Could not detect" in err

    # empty file and usage errors
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    assert run_cli(str(empty))[0] == 2
    assert run_cli()[0] == 2
    assert run_cli(str(sample_path), "--max-length", "zero")[0] == 2
    assert run_cli(str(sample_path), "--truncation", "middle")[0] == 2
    assert run_cli(str(sample_path), "--config", str(tmp_path / "missing.toml"))[0] == 2

    # offline + unknown model id fails fast with exit code 2
    pytest.importorskip("transformers")
    start = time.monotonic()
    code, _, err = run_cli(str(sample_path), "--tokenizer", "no-such-org/no-such-model", "--offline")
    assert code == 2 and "offline" in err
    assert time.monotonic() - start < 5

    # --fail-on: a warn-only file
    warn_only = tmp_path / "warn.jsonl"
    warn_only.write_text(
        json.dumps(conv_row("Which city is the capital of France?", "Paris is the capital of France.", "paris is the capital of France")) + "\n"
        + json.dumps(conv_row("What is two plus two?", "Two plus two is four.", "I believe two plus two equals four.")) + "\n",
        encoding="utf-8",
    )
    assert run_cli(str(warn_only))[0] == 0
    assert run_cli(str(warn_only), "--fail-on", "warn")[0] == 1
    assert run_cli(str(sample_path), "--max-length", "64", "--fail-on", "never")[0] == 0

    # a config file can turn a rule off
    cfg = tmp_path / "prefcheck.toml"
    cfg.write_text('[rules.near_identical_pair]\nseverity = "off"\n', encoding="utf-8")
    code, out, _ = run_cli(str(warn_only), "--config", str(cfg), "--fail-on", "warn")
    assert code == 0 and "No findings" in out
    cfg.write_text('[rules.not_a_rule]\nseverity = "off"\n', encoding="utf-8")
    assert run_cli(str(warn_only), "--config", str(cfg))[0] == 2


def test_list_rules_and_version(run_cli):
    code, out, _ = run_cli("--list-rules")
    assert code == 0 and "identical_after_truncation" in out and "length_bias" in out
    code, out, _ = run_cli("--version")
    assert code == 0 and out.startswith("prefcheck 0.1.1")


@pytest.mark.parametrize("flag", ["--clean", "--bad-rows", "--json", "--markdown"])
def test_output_path_equal_to_input_is_refused(run_cli, sample_path, tmp_path, flag):
    """Q-2: never overwrite the input file."""
    data = tmp_path / "in.jsonl"
    data.write_bytes(sample_path.read_bytes())
    before = data.read_bytes()
    code, _, err = run_cli(str(data), "--max-length", "64", flag, str(data))
    assert code == 2 and "same as the input" in err
    assert data.read_bytes() == before
    code, _, err = run_cli(str(data), "--lang", "ko", flag, str(data))
    assert code == 2 and "입력 파일과 같아" in err


def test_non_utf8_file_says_how_to_fix(run_cli, tmp_path):
    """Q-3: UTF-16 input gets a fix hint in both languages."""
    bad = tmp_path / "utf16.jsonl"
    bad.write_bytes('{"prompt":"a","chosen":"b","rejected":"c"}\n'.encode("utf-16"))
    code, _, err = run_cli(str(bad))
    assert code == 2 and "Save the file as UTF-8" in err
    code, _, err = run_cli(str(bad), "--lang", "ko")
    assert code == 2 and "UTF-8" in err and "다시 저장" in err
