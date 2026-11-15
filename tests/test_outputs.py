"""Output files: JSON, Markdown, bad-rows, clean and determinism (AC-7)."""

from __future__ import annotations

import json


def test_report_files_deterministic(run_cli, sample_path, tmp_path):
    def run(tag: str):
        d = tmp_path / tag
        code, out, _ = run_cli(
            str(sample_path), "--max-length", "64",
            "--json", str(d / "r.json"), "--markdown", str(d / "r.md"),
            "--bad-rows", str(d / "bad.jsonl"), "--clean", str(d / "clean.jsonl"),
        )
        assert code == 1
        return d, out

    d1, out1 = run("a")
    d2, out2 = run("b")
    for name in ("r.json", "r.md", "bad.jsonl", "clean.jsonl"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes(), name
    assert "wrote" in out1

    data = json.loads((d1 / "r.json").read_text(encoding="utf-8"))
    assert data["meta"]["tool_version"] == "0.1.0"
    assert data["meta"]["trl_rules"] == "trl-1.18"
    assert data["meta"]["mode"] == "approx"
    assert data["config"]["max_length"] == 64 and data["config"]["truncation_mode"] == "keep_start"
    assert data["stats"]["rows"] == 10 and data["stats"]["dropped_by_trl"] == 1
    assert data["summary"] == {"errors": 5, "warnings": 2}
    assert len(data["findings"]) == 7
    assert "/" not in data["meta"]["input"]  # no absolute paths in the report

    md = (d1 / "r.md").read_text(encoding="utf-8")
    assert "| Rule | Severity | Count | Description |" in md
    assert "`dropped_by_trl`" in md and "approximate mode" in md

    original = sample_path.read_text(encoding="utf-8").splitlines()
    bad_rows = {f["row"] for f in data["findings"] if f["severity"] == "error" and f["row"] is not None}
    assert bad_rows == {1, 2, 3, 5, 9}

    bad = [json.loads(line) for line in (d1 / "bad.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(bad) == len(bad_rows)
    assert all("_prefcheck" in row for row in bad)
    assert bad[0]["_prefcheck"] == ["identical_pair"]
    assert {k: v for k, v in bad[1].items() if k != "_prefcheck"} == json.loads(original[2])

    clean = (d1 / "clean.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(clean) == len(original) - len(bad_rows)
    assert clean == [line for i, line in enumerate(original) if i not in bad_rows]  # original text preserved

    # re-checking the clean copy gives no errors (the CI gate flow)
    code, _, _ = run_cli(str(d1 / "clean.jsonl"), "--max-length", "64")
    assert code == 0


def test_json_array_input_and_line_numbers(run_cli, tmp_path):
    rows = [{"chosen": "a b c", "rejected": "a b c"}, {"prompt": "x y", "chosen": " z", "rejected": " w"}]
    p = tmp_path / "rows.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    j = tmp_path / "out.json"
    code, _, _ = run_cli(str(p), "--json", str(j), "--clean", str(tmp_path / "c.jsonl"))
    data = json.loads(j.read_text(encoding="utf-8"))
    assert code == 1
    assert data["findings"][0]["row"] == 0 and data["findings"][0]["line"] is None
    # the first row is implicit, the second explicit: allowed (TRL extracts per row), no mixed_format
    assert not [f for f in data["findings"] if f["rule"] == "mixed_format"]
    assert len((tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()) == 1
