"""Settings change results (AC-3) and the pure TRL helpers."""

from __future__ import annotations

from prefcheck import CheckConfig, check_file
from prefcheck.trl_rules import common_prefix_length, truncate

TRUNCATION_RULES = {"dropped_by_trl", "empty_completion_after_truncation", "identical_after_truncation"}


def rules_by_row(report):
    out: dict = {}
    for f in report.findings:
        if f.row is not None:
            out.setdefault(f.row, set()).add(f.rule)
    return out


def test_truncation_mode_changes_findings(sample_path):
    start = rules_by_row(check_file(sample_path, CheckConfig(max_length=64, truncation_mode="keep_start")))
    assert "dropped_by_trl" in start[2] and "identical_after_truncation" in start[3]

    end = rules_by_row(check_file(sample_path, CheckConfig(max_length=64, truncation_mode="keep_end")))
    assert "dropped_by_trl" not in end.get(2, set())
    assert "identical_after_truncation" not in end.get(3, set())

    unlimited = check_file(sample_path, CheckConfig(max_length=None))
    assert not [f for f in unlimited.findings if f.rule in TRUNCATION_RULES]
    # the non-truncation defects are still found
    assert {"identical_pair", "contradictory_label", "schema_error"} <= {f.rule for f in unlimited.findings}


def test_common_prefix_and_truncate():
    assert common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_length([], [1]) == 0
    ids, mask = [1, 2, 3, 4, 5], [0, 0, 1, 1, 1]
    assert truncate(ids, mask, 3, "keep_start") == ([1, 2, 3], [0, 0, 1])
    assert truncate(ids, mask, 3, "keep_end") == ([3, 4, 5], [1, 1, 1])
    assert truncate(ids, mask, None, "keep_end") == (ids, mask)


def test_whitespace_only_difference_is_not_a_truncation_finding(tmp_path):
    """Q-1: no truncation happened, so no truncation rule may fire (regression)."""
    path = tmp_path / "ws.jsonl"
    path.write_text(
        '{"prompt":"Where is Paris?","chosen":" Paris is in France.","rejected":" Paris  is in France."}\n'
        '{"prompt":"Q2?","chosen":" yes it is","rejected":" no it is not"}\n',
        encoding="utf-8",
    )
    report = check_file(path, CheckConfig(max_length=None))
    assert not [f for f in report.findings if f.rule in TRUNCATION_RULES]
    assert "near_identical_pair" in rules_by_row(report)[0]
    report = check_file(path, CheckConfig(max_length=64))
    assert not [f for f in report.findings if f.rule in TRUNCATION_RULES]
