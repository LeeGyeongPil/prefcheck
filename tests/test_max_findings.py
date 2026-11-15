"""--max-findings 0 must not claim 'No findings.' when findings exist."""

from __future__ import annotations

import pytest

from prefcheck import CheckConfig, check_file, check_rows

from conftest import conv_row

NONE = {"en": "No findings.", "ko": "문제가 없습니다."}
TRUNC = {"en": "findings shown", "ko": "개만 표시했습니다"}


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_terminal_zero_shown_says_truncated(run_cli, sample_path, lang):
    code, out, _ = run_cli(str(sample_path), "--max-length", "64", "--max-findings", "0", "--lang", lang)
    assert code == 1
    assert NONE[lang] not in out
    assert TRUNC[lang] in out


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_markdown_zero_shown_says_truncated(sample_path, lang):
    report = check_file(sample_path, CheckConfig(max_length=64))
    assert report.findings
    md = report.to_markdown(lang, max_findings=0)
    assert NONE[lang] not in md
    assert TRUNC[lang] in md


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_no_findings_still_says_none(lang):
    rows = [
        conv_row("Hi?", "Hello there, friend.", "No."),
        conv_row("Capital of France?", "Paris.", "The capital city of France is definitely Lyon."),
    ]
    report = check_rows(rows, CheckConfig())
    assert not report.findings
    assert NONE[lang] in report.render_terminal(lang, 0)
    assert NONE[lang] in report.to_markdown(lang, max_findings=0)
