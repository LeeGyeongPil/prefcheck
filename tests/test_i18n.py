"""Screen language (AC-8): resolution order, key parity and real switching."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from prefcheck import i18n
from prefcheck.rules import RULES
from prefcheck.report import STAT_ORDER


def test_lang_resolution_and_keys(run_cli, sample_path, tmp_path):
    # resolution order
    assert i18n.resolve_lang(None, {}) == "en"
    assert i18n.resolve_lang(None, {"PREFCHECK_LANG": "ko"}) == "ko"
    assert i18n.resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert i18n.resolve_lang(None, {"LC_ALL": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}) == "ko"
    assert i18n.resolve_lang(None, {"LC_ALL": "fr_FR.UTF-8", "LANG": "ko_KR.UTF-8"}) == "en"
    assert i18n.resolve_lang(None, {"LANG": "fr_FR.UTF-8"}) == "en"
    assert i18n.resolve_lang("en", {"PREFCHECK_LANG": "ko", "LANG": "ko_KR.UTF-8"}) == "en"
    assert i18n.resolve_lang("ko", {"PREFCHECK_LANG": "en", "LANG": "en_US.UTF-8"}) == "ko"
    assert i18n.resolve_lang(None, {"PREFCHECK_LANG": "ko", "LANG": "fr_FR.UTF-8"}) == "ko"

    # both message tables have the same keys and the same placeholders
    en, ko = i18n.messages("en"), i18n.messages("ko")
    assert set(en) == set(ko)
    holders = lambda s: set(re.findall(r"{(\w+)}", s))  # noqa: E731
    assert all(holders(en[k]) == holders(ko[k]) for k in en)
    for rule_id, info in RULES.items():
        assert info.desc_key in en and f"msg.{rule_id}" in en
    assert all(f"metric.{k}" in en for k in STAT_ORDER)

    # --help really switches language
    code, out_ko, _ = run_cli("--lang", "ko", "--help")
    assert code == 0 and "사용법" in out_ko and "usage" not in out_ko
    code, out_en, _ = run_cli("--lang", "en", "--help")
    assert code == 0 and "usage" in out_en and "사용법" not in out_en

    # summary and Markdown follow the language; --lang beats the environment
    md_ko, md_en = tmp_path / "ko.md", tmp_path / "en.md"
    code, out, _ = run_cli(str(sample_path), "--max-length", "64", "--lang", "ko", "--markdown", str(md_ko), env={"LANG": "en_US.UTF-8"})
    assert code == 1 and "오류: 5" in out and "경고: 2" in out
    assert "prefcheck 보고서" in md_ko.read_text(encoding="utf-8")
    code, out, _ = run_cli(str(sample_path), "--max-length", "64", "--lang", "en", "--markdown", str(md_en), env={"PREFCHECK_LANG": "ko"})
    assert "errors: 5" in out and "prefcheck report" in md_en.read_text(encoding="utf-8")
    code, out, _ = run_cli(str(sample_path), "--max-length", "64", env={"PREFCHECK_LANG": "ko"})
    assert "오류: 5" in out
    code, out, _ = run_cli(str(sample_path), "--max-length", "64", env={"LANG": "ko_KR.UTF-8"})
    assert "오류: 5" in out
    code, out, _ = run_cli(str(sample_path), "--max-length", "64", env={"LANG": "fr_FR.UTF-8"})
    assert "errors: 5" in out

    # error messages are localised too
    _, _, err = run_cli(str(tmp_path / "nope.jsonl"), "--lang", "ko")
    assert "찾을 수 없습니다" in err


@pytest.mark.parametrize("lang,word", [("ko", "사용법"), ("en", "usage")])
def test_help_through_console_entry_point(lang, word):
    result = subprocess.run(
        [sys.executable, "-m", "prefcheck", "--lang", lang, "--help"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0 and word in result.stdout
