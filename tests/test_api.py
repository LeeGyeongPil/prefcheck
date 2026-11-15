"""Public API surface and the README example (AC-9)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import prefcheck
from prefcheck import CheckConfig, ConfigError, check_file, check_rows

from conftest import ROOT, conv_row

PUBLIC = [
    "check_rows", "check_file", "CheckConfig", "load_tokenizer", "Report", "Finding", "RULES",
    "PrefcheckError", "InputFormatError", "TokenizerLoadError",
]


def test_public_api_surface():
    for name in PUBLIC:
        assert name in prefcheck.__all__, name
        assert hasattr(prefcheck, name), name
    assert issubclass(prefcheck.InputFormatError, prefcheck.PrefcheckError)
    assert issubclass(prefcheck.TokenizerLoadError, prefcheck.PrefcheckError)
    assert issubclass(prefcheck.ConfigError, prefcheck.PrefcheckError)
    assert prefcheck.__version__ == "0.1.1"
    assert "identical_pair" in prefcheck.RULES


def test_minimal_example_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "api_minimal.py")], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "has_errors=True" in result.stdout


def test_report_api(sample_path):
    report = check_file(sample_path, CheckConfig(max_length=64))
    assert report.exit_code() == 1 and report.exit_code("never") == 0
    assert report.count("error") == 5 and report.count("warn") == 2
    assert report.bad_row_indices() == {1, 2, 3, 5, 9}
    assert report.findings[0].line == 2  # row 1 is on file line 2
    assert "| Rule |" in report.to_markdown("en")
    with pytest.raises(ValueError):
        report.exit_code("sometimes")


def test_check_rows_validates_input():
    with pytest.raises(prefcheck.InputFormatError):
        check_rows([])
    with pytest.raises(prefcheck.InputFormatError):
        check_rows([{"foo": 1}])
    report = check_rows([conv_row("a b", "c d", "e f")])
    assert not report.has_errors


def test_config_validation(tmp_path):
    with pytest.raises(ConfigError):
        CheckConfig(max_length=0)
    with pytest.raises(ConfigError):
        CheckConfig(truncation_mode="middle")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        CheckConfig(severities={"identical_pair": "fatal"})
    with pytest.raises(ConfigError):
        CheckConfig(severities={"nope": "off"})
    toml = tmp_path / "c.toml"
    toml.write_text('[check]\nmax_length = 128\ntruncation = "keep_end"\n[thresholds]\nlength_ratio_max = 3.0\n[rules.duplicate_row]\nseverity = "error"\n', encoding="utf-8")
    cfg = CheckConfig.from_toml(toml)
    assert cfg.max_length == 128 and cfg.truncation_mode == "keep_end" and cfg.thresholds.length_ratio_max == 3.0
    assert cfg.severities == {"duplicate_row": "error"}
    toml.write_text("[thresholds]\nbogus = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        CheckConfig.from_toml(toml)
    shipped = CheckConfig.from_toml(Path(__file__).parent.parent / "examples" / "prefcheck.toml")
    assert shipped.severities["near_identical_pair"] == "off"


def test_duplicate_and_severity_override():
    row = conv_row("what is a", "a b c d", "a b x y z")
    report = check_rows([row, row], CheckConfig(severities={"duplicate_row": "error"}))
    dup = [f for f in report.findings if f.rule == "duplicate_row"]
    assert len(dup) == 1 and dup[0].row == 1 and dup[0].severity == "error"
    assert dup[0].detail["other_row"] == 0
