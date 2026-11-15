"""Findings, the Report object and its renderers (JSON, Markdown, terminal)."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from . import i18n
from .rules import RULES

FAIL_ON_CHOICES = ("error", "warn", "never")

# Order of the metrics in tables.
STAT_ORDER = (
    "rows",
    "valid_rows",
    "dropped_by_trl",
    "dropped_ratio",
    "no_signal_rows",
    "no_signal_ratio",
    "truncated_ratio",
    "mean_prompt_tokens",
    "mean_chosen_tokens",
    "mean_rejected_tokens",
    "chosen_longer_ratio",
    "point_biserial_r",
    "length_ratio_p10",
    "length_ratio_p50",
    "length_ratio_p90",
    "contradictory_ratio",
    "duplicate_ratio",
)


@dataclass(frozen=True)
class Finding:
    """One problem. ``row`` is the 0-based record index, ``line`` the 1-based file line; ``row=None`` is dataset-wide."""

    row: int | None
    line: int | None
    rule: str
    severity: Literal["error", "warn"]
    detail: dict[str, Any] = field(default_factory=dict)


def finding_message(finding: Finding, lang: str = "en") -> str:
    """Human-readable text of a finding in the given language."""
    params = dict(finding.detail)
    if finding.rule == "schema_error":
        reason = params.pop("reason")
        params = {"reason_text": i18n.t(lang, f"schema.{reason}", **params)}
    elif finding.rule == "implicit_prompt_split":
        params = {"reason_text": i18n.t(lang, f"split.{params['reason']}")}
    elif finding.rule == "mixed_format":
        params["kinds_text"] = ", ".join(params.get("kinds", []))
    return i18n.t(lang, f"msg.{finding.rule}", **params)


@dataclass
class Report:
    findings: list[Finding]
    stats: dict[str, Any]
    meta: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)

    # --- counts ---------------------------------------------------------------------------
    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    @property
    def has_errors(self) -> bool:
        return self.count("error") > 0

    def exit_code(self, fail_on: str = "error") -> int:
        """0 or 1 for the CI gate (usage errors are exit code 2 and never come from a Report)."""
        if fail_on not in FAIL_ON_CHOICES:
            raise ValueError(f"fail_on must be one of {FAIL_ON_CHOICES}, got {fail_on!r}")
        if fail_on == "never":
            return 0
        if fail_on == "warn":
            return 1 if self.findings else 0
        return 1 if self.has_errors else 0

    def bad_row_indices(self) -> set[int]:
        """Rows with at least one error-severity finding."""
        return {f.row for f in self.findings if f.severity == "error" and f.row is not None}

    def error_rules_by_row(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for f in self.findings:
            if f.severity == "error" and f.row is not None:
                out.setdefault(f.row, []).append(f.rule)
        return out

    def count_by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rule_id in RULES:
            n = sum(1 for f in self.findings if f.rule == rule_id)
            if n:
                counts[rule_id] = n
        return counts

    # --- machine output -------------------------------------------------------------------
    def to_dict(self, lang: str = "en") -> dict[str, Any]:
        return {
            "meta": self.meta,
            "config": self.config,
            "summary": {"errors": self.count("error"), "warnings": self.count("warn")},
            "stats": self.stats,
            "findings": [
                {
                    "row": f.row,
                    "line": f.line,
                    "rule": f.rule,
                    "severity": f.severity,
                    "message": finding_message(f, lang),
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }

    def to_json(self, lang: str = "en") -> str:
        return json.dumps(self.to_dict(lang), ensure_ascii=False, indent=2, sort_keys=False) + "\n"

    # --- human output ---------------------------------------------------------------------
    def _settings_line(self, lang: str) -> str:
        cfg = self.config
        max_length = cfg.get("max_length")
        return i18n.t(
            lang,
            "summary.settings",
            max_length=max_length if max_length is not None else "none",
            truncation=cfg.get("truncation_mode"),
            tokenizer=self.meta.get("tokenizer"),
            trl_rules=self.meta.get("trl_rules"),
        )

    def to_markdown(self, lang: str = "en", max_findings: int | None = None) -> str:
        t = lambda key, **p: i18n.t(lang, key, **p)  # noqa: E731
        lines = [f"# {t('md.title')}", ""]
        if self.meta.get("mode") == "approx":
            lines += [f"> {t('banner.approx')}", ""]
        lines += [
            f"- {t('summary.input', input=self.meta.get('input', '-'))}",
            f"- {self._settings_line(lang)}",
            f"- {t('summary.rows', rows=self.stats.get('rows', 0), valid=self.stats.get('valid_rows', 0))}",
            "",
            f"## {t('md.summary')}",
            "",
            f"| {t('md.metric')} | {t('md.value')} |",
            "|---|---|",
            f"| {t('summary.errors_label')} | {self.count('error')} |",
            f"| {t('summary.warnings_label')} | {self.count('warn')} |",
        ]
        for key in STAT_ORDER:
            if key in self.stats:
                lines.append(f"| {t('metric.' + key)} | {_fmt(self.stats[key])} |")
        lines += ["", f"## {t('md.by_rule')}", ""]
        counts = self.count_by_rule()
        if counts:
            lines += [f"| {t('md.rule')} | {t('md.severity')} | {t('md.count')} | {t('md.description')} |", "|---|---|---|---|"]
            for rule_id, n in counts.items():
                sev = next(f.severity for f in self.findings if f.rule == rule_id)
                lines.append(f"| `{rule_id}` | {t('severity.' + sev)} | {n} | {_cell(t(RULES[rule_id].desc_key))} |")
        else:
            lines.append(t("findings.none"))
        lines += ["", f"## {t('md.findings')}", ""]
        shown = self.findings if max_findings is None else self.findings[:max_findings]
        if shown:
            lines += [f"| {t('col.row')} | {t('col.line')} | {t('md.severity')} | {t('md.rule')} | {t('col.message')} |", "|---|---|---|---|---|"]
            for f in shown:
                lines.append(
                    f"| {_dash(f.row)} | {_dash(f.line)} | {t('severity.' + f.severity)} | `{f.rule}` | {_cell(finding_message(f, lang))} |"
                )
            if len(shown) < len(self.findings):
                lines += ["", t("findings.truncated", shown=len(shown), total=len(self.findings))]
        else:
            lines.append(t("findings.none"))
        return "\n".join(lines) + "\n"

    def render_terminal(self, lang: str = "en", max_findings: int | None = 50) -> str:
        t = lambda key, **p: i18n.t(lang, key, **p)  # noqa: E731
        out = [t("summary.title", version=self.meta.get("tool_version", "")), ""]
        if self.meta.get("mode") == "approx":
            out += [t("banner.approx"), ""]
        out += [
            t("summary.input", input=self.meta.get("input", "-")),
            self._settings_line(lang),
            t("summary.rows", rows=self.stats.get("rows", 0), valid=self.stats.get("valid_rows", 0)),
            "",
            f"{t('summary.errors', count=self.count('error'))}",
            f"{t('summary.warnings', count=self.count('warn'))}",
            "",
            t("metrics.title"),
        ]
        out += _table([(t("metric." + k), _fmt(self.stats[k])) for k in STAT_ORDER if k in self.stats], indent=2)
        out.append("")
        counts = self.count_by_rule()
        if counts:
            out.append(t("rules.title"))
            out += _table([(rule_id, str(n)) for rule_id, n in counts.items()], indent=2)
            out.append("")
        shown = self.findings if max_findings is None else self.findings[:max_findings]
        out.append(t("findings.title", shown=len(shown), total=len(self.findings)))
        if shown:
            rows = [(t("col.row"), t("col.line"), t("md.severity"), t("md.rule"), t("col.message"))]
            rows += [
                (_dash(f.row), _dash(f.line), t("severity." + f.severity), f.rule, finding_message(f, lang)) for f in shown
            ]
            out += _table(rows, indent=2, gap=2)
            if len(shown) < len(self.findings):
                out.append("  " + t("findings.truncated", shown=len(shown), total=len(self.findings)))
        else:
            out.append("  " + t("findings.none"))
        return "\n".join(out) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _dash(value: Any) -> str:
    return "-" if value is None else str(value)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _width(text: str) -> int:
    """Display width: East Asian wide characters take two columns."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * (width - _width(text))


def _table(rows: list[tuple[str, ...]], *, indent: int = 0, gap: int = 2) -> list[str]:
    """Plain aligned columns; the last column is not padded."""
    if not rows:
        return []
    ncols = len(rows[0])
    widths = [max(_width(r[i]) for r in rows) for i in range(ncols - 1)]
    lines = []
    for r in rows:
        cells = [_pad(r[i], widths[i]) for i in range(ncols - 1)] + [r[-1]]
        lines.append(" " * indent + (" " * gap).join(cells))
    return lines
