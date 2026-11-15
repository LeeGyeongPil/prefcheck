"""The checking pipeline: parse rows, simulate TRL preparation, run rules, collect a Report."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._version import __version__
from .config import CheckConfig
from .errors import InputFormatError
from .formats import ParsedRow, RowSchemaError, completion_text, message_text, parse_row
from .io import Record, read_records
from .report import Finding, Report
from .rules import RULES, row_rules
from .similarity import digest, normalize_text
from .stats import length_bias_metrics, mean, round3
from .tokenization import ApproxTokenizer, Tokenizer
from .trl_rules import TRL_RULES_PROFILE, TokenizedPair, tokenize_pair


def check_rows(rows: Iterable[dict[str, Any]], config: CheckConfig | None = None, tokenizer: Tokenizer | None = None) -> Report:
    """Check preference rows that are already in memory.

    ``tokenizer=None`` uses the standard-library ``ApproxTokenizer`` (practice / mock mode).
    """
    records = [Record(row=i, line=None, data=row) for i, row in enumerate(rows)]
    if not records:
        raise InputFormatError("No rows to check.", key="error.no_rows", path="<rows>")
    return check_records(records, config, tokenizer)


def check_file(path: str | Path, config: CheckConfig | None = None, tokenizer: Tokenizer | None = None) -> Report:
    """Read a JSONL / JSON file and check it. Findings carry the file's line numbers."""
    records = read_records(path)
    report = check_records(records, config, tokenizer)
    report.meta["input"] = Path(path).name
    return report


def check_records(records: list[Record], config: CheckConfig | None = None, tokenizer: Tokenizer | None = None) -> Report:
    """Run the full pipeline on records (used by the CLI, which also needs the records for output files)."""
    config = config or CheckConfig()
    tokenizer = tokenizer or ApproxTokenizer()

    if not any(isinstance(r.data, dict) and "chosen" in r.data and "rejected" in r.data for r in records):
        raise InputFormatError(
            "Could not detect a preference format: no row has both 'chosen' and 'rejected'. "
            "Expected TRL-style rows such as {\"prompt\": ..., \"chosen\": ..., \"rejected\": ...}.",
            key="error.format_undetected",
        )

    raw: list[Finding] = []

    def add(rec: Record, rule: str, detail: dict[str, Any] | None = None) -> None:
        raw.append(Finding(rec.row, rec.line, rule, RULES[rule].severity, detail or {}))  # type: ignore[arg-type]

    # 1. validate rows and extract implicit prompts
    parsed: list[tuple[Record, ParsedRow]] = []
    for rec in records:
        try:
            parsed.append((rec, parse_row(rec.data)))
        except RowSchemaError as exc:
            add(rec, "schema_error", {"reason": exc.reason, **exc.detail})

    # 2. dataset-wide format decisions (TRL looks at the first row only)
    first = parsed[0][1] if parsed else None
    add_eos = first is not None and not first.conversational
    if first is not None:
        odd = [r for r, p in parsed if p.conversational != first.conversational or (first.explicit and not p.explicit)]
        if odd:
            kinds = sorted({_describe(p) for _, p in parsed})
            raw.append(Finding(None, None, "mixed_format", RULES["mixed_format"].severity, {"first": _describe(first), "kinds": kinds, "rows": len(odd)}))  # type: ignore[arg-type]

    # 3. per-row simulation and rules
    pairs: list[tuple[Record, ParsedRow, TokenizedPair]] = []
    seen_exact: dict[tuple[str, str, str], int] = {}
    seen_norm: dict[tuple[str, str, str], int] = {}
    for rec, row in parsed:
        try:
            pair = tokenize_pair(row, tokenizer, config, add_eos=add_eos)
        except Exception as exc:  # chat templates raise arbitrary exceptions
            add(rec, "schema_error", {"reason": "template_error", "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        pairs.append((rec, row, pair))
        for rule, detail in row_rules(row, pair, config, tokenizer):
            add(rec, rule, detail)

        # duplicates and contradictions compare hashed keys only, never the original text
        ptext = message_text(row.prompt)
        exact = (digest(ptext), digest(message_text(row.chosen)), digest(message_text(row.rejected)))
        norm = (
            digest(normalize_text(ptext)),
            digest(normalize_text(completion_text(row.chosen))),
            digest(normalize_text(completion_text(row.rejected))),
        )
        if exact in seen_exact:
            add(rec, "duplicate_row", {"other_row": seen_exact[exact]})
        else:
            seen_exact[exact] = rec.row
        flipped = (norm[0], norm[2], norm[1])
        if norm[1] != norm[2] and flipped in seen_norm:
            add(rec, "contradictory_label", {"other_row": seen_norm[flipped]})
        seen_norm.setdefault(norm, rec.row)

    # 4. statistics (rows TRL would drop are excluded: they never reach the loss)
    kept = [(rec, row, pair) for rec, row, pair in pairs if not pair.dropped]
    stats = _compute_stats(len(records), parsed, pairs, kept, raw)
    th = config.thresholds
    ratio, r = stats.get("chosen_longer_ratio"), stats.get("point_biserial_r")
    if ratio is not None and r is not None and ratio >= th.length_bias_ratio and r >= th.length_bias_r:
        raw.append(
            Finding(None, None, "length_bias", RULES["length_bias"].severity,  # type: ignore[arg-type]
                    {"chosen_longer_ratio": ratio, "point_biserial_r": r, "ratio_threshold": th.length_bias_ratio, "r_threshold": th.length_bias_r})
        )

    findings = _apply_severities(raw, config)
    order = {rule_id: i for i, rule_id in enumerate(RULES)}
    findings.sort(key=lambda f: (f.row is None, f.row if f.row is not None else 0, order[f.rule]))

    meta = {
        "tool": "prefcheck",
        "tool_version": __version__,
        "trl_rules": TRL_RULES_PROFILE,
        "mode": "approx" if getattr(tokenizer, "approximate", False) else "hf",
        "tokenizer": getattr(tokenizer, "name", "custom"),
    }
    return Report(findings=findings, stats=stats, meta=meta, config=config.to_dict())


def _describe(row: ParsedRow) -> str:
    return f"{row.kind}/{'explicit' if row.explicit else 'implicit'}"


def _apply_severities(raw: list[Finding], config: CheckConfig) -> list[Finding]:
    out: list[Finding] = []
    for f in raw:
        sev = config.severities.get(f.rule, f.severity)
        if sev == "off":
            continue
        out.append(f if sev == f.severity else Finding(f.row, f.line, f.rule, sev, f.detail))  # type: ignore[arg-type]
    return out


def _compute_stats(
    total: int,
    parsed: list[tuple[Record, ParsedRow]],
    pairs: list[tuple[Record, ParsedRow, TokenizedPair]],
    kept: list[tuple[Record, ParsedRow, TokenizedPair]],
    raw: list[Finding],
) -> dict[str, Any]:
    valid = len(pairs)

    def rows_with(*rules: str) -> int:
        return len({f.row for f in raw if f.rule in rules})

    def ratio(n: int) -> float | None:
        return round3(n / valid) if valid else None

    dropped = sum(1 for _, _, p in pairs if p.dropped)
    no_signal = rows_with("identical_after_truncation", "empty_completion_after_truncation")
    chosen_lens = [len(p.chosen_ids) for _, _, p in kept]
    rejected_lens = [len(p.rejected_ids) for _, _, p in kept]
    stats: dict[str, Any] = {
        "rows": total,
        "valid_rows": valid,
        "dropped_by_trl": dropped,
        "dropped_ratio": ratio(dropped),
        "no_signal_rows": no_signal,
        "no_signal_ratio": ratio(no_signal),
        "truncated_ratio": round3(sum(1 for _, _, p in kept if p.was_truncated) / len(kept)) if kept else None,
        "mean_prompt_tokens": _r1(mean([len(p.prompt_ids) for _, _, p in kept])),
        "mean_chosen_tokens": _r1(mean(chosen_lens)),
        "mean_rejected_tokens": _r1(mean(rejected_lens)),
    }
    stats.update(length_bias_metrics(chosen_lens, rejected_lens))
    stats["contradictory_ratio"] = ratio(rows_with("contradictory_label"))
    stats["duplicate_ratio"] = ratio(rows_with("duplicate_row"))
    return stats


def _r1(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


__all__ = ["check_rows", "check_file", "check_records"]
