"""Rule table and the per-row rules.

Dataset-wide rules (contradictions, duplicates, mixed formats, length bias) need the whole
file and run in ``checker.py``. Rule ids are stable: they appear in JSON, config files and
``--bad-rows`` output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import CheckConfig
from .formats import ParsedRow, completion_text
from .similarity import is_near_identical, similarity
from .tokenization import Tokenizer
from .trl_rules import TokenizedPair


@dataclass(frozen=True)
class RuleInfo:
    id: str
    severity: str  # default: "error" | "warn"
    scope: str  # "row" | "dataset"
    desc_key: str  # message key in the locale files


def _rule(rule_id: str, severity: str, scope: str = "row") -> tuple[str, RuleInfo]:
    return rule_id, RuleInfo(rule_id, severity, scope, f"rule.{rule_id}.desc")


# Order is the order findings are listed for the same row.
RULES: Mapping[str, RuleInfo] = dict(
    [
        _rule("schema_error", "error"),
        _rule("identical_pair", "error"),
        _rule("dropped_by_trl", "error"),
        _rule("empty_completion_after_truncation", "error"),
        _rule("identical_after_truncation", "error"),
        _rule("contradictory_label", "error"),
        _rule("prompt_not_prefix", "warn"),
        _rule("near_identical_pair", "warn"),
        _rule("extreme_length_ratio", "warn"),
        _rule("duplicate_row", "warn"),
        _rule("implicit_prompt_split", "warn"),
        _rule("mixed_format", "error", "dataset"),
        _rule("length_bias", "warn", "dataset"),
    ]
)

RuleHit = tuple[str, dict[str, Any]]


def row_rules(row: ParsedRow, pair: TokenizedPair, config: CheckConfig, tokenizer: Tokenizer) -> list[RuleHit]:
    """Apply the per-row rules to one tokenised pair. One root cause yields at most one hit."""
    th = config.thresholds
    hits: list[RuleHit] = []
    identical = row.chosen == row.rejected

    if row.split_warning and not (identical and row.split_warning == "no_divergence"):
        hits.append(("implicit_prompt_split", {"reason": row.split_warning}))
    if identical:
        hits.append(("identical_pair", {}))
    if pair.dropped_tail:
        text = tokenizer.decode(pair.dropped_tail)
        hits.append(("prompt_not_prefix", {"dropped_tokens": len(pair.dropped_tail), "dropped_text": text[:80]}))

    if pair.dropped:
        hits.append(("dropped_by_trl", {"prompt_tokens": len(pair.prompt_ids), "max_length": config.max_length}))
        return hits  # TRL never trains on this row, so the other signal rules do not apply

    if pair.chosen_mask_sum == 0 or pair.rejected_mask_sum == 0:
        hits.append(
            ("empty_completion_after_truncation", {"chosen_tokens": pair.chosen_mask_sum, "rejected_tokens": pair.rejected_mask_sum})
        )
    elif not identical and pair.identical_after_truncation and pair.was_truncated:
        hits.append(
            (
                "identical_after_truncation",
                {"max_length": config.max_length, "truncation": config.truncation_mode, "kept_tokens": len(pair.chosen_trunc)},
            )
        )
    elif not identical:
        ctext, rtext = completion_text(row.chosen), completion_text(row.rejected)
        # Same tokens without any truncation (e.g. whitespace-only differences) is a near-duplicate, not a truncation loss.
        if pair.identical_after_truncation or is_near_identical(ctext, rtext, th.near_identical_jaccard):
            hits.append(("near_identical_pair", {"similarity": round(similarity(ctext, rtext), 3)}))

    lc, lr = len(pair.chosen_ids), len(pair.rejected_ids)
    if lc > 0 and lr > 0:
        ratio = lc / lr
        if ratio >= th.length_ratio_max or ratio <= 1 / th.length_ratio_max:
            hits.append(("extreme_length_ratio", {"ratio": round(ratio, 3), "chosen_tokens": lc, "rejected_tokens": lr}))
    return hits
