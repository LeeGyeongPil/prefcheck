"""Reproduction of how TRL 1.18 ``DPOTrainer`` prepares a preference pair (profile ``trl-1.18``).

Source: ``trl/trainer/dpo_trainer.py`` (``tokenize_fn``, the max_length filter and
``DataCollatorForPreference``) and ``trl/data_utils.py`` (``common_prefix_length``).
Steps, in order:

1. Standard (non-conversational) datasets get the EOS token appended to chosen and rejected
   unless it is already there. TRL decides this from the first row only.
2. Conversational rows are tokenised with the chat template (prompt alone with
   ``add_generation_prompt=True``; prompt+chosen and prompt+rejected without). Standard rows
   are tokenised as plain strings.
3. The completion starts where the prompt tokens stop being a common prefix of both
   prompt+completion sequences (``prompt_len``).
4. With ``keep_start`` and a ``max_length``, examples with ``len(prompt_ids) >= max_length``
   are dropped before training.
5. The collator cuts ``prompt+completion`` to ``ids[:max_length]`` (``keep_start``) or
   ``ids[-max_length:]`` (``keep_end``); the completion mask is cut the same way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import CheckConfig
from .formats import ParsedRow
from .tokenization import Tokenizer

TRL_RULES_PROFILE = "trl-1.18"


def common_prefix_length(a: Sequence[Any], b: Sequence[Any]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def truncate(ids: Sequence[Any], mask: Sequence[int], max_length: int | None, mode: str) -> tuple[list[Any], list[int]]:
    """Cut ids and mask exactly like TRL's collator."""
    if max_length is None:
        return list(ids), list(mask)
    sl = slice(None, max_length) if mode == "keep_start" else slice(-max_length, None)
    return list(ids[sl]), list(mask[sl])


@dataclass(frozen=True)
class TokenizedPair:
    prompt_ids: list[Any]
    chosen_ids: list[Any]
    rejected_ids: list[Any]
    prompt_tokens_full: int  # length of the prompt tokenised alone
    dropped_tail: list[Any]  # prompt tokens that fall out of the training sequence (usually empty)
    dropped: bool  # removed by TRL's keep_start filter
    chosen_trunc: list[Any]
    rejected_trunc: list[Any]
    chosen_mask_sum: int
    rejected_mask_sum: int
    was_truncated: bool

    @property
    def identical_after_truncation(self) -> bool:
        return self.chosen_trunc == self.rejected_trunc


def tokenize_pair(row: ParsedRow, tokenizer: Tokenizer, config: CheckConfig, *, add_eos: bool) -> TokenizedPair:
    """Run steps 1-5 for one validated row. Template errors propagate to the caller."""
    chosen, rejected = row.chosen, row.rejected
    if add_eos and not row.conversational:
        eos = tokenizer.eos_token
        if eos:
            if not str(chosen).endswith(eos):
                chosen = f"{chosen}{eos}"
            if not str(rejected).endswith(eos):
                rejected = f"{rejected}{eos}"

    if row.conversational:
        kw: dict[str, Any] = dict(row.chat_template_kwargs or {})
        if row.tools is not None:
            kw["tools"] = row.tools
        p = tokenizer.apply_chat_template(row.prompt, add_generation_prompt=True, **kw)
        pc = tokenizer.apply_chat_template(list(row.prompt) + list(chosen), **kw)
        pr = tokenizer.apply_chat_template(list(row.prompt) + list(rejected), **kw)
    else:
        p = tokenizer.encode(str(row.prompt))
        pc = tokenizer.encode(str(row.prompt) + str(chosen))
        pr = tokenizer.encode(str(row.prompt) + str(rejected))

    plen = min(common_prefix_length(p, pc), common_prefix_length(p, pr))
    prompt_ids, chosen_ids, rejected_ids = p[:plen], pc[plen:], pr[plen:]

    max_length, mode = config.max_length, config.truncation_mode
    dropped = max_length is not None and mode == "keep_start" and len(prompt_ids) >= max_length
    c_trunc, c_mask = truncate(prompt_ids + chosen_ids, [0] * len(prompt_ids) + [1] * len(chosen_ids), max_length, mode)
    r_trunc, r_mask = truncate(prompt_ids + rejected_ids, [0] * len(prompt_ids) + [1] * len(rejected_ids), max_length, mode)
    was_truncated = max_length is not None and (
        len(prompt_ids) + len(chosen_ids) > max_length or len(prompt_ids) + len(rejected_ids) > max_length
    )
    return TokenizedPair(
        prompt_ids=prompt_ids,
        chosen_ids=chosen_ids,
        rejected_ids=rejected_ids,
        prompt_tokens_full=len(p),
        dropped_tail=list(p[plen:]),
        dropped=dropped,
        chosen_trunc=c_trunc,
        rejected_trunc=r_trunc,
        chosen_mask_sum=sum(c_mask),
        rejected_mask_sum=sum(r_mask),
        was_truncated=was_truncated,
    )
