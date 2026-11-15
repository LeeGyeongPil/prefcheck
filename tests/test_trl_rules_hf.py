"""TRL 1.18 preparation rules with a real `transformers` tokenizer (local fixture, no network). AC-2.

Hand computation with tests/fixtures/tiny-tokenizer (ChatML-like template, one token per word):
  prompt alone  = <|im_start|> user what color is the sky <|im_end|> <|im_start|> assistant   -> 10 tokens
  prompt+chosen = 8 tokens of the user turn + <|im_start|> assistant the sky is blue <|im_end|> -> 15 tokens
  completion    = the sky is blue <|im_end|> -> 5 tokens (rejected "green": 5 tokens)
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from prefcheck import CheckConfig, check_rows, load_tokenizer  # noqa: E402
from prefcheck.formats import parse_row  # noqa: E402
from prefcheck.trl_rules import tokenize_pair  # noqa: E402

from conftest import conv_row  # noqa: E402

CONV = conv_row("what color is the sky", "the sky is blue", "the sky is green")


def test_tiny_tokenizer_matches_hand_computed(tiny_tokenizer_path):
    tok = load_tokenizer(tiny_tokenizer_path, offline=True)
    assert tok.eos_token == "</s>"

    # conversational: common prefix decides the prompt length
    row = parse_row(CONV)
    pair = tokenize_pair(row, tok, CheckConfig(max_length=None), add_eos=False)
    assert (len(pair.prompt_ids), len(pair.chosen_ids), len(pair.rejected_ids)) == (10, 5, 5)
    assert pair.dropped_tail == [] and not pair.dropped

    # keep_start 12: only 2 completion tokens survive and both sequences become identical
    pair = tokenize_pair(row, tok, CheckConfig(max_length=12), add_eos=False)
    assert (pair.chosen_mask_sum, pair.rejected_mask_sum) == (2, 2)
    assert pair.identical_after_truncation and not pair.dropped

    # keep_start 10: the prompt alone fills max_length -> TRL drops the row
    assert tokenize_pair(row, tok, CheckConfig(max_length=10), add_eos=False).dropped
    assert not tokenize_pair(row, tok, CheckConfig(max_length=11), add_eos=False).dropped

    # keep_end 12: the last 12 tokens keep all 5 completion tokens, nothing is dropped, rows differ
    pair = tokenize_pair(row, tok, CheckConfig(max_length=12, truncation_mode="keep_end"), add_eos=False)
    assert (pair.chosen_mask_sum, pair.rejected_mask_sum) == (5, 5)
    assert not pair.dropped and not pair.identical_after_truncation

    # standard format: EOS is appended once, and not again when it is already there
    std = parse_row({"prompt": "the sky is", "chosen": " blue", "rejected": " green"})
    pair = tokenize_pair(std, tok, CheckConfig(max_length=None), add_eos=True)
    assert (len(pair.prompt_ids), len(pair.chosen_ids)) == (3, 2)  # blue </s>
    std_eos = parse_row({"prompt": "the sky is", "chosen": " blue</s>", "rejected": " green</s>"})
    assert len(tokenize_pair(std_eos, tok, CheckConfig(max_length=None), add_eos=True).chosen_ids) == 2
    assert len(tokenize_pair(std, tok, CheckConfig(max_length=None), add_eos=False).chosen_ids) == 1  # no EOS

    pair = tokenize_pair(std, tok, CheckConfig(max_length=4), add_eos=True)
    assert pair.chosen_mask_sum == 1  # keep_start 4 keeps `blue` only
    assert not tokenize_pair(std, tok, CheckConfig(max_length=4), add_eos=True).dropped
    assert tokenize_pair(std, tok, CheckConfig(max_length=3), add_eos=True).dropped
    pair = tokenize_pair(std, tok, CheckConfig(max_length=4, truncation_mode="keep_end"), add_eos=True)
    assert pair.chosen_mask_sum == 2  # last 4 of [the sky is blue </s>]


def test_report_in_hf_mode(tiny_tokenizer_path):
    tok = load_tokenizer(tiny_tokenizer_path, offline=True)
    report = check_rows([CONV], CheckConfig(max_length=12), tokenizer=tok)
    assert report.meta["mode"] == "hf" and report.meta["tokenizer"] == "tiny-tokenizer"
    assert [f.rule for f in report.findings] == ["identical_after_truncation"]
