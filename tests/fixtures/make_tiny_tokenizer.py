"""Regenerate the two tiny tokenizer fixtures (needs the `tokenizers` package).

    uv run python tests/fixtures/make_tiny_tokenizer.py

Both are WordLevel tokenizers over a ~50 word vocabulary, split on whitespace, so every
expected token count in the tests can be computed by hand. No model files are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

HERE = Path(__file__).parent

SPECIALS = ["[UNK]", "<|im_start|>", "<|im_end|>", "</s>"]
WORDS = (
    "user assistant system the sky is blue green what color grass red a b c d e f g h "
    "i j k l m n count 1 2 3 4 5 6 7 8 9 hello there yes no"
).split()

# ChatML-like: every message is `<|im_start|> role content <|im_end|>`.
CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }} {{ m.content }}<|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
)
# Wrapping template: a header carries the number of messages, so rendering the prompt alone
# is never a prefix of rendering prompt + response.
WRAP = (
    "<|im_start|>count {{ messages|length }} <|im_end|> "
    "{% for m in messages %}<|im_start|>{{ m.role }} {{ m.content }}<|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
)


def build(folder: str, template: str) -> None:
    vocab = {tok: i for i, tok in enumerate(SPECIALS + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(SPECIALS)
    out = HERE / folder
    out.mkdir(exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": "</s>",
        "unk_token": "[UNK]",
        "model_max_length": 512,
        "chat_template": template,
    }
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build("tiny-tokenizer", CHATML)
    build("tiny-tokenizer-wrap", WRAP)
    print("wrote tiny-tokenizer and tiny-tokenizer-wrap")
