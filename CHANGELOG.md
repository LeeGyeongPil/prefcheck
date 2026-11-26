# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-11-27

### Fixed
- `--clean` and `--bad-rows` now write a JSON array when the input is a `.json` array (they used to write JSONL, which `json.load` could not read). JSONL input output is unchanged.

## [0.1.1] - 2026-11-15

### Fixed
- `--max-findings 0` (and the Markdown renderer with a zero limit) no longer prints "No findings." when findings exist; it shows the truncation notice instead.

## [0.1.0] - 2026-11-15

### Added
- `prefcheck` CLI and Python library for linting DPO/ORPO preference datasets before training.
- Reproduction of the TRL 1.18 preparation steps (`trl-1.18` profile): implicit prompt extraction,
  EOS handling for standard data, common-prefix prompt boundary, the `keep_start` drop filter and
  `keep_start` / `keep_end` truncation of the completion mask.
- Rules: `schema_error`, `identical_pair`, `dropped_by_trl`, `empty_completion_after_truncation`,
  `identical_after_truncation`, `contradictory_label`, `prompt_not_prefix`, `near_identical_pair`,
  `extreme_length_ratio`, `duplicate_row`, `implicit_prompt_split`, `mixed_format`, `length_bias`.
- Metrics: dropped / no-signal ratios, truncation ratio, mean token counts, chosen-longer ratio,
  point-biserial correlation of length with the label, chosen/rejected length-ratio quantiles.
- Approximate (practice / mock) tokenizer that needs only the standard library, and a real
  tokenizer mode through the optional `prefcheck[hf]` extra (no torch required).
- Terminal summary, `--json`, `--markdown`, `--bad-rows`, `--clean`, exit codes 0/1/2, `--fail-on`,
  `--list-rules`, `prefcheck.toml` configuration.
- English and Korean screens (`--lang`, `PREFCHECK_LANG`, `LC_ALL` / `LC_MESSAGES` / `LANG`).

[Unreleased]: https://github.com/LeeGyeongPil/prefcheck/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/LeeGyeongPil/prefcheck/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/LeeGyeongPil/prefcheck/releases/tag/v0.1.0
