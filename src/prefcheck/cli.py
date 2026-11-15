"""Command line interface."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, NoReturn

from . import i18n
from ._version import __version__
from .checker import check_records
from .config import TRUNCATION_MODES, CheckConfig, Thresholds, config_kwargs_from_toml
from .errors import ConfigError, PrefcheckError
from .io import read_records, write_bad_rows, write_clean
from .report import FAIL_ON_CHOICES
from .rules import RULES
from .tokenization import ApproxTokenizer, Tokenizer, load_tokenizer

PROG = "prefcheck"

# Exit codes
EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2


def _preparse_lang(argv: Sequence[str]) -> str | None:
    """Find ``--lang`` before the real parse so --help can already be localised."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


class _Formatter(argparse.HelpFormatter):
    def __init__(self, *args: Any, lang: str = "en", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lang = lang

    def add_usage(self, usage: Any, actions: Any, groups: Any, prefix: str | None = None) -> None:
        if prefix is None:
            prefix = i18n.t(self._lang, "usage_prefix")
        super().add_usage(usage, actions, groups, prefix)


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args: Any, lang: str = "en", err: IO[str] | None = None, **kwargs: Any) -> None:
        self.lang = lang
        self._err = err
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        stream = self._err or sys.stderr
        self.print_usage(stream)
        stream.write(i18n.t(self.lang, "error.argparse", prog=self.prog, message=message) + "\n")
        raise SystemExit(EXIT_USAGE)


def build_parser(lang: str, err: IO[str] | None = None) -> argparse.ArgumentParser:
    def t(key: str, **p: Any) -> str:
        return i18n.t(lang, key, **p)

    parser = _Parser(
        prog=PROG,
        description=t("description"),
        add_help=False,
        lang=lang,
        err=err,
        formatter_class=lambda prog: _Formatter(prog, lang=lang, max_help_position=34),
    )
    parser._positionals.title = t("section.positional")
    parser._optionals.title = t("section.options")
    parser.add_argument("data", nargs="?", metavar="DATA", help=t("help.data"))
    parser.add_argument("--tokenizer", metavar="NAME|PATH", help=t("help.tokenizer"))
    parser.add_argument("--offline", action="store_true", help=t("help.offline"))
    parser.add_argument("--revision", metavar="REV", help=t("help.revision"))
    parser.add_argument("--max-length", metavar="N|none", help=t("help.max_length"))
    parser.add_argument("--truncation", choices=TRUNCATION_MODES, help=t("help.truncation"))
    parser.add_argument("--config", metavar="PATH", help=t("help.config"))
    parser.add_argument("--json", metavar="PATH", dest="json_path", help=t("help.json"))
    parser.add_argument("--markdown", metavar="PATH", help=t("help.markdown"))
    parser.add_argument("--bad-rows", metavar="PATH", help=t("help.bad_rows"))
    parser.add_argument("--clean", metavar="PATH", help=t("help.clean"))
    parser.add_argument("--fail-on", choices=FAIL_ON_CHOICES, default="error", help=t("help.fail_on"))
    parser.add_argument("--max-findings", type=int, default=50, metavar="N", help=t("help.max_findings"))
    parser.add_argument("--lang", choices=i18n.SUPPORTED_LANGS, help=t("help.lang"))
    parser.add_argument("--list-rules", action="store_true", help=t("help.list_rules"))
    parser.add_argument("--version", action="store_true", help=t("help.version"))
    parser.add_argument("-h", "--help", action="help", help=t("help.help"))
    return parser


def _parse_max_length(value: str) -> int | None:
    if value.strip().lower() == "none":
        return None
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n < 1:
        raise ConfigError(f"Invalid --max-length {value!r}: use a positive integer or 'none'.", key="error.max_length", value=value)
    return n


def _build_config(args: argparse.Namespace) -> CheckConfig:
    kwargs: dict[str, Any] = config_kwargs_from_toml(args.config) if args.config else {}
    if args.max_length is not None:
        kwargs["max_length"] = _parse_max_length(args.max_length)
    if args.truncation is not None:
        kwargs["truncation_mode"] = args.truncation
    kwargs.setdefault("thresholds", Thresholds())
    return CheckConfig(**kwargs)


def _print_rules(lang: str, out: IO[str]) -> None:
    out.write(i18n.t(lang, "rules.list_title") + "\n")
    width = max(len(r) for r in RULES)
    for rule_id, info in RULES.items():
        sev = i18n.t(lang, "severity." + info.severity)
        scope = i18n.t(lang, "scope." + info.scope)
        out.write(f"  {rule_id:<{width}}  {sev:<5} {scope}  {i18n.t(lang, info.desc_key)}\n")


def _error_text(exc: PrefcheckError, lang: str) -> str:
    if exc.key and i18n.has(lang, exc.key):
        return i18n.t(lang, exc.key, **exc.params)
    return str(exc)


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Run the CLI and return the exit code (0 ok, 1 findings at --fail-on level, 2 usage/input error)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if env is None else env
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    lang = i18n.resolve_lang(_preparse_lang(argv), env)

    parser = build_parser(lang, err)
    try:
        if "-h" in argv or "--help" in argv:
            # argparse prints help to the real stdout; route it to ours
            out.write(parser.format_help())
            return EXIT_OK
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE

    lang = i18n.resolve_lang(args.lang, env)
    try:
        return _run(args, lang, env, out, err)
    except PrefcheckError as exc:
        err.write(i18n.t(lang, "error.prefix", prog=PROG, message=_error_text(exc, lang)) + "\n")
        return EXIT_USAGE


def _run(args: argparse.Namespace, lang: str, env: Mapping[str, str], out: IO[str], err: IO[str]) -> int:
    if args.version:
        out.write(f"{PROG} {__version__}\n")
        return EXIT_OK
    if args.list_rules:
        _print_rules(lang, out)
        return EXIT_OK
    if not args.data:
        err.write(i18n.t(lang, "error.no_data", prog=PROG) + "\n")
        return EXIT_USAGE
    if args.max_findings < 0:
        raise ConfigError("--max-findings must be 0 or greater.", key="error.max_findings")

    _reject_output_equal_to_input(args)
    config = _build_config(args)
    tokenizer: Tokenizer
    if args.tokenizer:
        offline = True if args.offline else None  # None: honour HF_HUB_OFFLINE
        tokenizer = load_tokenizer(args.tokenizer, offline=offline, revision=args.revision)
    else:
        tokenizer = ApproxTokenizer()

    records = read_records(args.data)
    report = check_records(records, config, tokenizer)
    report.meta["input"] = Path(args.data).name

    written: list[str] = []
    if args.json_path:
        _write_text(args.json_path, report.to_json(lang))
        written.append(args.json_path)
    if args.markdown:
        _write_text(args.markdown, report.to_markdown(lang))
        written.append(args.markdown)
    if args.bad_rows:
        write_bad_rows(args.bad_rows, records, report.error_rules_by_row())
        written.append(args.bad_rows)
    if args.clean:
        write_clean(args.clean, records, report.bad_row_indices())
        written.append(args.clean)

    out.write(report.render_terminal(lang, args.max_findings))
    for path in written:
        out.write(i18n.t(lang, "wrote", path=path) + "\n")
    return report.exit_code(args.fail_on)


def _reject_output_equal_to_input(args: argparse.Namespace) -> None:
    """Refuse to overwrite the input file with --json/--markdown/--bad-rows/--clean output."""
    source = Path(args.data).resolve()
    for out_path in (args.json_path, args.markdown, args.bad_rows, args.clean):
        if out_path and Path(out_path).resolve() == source:
            raise ConfigError(
                f"Output path {out_path} is the same as the input file and would overwrite it. Choose a different output path.",
                key="error.output_is_input", path=out_path,
            )


def _write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
