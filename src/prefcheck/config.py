"""Check configuration: training settings plus rule thresholds and severities."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigError

TruncationMode = Literal["keep_start", "keep_end"]
TRUNCATION_MODES = ("keep_start", "keep_end")
SEVERITY_CHOICES = ("off", "warn", "error")


@dataclass(frozen=True)
class Thresholds:
    """Rule thresholds. Defaults are deliberately conservative reference values."""

    near_identical_jaccard: float = 0.9
    length_ratio_max: float = 4.0
    length_bias_ratio: float = 0.7
    length_bias_r: float = 0.3

    def __post_init__(self) -> None:
        if not 0 < self.near_identical_jaccard <= 1:
            raise ConfigError("near_identical_jaccard must be in (0, 1].", key="error.config_value", name="near_identical_jaccard", reason="must be in (0, 1]")
        if self.length_ratio_max <= 1:
            raise ConfigError("length_ratio_max must be greater than 1.", key="error.config_value", name="length_ratio_max", reason="must be greater than 1")
        if not 0 < self.length_bias_ratio <= 1:
            raise ConfigError("length_bias_ratio must be in (0, 1].", key="error.config_value", name="length_bias_ratio", reason="must be in (0, 1]")
        if not -1 <= self.length_bias_r <= 1:
            raise ConfigError("length_bias_r must be in [-1, 1].", key="error.config_value", name="length_bias_r", reason="must be in [-1, 1]")


@dataclass(frozen=True)
class CheckConfig:
    """What to simulate and how strict to be.

    ``max_length`` and ``truncation_mode`` mirror TRL's ``DPOConfig``. ``max_length=None``
    means no truncation. ``severities`` maps a rule id to ``"off"``, ``"warn"`` or ``"error"``.
    """

    max_length: int | None = 1024
    truncation_mode: TruncationMode = "keep_start"
    thresholds: Thresholds = field(default_factory=Thresholds)
    severities: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from .rules import RULES  # local import: rules imports this module for types

        if self.max_length is not None and (not isinstance(self.max_length, int) or isinstance(self.max_length, bool) or self.max_length < 1):
            raise ConfigError("max_length must be a positive integer or None.", key="error.max_length", value=str(self.max_length))
        if self.truncation_mode not in TRUNCATION_MODES:
            raise ConfigError(
                f"truncation_mode must be one of {TRUNCATION_MODES}, got {self.truncation_mode!r}.",
                key="error.config_value", name="truncation_mode", reason=f"must be one of {', '.join(TRUNCATION_MODES)}",
            )
        for rule_id, severity in self.severities.items():
            if rule_id not in RULES:
                raise ConfigError(
                    f"Unknown rule id {rule_id!r}. Run `prefcheck --list-rules` to see valid ids.",
                    key="error.unknown_rule", rule=rule_id,
                )
            if severity not in SEVERITY_CHOICES:
                raise ConfigError(
                    f"Severity for {rule_id!r} must be one of {SEVERITY_CHOICES}, got {severity!r}.",
                    key="error.config_value", name=f"rules.{rule_id}.severity", reason=f"must be one of {', '.join(SEVERITY_CHOICES)}",
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "truncation_mode": self.truncation_mode,
            "thresholds": asdict(self.thresholds),
            "severities": dict(sorted(self.severities.items())),
        }

    @classmethod
    def from_toml(cls, path: str | Path) -> CheckConfig:
        """Read a ``prefcheck.toml``. See ``examples/prefcheck.toml`` for the layout."""
        return cls(**config_kwargs_from_toml(path))


def config_kwargs_from_toml(path: str | Path) -> dict[str, Any]:
    """Parse a TOML file into ``CheckConfig`` keyword arguments (validated by CheckConfig itself)."""
    p = Path(path)
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {p}. Check the path.", key="error.config_missing", path=str(p)) from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ConfigError(f"Cannot read config {p}: {exc}.", key="error.config", path=str(p), reason=str(exc)) from exc

    known_top = {"check", "thresholds", "rules"}
    unknown = sorted(set(data) - known_top)
    if unknown:
        raise ConfigError(
            f"Unknown config section(s) in {p}: {', '.join(unknown)}. Allowed: check, thresholds, rules.",
            key="error.config", path=str(p), reason=f"unknown section {', '.join(unknown)} (allowed: check, thresholds, rules)",
        )
    kwargs: dict[str, Any] = {}

    check = data.get("check", {})
    for key in check:
        if key not in ("max_length", "truncation"):
            raise ConfigError(f"Unknown key [check].{key} in {p}.", key="error.config", path=str(p), reason=f"unknown key [check].{key}")
    if "max_length" in check:
        value = check["max_length"]
        kwargs["max_length"] = None if value in ("none", "None") else value
    if "truncation" in check:
        kwargs["truncation_mode"] = check["truncation"]

    thresholds = data.get("thresholds", {})
    valid = set(Thresholds.__dataclass_fields__)
    for key in thresholds:
        if key not in valid:
            raise ConfigError(
                f"Unknown threshold {key!r} in {p}. Valid: {', '.join(sorted(valid))}.",
                key="error.config", path=str(p), reason=f"unknown threshold {key} (valid: {', '.join(sorted(valid))})",
            )
    if thresholds:
        kwargs["thresholds"] = Thresholds(**thresholds)

    severities: dict[str, str] = {}
    for rule_id, table in data.get("rules", {}).items():
        if not isinstance(table, dict) or "severity" not in table:
            raise ConfigError(
                f"[rules.{rule_id}] in {p} needs a `severity = \"off|warn|error\"` entry.",
                key="error.config", path=str(p), reason=f"[rules.{rule_id}] needs a severity entry",
            )
        severities[rule_id] = table["severity"]
    if severities:
        kwargs["severities"] = severities
    return kwargs
