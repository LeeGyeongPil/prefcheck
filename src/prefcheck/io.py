"""Reading JSONL / JSON files (keeping line numbers and the original text) and writing row files."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InputFormatError


@dataclass(frozen=True)
class Record:
    """One input record. ``row`` is the 0-based record index, ``line`` the 1-based file line."""

    row: int
    line: int | None
    data: Any
    raw: str | None = None  # original text of the line (JSONL); None for JSON arrays / in-memory rows


def read_records(path: str | Path) -> list[Record]:
    """Read a ``.jsonl`` file (one JSON value per line) or a ``.json`` array of rows."""
    p = Path(path)
    try:
        text = p.read_bytes().decode("utf-8-sig")
    except FileNotFoundError:
        raise InputFormatError(f"File not found: {p}. Check the path.", key="error.file_not_found", path=str(p)) from None
    except IsADirectoryError:
        raise InputFormatError(f"{p} is a directory, expected a .jsonl or .json file.", key="error.read", path=str(p), reason="is a directory") from None
    except UnicodeDecodeError as exc:
        raise InputFormatError(
            f"Cannot read {p}: it is not valid UTF-8 ({exc.reason} at byte {exc.start}). Save the file as UTF-8 (JSONL) and retry.",
            key="error.not_utf8", path=str(p), reason=exc.reason, pos=exc.start,
        ) from exc
    except OSError as exc:
        raise InputFormatError(f"Cannot read {p}: {exc}.", key="error.read", path=str(p), reason=str(exc)) from exc

    if text.lstrip().startswith("["):
        return _read_json_array(p, text)
    return _read_jsonl(p, text)


def _read_json_array(p: Path, text: str) -> list[Record]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputFormatError(
            f"Invalid JSON in {p} (line {exc.lineno}, column {exc.colno}): {exc.msg}. Fix the syntax or use JSONL.",
            key="error.json_file", path=str(p), line=exc.lineno, col=exc.colno, msg=exc.msg,
        ) from exc
    if not data:
        raise InputFormatError(f"{p} has no rows.", key="error.no_rows", path=str(p))
    return [Record(row=i, line=None, data=item) for i, item in enumerate(data)]


def _read_jsonl(p: Path, text: str) -> list[Record]:
    records: list[Record] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        raw = line[:-1] if line.endswith("\r") else line
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InputFormatError(
                f"Invalid JSON at {p} line {lineno}: {exc.msg} (column {exc.colno}). Fix or remove that line.",
                key="error.json_line", path=str(p), line=lineno, col=exc.colno, msg=exc.msg,
            ) from exc
        records.append(Record(row=len(records), line=lineno, data=data, raw=raw))
    if not records:
        raise InputFormatError(f"{p} has no rows.", key="error.no_rows", path=str(p))
    return records


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _original(rec: Record) -> str:
    return rec.raw if rec.raw is not None else _dump(rec.data)


def _is_json_array(records: list[Record]) -> bool:
    """True when the records came from a ``.json`` array (they carry no file line number)."""
    return bool(records) and records[0].line is None and records[0].raw is None


def _write_json_array(path: str | Path, items: list[Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_clean(path: str | Path, records: list[Record], bad_rows: Collection[int]) -> int:
    """Write the records without error rows in the input's format (JSONL text kept as is, or a JSON array)."""
    kept = [r for r in records if r.row not in bad_rows]
    if _is_json_array(records):
        _write_json_array(path, [r.data for r in kept])
    else:
        _write_lines(path, [_original(r) for r in kept])
    return len(kept)


def write_bad_rows(path: str | Path, records: list[Record], rules_by_row: dict[int, list[str]]) -> int:
    """Write only the error rows: the original object plus a ``_prefcheck`` field listing the rule ids."""
    objs: list[dict[str, Any]] = []
    for rec in records:
        if rec.row not in rules_by_row:
            continue
        obj = dict(rec.data) if isinstance(rec.data, dict) else {"_value": rec.data}
        obj["_prefcheck"] = rules_by_row[rec.row]
        objs.append(obj)
    if _is_json_array(records):
        _write_json_array(path, objs)
    else:
        _write_lines(path, [_dump(o) for o in objs])
    return len(objs)


def _write_lines(path: str | Path, lines: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(line + "\n")
