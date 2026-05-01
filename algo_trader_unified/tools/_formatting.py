"""Read-only formatting helpers for operator tools."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        default=json_default,
        separators=(",", ":"),
        sort_keys=True,
    )


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("--limit must be non-negative")
    return parsed


def limit_arg(value: str) -> int:
    try:
        return non_negative_int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def state_path(root_dir: str) -> Path:
    return Path(root_dir) / "data/state/portfolio_state.json"


def apply_limit(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    return records[:limit]


def sort_records(
    records: list[dict[str, Any]],
    sort_key: str | None,
    *,
    reverse: bool = False,
) -> list[dict[str, Any]]:
    if sort_key is None:
        return list(records)

    def key(record: dict[str, Any]) -> tuple[bool, str]:
        value = record.get(sort_key)
        missing = value is None
        if isinstance(value, (datetime, date)):
            normalized = value.isoformat()
        else:
            normalized = "" if value is None else str(value)
        return (missing, normalized)

    return sorted(records, key=key, reverse=reverse)


def compact_table(records: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if columns is None:
        if not records:
            return ""
        columns = list(records[0].keys())

    rows = [[_display_value(record.get(column)) for column in columns] for record in records]
    widths = [
        max([len(column), *(len(row[index]) for row in rows)])
        for index, column in enumerate(columns)
    ]
    lines = [
        "  ".join(column.ljust(widths[index]) for index, column in enumerate(columns)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(lines)


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)
