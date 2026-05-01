"""Shared validation helpers for lifecycle state transitions."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


def validate_numeric_field(
    name: str,
    value: Any,
    *,
    minimum: int | float,
    allow_equal: bool,
    allow_int: bool,
) -> int | float:
    allowed_types = (int, float, Decimal) if allow_int else (float, Decimal)
    if isinstance(value, bool) or not isinstance(value, allowed_types):
        raise ValueError(f"{name} must be numeric, not {type(value).__name__}")
    numeric = float(value) if isinstance(value, Decimal) else value
    if allow_equal:
        valid = numeric >= minimum
    else:
        valid = numeric > minimum
    if not valid:
        comparator = ">=" if allow_equal else ">"
        raise ValueError(f"{name} must be {comparator} {minimum}")
    return numeric
