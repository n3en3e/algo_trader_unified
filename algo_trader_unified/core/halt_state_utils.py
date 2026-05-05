"""Small shared helpers for local halt-state checks."""

from __future__ import annotations

from typing import Any


def halt_is_active(halt_state: dict[str, Any] | None) -> bool:
    if not isinstance(halt_state, dict):
        return False
    if halt_state.get("resumed") is True:
        return False
    return halt_state.get("tier") in {"soft", "hard"}
