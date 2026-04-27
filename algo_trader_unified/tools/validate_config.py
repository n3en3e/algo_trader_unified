"""Validate Phase 1 static configuration."""

from __future__ import annotations

import re
import sys

from algo_trader_unified.config.portfolio import (
    PAPER_RESEARCH_CAPITAL,
    STRATEGY_ALLOCATIONS,
)
from algo_trader_unified.config.variants import VALID_EXECUTION_MODES


_ID_RE = re.compile(r"^S0[1-8]_[A-Z0-9_]+$")
OFF_IBKR_LEG_TYPES = {"CRYPTO_SPOT", "MANUAL_OPPORTUNITY"}
OFF_IBKR_LEG_SPECS: tuple[dict[str, object], ...] = ()


def validate() -> list[str]:
    errors: list[str] = []
    total = sum(STRATEGY_ALLOCATIONS.values())
    if total != PAPER_RESEARCH_CAPITAL:
        errors.append(
            f"allocations sum to {total}, expected {PAPER_RESEARCH_CAPITAL}"
        )
    ids = list(STRATEGY_ALLOCATIONS)
    if len(ids) != len(set(ids)):
        errors.append("strategy IDs are not unique")
    for strategy_id in ids:
        if not _ID_RE.match(strategy_id):
            errors.append(f"strategy ID has invalid format: {strategy_id}")
    if not VALID_EXECUTION_MODES:
        errors.append("no execution modes configured")
    for leg in OFF_IBKR_LEG_SPECS:
        if leg.get("secType") in OFF_IBKR_LEG_TYPES and leg.get("conId") is not None:
            errors.append(f"off-IBKR leg must not require conId: {leg}")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
        return 1
    print("PASS config validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
