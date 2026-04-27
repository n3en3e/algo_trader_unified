"""Halt account, sleeve, or strategy scope."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.config.portfolio import DIAGNOSTIC_CLIENT_ID_RANGE
from algo_trader_unified.core.ledger import LedgerAppender


def _validate_scope_id(scope: str, scope_id: str | None) -> str | None:
    if scope == "account" and scope_id is not None:
        raise ValueError("--id must be omitted for account scope")
    if scope in {"sleeve", "strategy"} and not scope_id:
        raise ValueError("--id is required for sleeve and strategy scope")
    return scope_id


def _write_halt_state(root: Path, payload: dict[str, object]) -> None:
    state_dir = root / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "halt_state.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--scope", choices=("account", "sleeve", "strategy"), required=True)
    parser.add_argument("--id", dest="scope_id")
    parser.add_argument("--tier", choices=("soft", "hard"), required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--client-id", type=int, default=20)
    args = parser.parse_args(argv)

    try:
        scope_id = _validate_scope_id(args.scope, args.scope_id)
        if args.client_id in DIAGNOSTIC_CLIENT_ID_RANGE:
            raise ValueError("diagnostic client IDs 90-99 may not run halt tooling")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    root = Path(args.root)
    triggered_at = datetime.now(timezone.utc).isoformat()
    ledger = LedgerAppender(root)
    event = ledger.append(
        event_type="HALT_TRIGGERED",
        strategy_id=scope_id if args.scope == "strategy" else "ACCOUNT",
        execution_mode="disabled",
        source_module="tools.halt",
        payload={
            "scope": args.scope,
            "id": scope_id,
            "tier": args.tier,
            "operator": args.operator,
            "reason": args.reason,
            "triggered_at": triggered_at,
        },
    )
    halt_state = {
        "scope": args.scope,
        "id": scope_id,
        "tier": args.tier,
        "operator": args.operator,
        "reason": args.reason,
        "triggered_at": triggered_at,
        "halt_event_id": event.event_id,
    }
    _write_halt_state(root, halt_state)
    print(f"halted scope={args.scope} tier={args.tier} event_id={event.event_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

