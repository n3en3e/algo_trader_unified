"""Resume a halted account, sleeve, or strategy scope."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.config.portfolio import DIAGNOSTIC_CLIENT_ID_RANGE
from algo_trader_unified.core.ledger import LedgerAppender


def _halt_state_path(root: Path) -> Path:
    return root / "data" / "state" / "halt_state.json"


def _write_halt_state(root: Path, payload: dict[str, object]) -> None:
    path = _halt_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--scope", choices=("account", "sleeve", "strategy"), required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--halt-event-id", required=True)
    parser.add_argument("--client-id", type=int, default=20)
    args = parser.parse_args(argv)

    if args.client_id in DIAGNOSTIC_CLIENT_ID_RANGE:
        print("ERROR: diagnostic client IDs 90-99 may not run resume tooling", file=sys.stderr)
        return 1

    root = Path(args.root)
    path = _halt_state_path(root)
    previous = {}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
    previous_tier = previous.get("tier")
    if previous_tier not in {"soft", "hard"}:
        print("ERROR: existing halt_state.json does not contain previous tier", file=sys.stderr)
        return 1

    resumed_at = datetime.now(timezone.utc).isoformat()
    triggered_at = str(previous.get("triggered_at") or "")
    ledger = LedgerAppender(root)
    event = ledger.append(
        event_type="HALT_RESUMED",
        strategy_id=previous.get("id") if args.scope == "strategy" else "ACCOUNT",
        execution_mode="disabled",
        source_module="tools.resume_halt",
        payload={
            "scope": args.scope,
            "previous_tier": previous_tier,
            "operator": args.operator,
            "reason": args.reason,
            "triggered_at": triggered_at,
            "resumed_at": resumed_at,
            "halt_event_id": args.halt_event_id,
        },
    )
    resumed_state = {
        "scope": args.scope,
        "tier": None,
        "resumed": True,
        "operator": args.operator,
        "reason": args.reason,
        "resumed_at": resumed_at,
        "resume_event_id": event.event_id,
        "halt_event_id": args.halt_event_id,
    }
    _write_halt_state(root, resumed_state)
    print(f"resumed scope={args.scope} event_id={event.event_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

