"""Open a dry-run position from a filled order intent."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.positions import open_position_from_filled_intent
from algo_trader_unified.core.state_store import StateStore


def _parse_opened_at(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--opened-at must be a parseable ISO timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"--opened-at must be timezone-aware: {value!r}")
    return parsed.isoformat()


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_payload(position: dict) -> dict:
    return {
        "intent_id": position["intent_id"],
        "position_id": position["position_id"],
        "status": position["status"],
        "position_opened_event_id": position["position_opened_event_id"],
        "dry_run": position["dry_run"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent-id", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--opened-at", type=_parse_opened_at)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Accepted for clarity; dry-run is the only supported position-open mode.",
    )
    args = parser.parse_args(argv)

    root_dir = Path(args.root_dir)
    opened_at = args.opened_at or _current_timestamp()
    try:
        state_store = StateStore(root_dir / "data/state/portfolio_state.json")
        ledger = LedgerAppender(root_dir)
        position = open_position_from_filled_intent(
            state_store=state_store,
            ledger=ledger,
            intent_id=args.intent_id,
            opened_at=opened_at,
        )
    except Exception as exc:
        print(f"open position failed for {args.intent_id!r}: {exc}", file=sys.stderr)
        return 1

    payload = _result_payload(position)
    if args.json_output:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    else:
        print(
            "opened position "
            f"intent_id={payload['intent_id']} "
            f"position_id={payload['position_id']} "
            f"status={payload['status']} "
            f"position_opened_event_id={payload['position_opened_event_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
