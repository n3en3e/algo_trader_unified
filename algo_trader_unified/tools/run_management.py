"""Run one management signal scan."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.management import (
    default_management_signal_provider,
    run_management_scan,
)
from algo_trader_unified.core.state_store import StateStore

DEFAULT_MANAGEMENT_SIGNAL_PROVIDER = default_management_signal_provider


def _parse_now(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--now must be a parseable ISO timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"--now must be timezone-aware: {value!r}")
    return parsed.isoformat()


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _human_summary(result: dict) -> str:
    created = result["close_intents_created_count"]
    evaluated = result["evaluated_count"]
    skipped = result["skipped_active_close_intent_count"]
    errors = result["errors_count"]
    if created == 0:
        return (
            "management scan complete: no close intents created "
            f"evaluated={evaluated} skipped_active_close_intent={skipped} errors={errors}"
        )
    return (
        "management scan complete: "
        f"close_intents_created={created} evaluated={evaluated} "
        f"skipped_active_close_intent={skipped} errors={errors}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--strategy-id")
    parser.add_argument("--now", type=_parse_now)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Accepted for clarity; management scan close intents are dry-run only.",
    )
    args = parser.parse_args(argv)

    root_dir = Path(args.root_dir)
    now = args.now or _current_timestamp()

    state_store = StateStore(root_dir / "data/state/portfolio_state.json")
    ledger = LedgerAppender(root_dir)
    result = run_management_scan(
        state_store=state_store,
        ledger=ledger,
        strategy_id=args.strategy_id,
        management_signal_provider=DEFAULT_MANAGEMENT_SIGNAL_PROVIDER,
        now=now,
    )
    result["dry_run"] = True

    if args.json_output:
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    else:
        print(_human_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
