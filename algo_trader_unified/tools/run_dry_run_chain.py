"""Run the dry-run scheduler job chain once and exit."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.job_chain import run_dry_run_job_chain
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.readiness import ReadinessManager
from algo_trader_unified.core.scheduler import UnifiedScheduler
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools._formatting import dumps_json


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
    summary = result["summary"]
    return (
        f"dry-run chain {result['status']}: "
        f"steps_run={len(result['steps_run'])} "
        f"steps_skipped={len(result['steps_skipped'])} "
        f"errors={result['errors_count']} "
        f"entry_scan_runs={summary['entry_scan_runs']} "
        f"management_scan_runs={summary['management_scan_runs']} "
        f"submission_runs={summary['submission_runs']} "
        f"confirmation_runs={summary['confirmation_runs']} "
        f"fill_confirmation_runs={summary['fill_confirmation_runs']} "
        f"position_transition_runs={summary['position_transition_runs']}"
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
        help="Accepted for clarity; the chain is dry-run only.",
    )
    parser.add_argument("--skip-entry-scan", action="store_true")
    parser.add_argument("--skip-management-scan", action="store_true")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--skip-fill-confirmation", action="store_true")
    parser.add_argument("--skip-position-transitions", action="store_true")
    args = parser.parse_args(argv)

    try:
        root_dir = Path(args.root_dir)
        now = args.now or _current_timestamp()
        state_store = StateStore(root_dir / "data/state/portfolio_state.json")
        ledger = LedgerAppender(root_dir)
        readiness_manager = ReadinessManager(state_store, ledger)
        scheduler = UnifiedScheduler(
            state_store=state_store,
            ledger=ledger,
            readiness_manager=readiness_manager,
        )
        result = run_dry_run_job_chain(
            scheduler=scheduler,
            state_store=state_store,
            ledger=ledger,
            now=now,
            strategy_id=args.strategy_id,
            execution_adapter=DryRunExecutionAdapter(),
            include_entry_scan=not args.skip_entry_scan,
            include_management_scan=not args.skip_management_scan,
            include_submission=not args.skip_submission,
            include_confirmation=not args.skip_confirmation,
            include_fill_confirmation=not args.skip_fill_confirmation,
            include_position_transitions=not args.skip_position_transitions,
        )
    except ValueError as exc:
        print(f"dry-run chain failed: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(dumps_json(result))
    else:
        print(_human_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
