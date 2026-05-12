"""Build the Stage 4G-6 manual lifecycle write acceptance report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4g6_lifecycle_write_acceptance import (
    build_stage4g6_lifecycle_write_acceptance_report,
)


def run_stage4g6_lifecycle_write_acceptance(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4g6_lifecycle_write_acceptance_report,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4G-6 lifecycle write acceptance requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--state-write-executor-json", required=True)
    parser.add_argument("--state-snapshot-json")
    parser.add_argument("--ledger-snapshot-json")
    args = parser.parse_args(argv)

    try:
        executor_report = json.loads(args.state_write_executor_json)
        state_snapshot = (
            json.loads(args.state_snapshot_json)
            if args.state_snapshot_json is not None
            else None
        )
        ledger_snapshot = (
            json.loads(args.ledger_snapshot_json)
            if args.ledger_snapshot_json is not None
            else None
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "stage4g6_lifecycle_write_acceptance_report": True,
                        "success": False,
                        "errors": [message],
                        "warnings": [],
                    },
                    sort_keys=True,
                )
            )
        else:
            print(message, file=sys.stderr)
        return 1

    report = report_builder(
        state_write_executor_report=executor_report,
        existing_state_snapshot=state_snapshot,
        ledger_snapshot=ledger_snapshot,
    )
    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4h", {})
    return (
        0
        if readiness.get("ready_to_begin_controlled_automated_paper_trading_launch")
        is True
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    return run_stage4g6_lifecycle_write_acceptance(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4h", {})
    checks = report.get("artifact_checks", {})
    lines = [
        "Stage 4G-6 manual lifecycle write acceptance",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_begin_controlled_automated_paper_trading_launch: "
        f"{readiness.get('ready_to_begin_controlled_automated_paper_trading_launch')}",
        f"executor_completed: {checks.get('executor_completed')}",
        f"state_store_write_succeeded: {checks.get('state_store_write_succeeded')}",
        f"ledger_write_succeeded: {checks.get('ledger_write_succeeded')}",
    ]
    blockers = readiness.get("blockers")
    if blockers:
        lines.append("blockers:")
        for value in blockers:
            lines.append(f"  - {value}")
    for key in ("warnings", "errors"):
        values = report.get(key)
        if values:
            lines.append(f"{key}:")
            for value in values:
                lines.append(f"  - {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
