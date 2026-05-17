"""Build the Stage 4J-6 controlled scheduled PAPER operation acceptance report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4j6_controlled_paper_operation_acceptance import (
    build_stage4j6_controlled_paper_operation_acceptance_report,
)


def run_stage4j6_controlled_paper_operation_acceptance(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4j6_controlled_paper_operation_acceptance_report,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4J-6 controlled PAPER operation acceptance requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--stage4j5-executor-json", required=True)
    parser.add_argument("--scheduler-activation-snapshot-json")
    parser.add_argument("--lifecycle-activation-snapshot-json")
    parser.add_argument("--activation-snapshot-json")
    parser.add_argument("--state-snapshot-json")
    parser.add_argument("--risk-snapshot-json")
    parser.add_argument("--scheduler-snapshot-json")
    parser.add_argument("--lifecycle-snapshot-json")
    parser.add_argument("--paper-broker-snapshot-json")
    parser.add_argument("--market-window-snapshot-json")
    args = parser.parse_args(argv)

    try:
        stage4j5_executor_report = json.loads(args.stage4j5_executor_json)
        scheduler_activation_snapshot = _loads_optional(args.scheduler_activation_snapshot_json)
        lifecycle_activation_snapshot = _loads_optional(args.lifecycle_activation_snapshot_json)
        activation_snapshot = _loads_optional(args.activation_snapshot_json)
        state_snapshot = _loads_optional(args.state_snapshot_json)
        risk_snapshot = _loads_optional(args.risk_snapshot_json)
        scheduler_snapshot = _loads_optional(args.scheduler_snapshot_json)
        lifecycle_snapshot = _loads_optional(args.lifecycle_snapshot_json)
        paper_broker_snapshot = _loads_optional(args.paper_broker_snapshot_json)
        market_window_snapshot = _loads_optional(args.market_window_snapshot_json)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "stage4j6_controlled_paper_operation_acceptance_report": True,
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
        stage4j5_executor_report=stage4j5_executor_report,
        scheduler_activation_snapshot=scheduler_activation_snapshot,
        lifecycle_activation_snapshot=lifecycle_activation_snapshot,
        activation_snapshot=activation_snapshot,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
        market_window_snapshot=market_window_snapshot,
    )
    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4j_complete_or_next_gate", {})
    return 0 if readiness.get("ready_for_next_explicit_gate") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_stage4j6_controlled_paper_operation_acceptance(sys.argv[1:] if argv is None else argv)


def _loads_optional(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4j_complete_or_next_gate", {})
    acceptance = report.get("executor_acceptance", {})
    selected = report.get("selected_strategy", {})
    operation = report.get("operation", {})
    lines = [
        "Stage 4J-6 controlled scheduled PAPER operation acceptance",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"accepted: {acceptance.get('accepted') if isinstance(acceptance, dict) else None}",
        f"acceptance_status: {acceptance.get('acceptance_status') if isinstance(acceptance, dict) else None}",
        f"selected_strategy_id: {selected.get('selected_strategy_id') if isinstance(selected, dict) else None}",
        f"operation_id: {operation.get('operation_id') if isinstance(operation, dict) else None}",
        f"recommended_next_gate: {readiness.get('recommended_next_gate') if isinstance(readiness, dict) else None}",
    ]
    blockers = readiness.get("blockers") if isinstance(readiness, dict) else None
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
