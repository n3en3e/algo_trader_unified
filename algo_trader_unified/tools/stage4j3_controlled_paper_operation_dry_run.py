"""Build the Stage 4J-3 controlled scheduled PAPER operation dry-run report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4j3_controlled_paper_operation_dry_run import (
    build_stage4j3_controlled_paper_operation_dry_run_report,
)


def run_stage4j3_controlled_paper_operation_dry_run(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4j3_controlled_paper_operation_dry_run_report,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4J-3 controlled PAPER operation dry run requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--stage4j2-plan-json", required=True)
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
        stage4j2_operation_plan_report = json.loads(args.stage4j2_plan_json)
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
                        "stage4j3_controlled_paper_operation_dry_run_report": True,
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
        stage4j2_operation_plan_report=stage4j2_operation_plan_report,
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
    readiness = report.get("readiness_for_stage4j4", {})
    return (
        0
        if readiness.get("ready_to_build_controlled_paper_operation_execution_gate") is True
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    return run_stage4j3_controlled_paper_operation_dry_run(
        sys.argv[1:] if argv is None else argv
    )


def _loads_optional(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4j4", {})
    artifacts = report.get("artifact_checks", {})
    selected = report.get("selected_strategy", {})
    operation = report.get("operation", {})
    lines = [
        "Stage 4J-3 controlled scheduled PAPER operation dry run",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_build_controlled_paper_operation_execution_gate: "
        f"{readiness.get('ready_to_build_controlled_paper_operation_execution_gate')}",
        f"stage4j2_report_ready: {artifacts.get('stage4j2_report_ready')}",
        f"selected_strategy_id: {selected.get('selected_strategy_id')}",
        f"operation_id: {operation.get('operation_id') if isinstance(operation, dict) else None}",
        f"trace_steps: {len(report.get('dry_run_trace', []))}",
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
