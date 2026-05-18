"""Build the Stage 4L-2 signal readiness gate report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4l2_signal_readiness_gate import (
    build_stage4l2_signal_readiness_gate_report,
)


def run_stage4l2_signal_readiness_gate(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4l2_signal_readiness_gate_report,
) -> int:
    if "--dry-run-only" not in argv:
        print("ERROR: Stage 4L-2 signal readiness gate CLI requires --dry-run-only because it is gate/reporting only", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--stage4l1-plan-json", required=True)
    parser.add_argument("--ack", action="append", default=[])
    parser.add_argument("--strategy-registry-snapshot-json")
    parser.add_argument("--signal-schema-snapshot-json")
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
        stage4l1_signal_readiness_plan_report = json.loads(args.stage4l1_plan_json)
        strategy_registry_snapshot = _loads_optional(args.strategy_registry_snapshot_json)
        signal_schema_snapshot = _loads_optional(args.signal_schema_snapshot_json)
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
                        "stage4l2_signal_readiness_gate_report": True,
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
        stage4l1_signal_readiness_plan_report=stage4l1_signal_readiness_plan_report,
        operator_acknowledgements=args.ack,
        strategy_registry_snapshot=strategy_registry_snapshot,
        signal_schema_snapshot=signal_schema_snapshot,
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
    readiness = report.get("readiness_for_stage4l3", {})
    return 0 if isinstance(readiness, dict) and readiness.get("ready_to_execute_controlled_signal_readiness_validator") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_stage4l2_signal_readiness_gate(sys.argv[1:] if argv is None else argv)


def _loads_optional(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4l3", {})
    selected = report.get("selected_strategy", {})
    operation = report.get("operation", {})
    gate = report.get("signal_readiness_gate", {})
    lines = [
        "Stage 4L-2 signal readiness gate",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"ready_to_execute_controlled_signal_readiness_validator: {readiness.get('ready_to_execute_controlled_signal_readiness_validator')}",
        f"next_recommended_phase: {readiness.get('next_recommended_phase')}",
        f"gate_status: {gate.get('gate_status')}",
        f"selected_strategy_id: {selected.get('selected_strategy_id')}",
        f"operation_id: {operation.get('operation_id')}",
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
