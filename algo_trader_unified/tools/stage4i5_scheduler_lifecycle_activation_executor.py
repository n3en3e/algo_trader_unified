"""Build the Stage 4I-5 scheduler/lifecycle activation executor report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4i5_scheduler_lifecycle_activation_executor import (
    build_stage4i5_scheduler_lifecycle_activation_executor_report,
)


def run_stage4i5_scheduler_lifecycle_activation_executor(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4i5_scheduler_lifecycle_activation_executor_report,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4I-5 scheduler/lifecycle activation executor requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--stage4i4-gate-json", required=True)
    parser.add_argument("--activation-snapshot-json")
    parser.add_argument("--state-snapshot-json")
    parser.add_argument("--risk-snapshot-json")
    parser.add_argument("--scheduler-snapshot-json")
    parser.add_argument("--lifecycle-snapshot-json")
    parser.add_argument("--paper-broker-snapshot-json")
    parser.add_argument("--market-window-snapshot-json")
    parser.add_argument("--allow-scheduler-lifecycle-activation", action="store_true")
    parser.add_argument("--ack", action="append", dest="operator_acknowledgements")
    args = parser.parse_args(argv)

    try:
        stage4i4_gate_report = json.loads(args.stage4i4_gate_json)
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
                        "dry_run": False,
                        "stage4i5_scheduler_lifecycle_activation_executor_report": True,
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
        stage4i4_activation_gate_report=stage4i4_gate_report,
        scheduler_activation_writer=_UnavailableWriter(),
        lifecycle_activation_writer=_UnavailableWriter(),
        audit_writer=None,
        activation_snapshot=activation_snapshot,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
        market_window_snapshot=market_window_snapshot,
        operator_acknowledgements=args.operator_acknowledgements or [],
        allow_scheduler_lifecycle_activation=args.allow_scheduler_lifecycle_activation,
    )
    if args.allow_scheduler_lifecycle_activation:
        report.setdefault("warnings", []).append(
            "real writer execution is unavailable from the Stage 4I-5 CLI; use injected writers in tests or an approved future phase"
        )
        report.setdefault("readiness_for_stage4i6", {}).setdefault("warnings", []).append(
            "real writer execution is unavailable from the Stage 4I-5 CLI; use injected writers in tests or an approved future phase"
        )
    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4i6", {})
    return (
        0
        if readiness.get("ready_to_build_scheduler_lifecycle_activation_acceptance") is True
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    return run_stage4i5_scheduler_lifecycle_activation_executor(
        sys.argv[1:] if argv is None else argv
    )


def _loads_optional(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


class _UnavailableWriter:
    def activate_scheduler(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("real scheduler activation writer is unavailable from the Stage 4I-5 CLI")

    def activate_lifecycle(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("real lifecycle activation writer is unavailable from the Stage 4I-5 CLI")


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4i6", {})
    selected = report.get("selected_strategy", {})
    execution = report.get("execution", {})
    lines = [
        "Stage 4I-5 scheduler/lifecycle activation executor",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_build_scheduler_lifecycle_activation_acceptance: "
        f"{readiness.get('ready_to_build_scheduler_lifecycle_activation_acceptance')}",
        f"selected_strategy_id: {selected.get('selected_strategy_id')}",
        f"attempted: {execution.get('attempted')}",
        f"completed: {execution.get('completed')}",
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
