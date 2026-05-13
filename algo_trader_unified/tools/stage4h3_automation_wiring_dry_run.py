"""Build the Stage 4H-3 controlled automation wiring dry-run report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4h3_automation_wiring_dry_run import (
    build_stage4h3_automation_wiring_dry_run_report,
)


def run_stage4h3_automation_wiring_dry_run(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4h3_automation_wiring_dry_run_report,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4H-3 automation wiring dry run requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--stage4h2-preview-json", required=True)
    parser.add_argument("--state-snapshot-json")
    parser.add_argument("--risk-snapshot-json")
    parser.add_argument("--scheduler-snapshot-json")
    parser.add_argument("--lifecycle-snapshot-json")
    parser.add_argument("--paper-broker-snapshot-json")
    args = parser.parse_args(argv)

    try:
        stage4h2_wiring_preview_report = json.loads(args.stage4h2_preview_json)
        state_snapshot = _loads_optional(args.state_snapshot_json)
        risk_snapshot = _loads_optional(args.risk_snapshot_json)
        scheduler_snapshot = _loads_optional(args.scheduler_snapshot_json)
        lifecycle_snapshot = _loads_optional(args.lifecycle_snapshot_json)
        paper_broker_snapshot = _loads_optional(args.paper_broker_snapshot_json)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "stage4h3_automation_wiring_dry_run_report": True,
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
        stage4h2_wiring_preview_report=stage4h2_wiring_preview_report,
        state_snapshot=state_snapshot,
        risk_snapshot=risk_snapshot,
        scheduler_snapshot=scheduler_snapshot,
        lifecycle_snapshot=lifecycle_snapshot,
        paper_broker_snapshot=paper_broker_snapshot,
    )
    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4h4", {})
    return 0 if readiness.get("ready_to_build_one_strategy_automation_enablement_gate") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_stage4h3_automation_wiring_dry_run(sys.argv[1:] if argv is None else argv)


def _loads_optional(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4h4", {})
    artifacts = report.get("artifact_checks", {})
    selected = report.get("selected_strategy", {})
    scheduler = report.get("scheduler_checks", {})
    lifecycle = report.get("lifecycle_checks", {})
    lines = [
        "Stage 4H-3 controlled automation wiring dry run",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_build_one_strategy_automation_enablement_gate: "
        f"{readiness.get('ready_to_build_one_strategy_automation_enablement_gate')}",
        f"stage4h2_preview_ready: {artifacts.get('stage4h2_preview_ready')}",
        f"selected_preview_strategy_id: {selected.get('selected_preview_strategy_id')}",
        f"scheduler_already_enabled: {scheduler.get('scheduler_already_enabled')}",
        f"lifecycle_already_enabled: {lifecycle.get('lifecycle_already_enabled')}",
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
