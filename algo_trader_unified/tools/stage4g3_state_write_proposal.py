"""Build the Stage 4G-3 manual state write proposal report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4g3_state_write_proposal import (
    build_stage4g3_state_write_proposal,
)


def run_stage4g3_state_write_proposal(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4g3_state_write_proposal,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--lifecycle-state-preview-json", required=True)
    parser.add_argument("--state-snapshot-json")
    parser.add_argument("--operator-notes-json")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: Stage 4G-3 state write proposal requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    try:
        lifecycle_state_preview_report = json.loads(args.lifecycle_state_preview_json)
        existing_state_snapshot = (
            json.loads(args.state_snapshot_json)
            if args.state_snapshot_json is not None
            else None
        )
        operator_notes = (
            json.loads(args.operator_notes_json)
            if args.operator_notes_json is not None
            else None
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "stage4g3_state_write_proposal": True,
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
        lifecycle_state_preview_report=lifecycle_state_preview_report,
        existing_state_snapshot=existing_state_snapshot,
        operator_notes=operator_notes,
    )

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4g4", {})
    return (
        0
        if readiness.get("ready_to_build_manual_state_write_dry_run") is True
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    return run_stage4g3_state_write_proposal(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4g4", {})
    proposal = report.get("proposal", {})
    input_summary = report.get("input_summary", {})
    lines = [
        "Stage 4G-3 manual state write proposal",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_build_manual_state_write_dry_run: "
        f"{readiness.get('ready_to_build_manual_state_write_dry_run')}",
        f"proposal_available: {proposal.get('available')}",
        f"broker_order_id: {input_summary.get('broker_order_id')}",
        f"client_order_id: {input_summary.get('client_order_id')}",
        f"proposed_lifecycle_state: {input_summary.get('proposed_lifecycle_state')}",
        f"proposed_order_status: {input_summary.get('proposed_order_status')}",
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
