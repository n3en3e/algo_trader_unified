"""Build the Stage 4G-2 manual lifecycle state preview report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4g2_lifecycle_state_preview import (
    build_stage4g2_lifecycle_state_preview,
)


def run_stage4g2_lifecycle_state_preview(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4g2_lifecycle_state_preview,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--lifecycle-intake-json", required=True)
    parser.add_argument("--state-snapshot-json")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: Stage 4G-2 lifecycle state preview requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    try:
        lifecycle_intake_report = json.loads(args.lifecycle_intake_json)
        existing_state_snapshot = (
            json.loads(args.state_snapshot_json)
            if args.state_snapshot_json is not None
            else None
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "stage4g2_lifecycle_state_preview": True,
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
        lifecycle_intake_report=lifecycle_intake_report,
        existing_state_snapshot=existing_state_snapshot,
    )

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4g3", {})
    return (
        0
        if readiness.get("ready_to_build_manual_state_write_proposal") is True
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    return run_stage4g2_lifecycle_state_preview(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4g3", {})
    preview = report.get("preview", {})
    input_summary = report.get("input_summary", {})
    lines = [
        "Stage 4G-2 manual lifecycle state preview",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_build_manual_state_write_proposal: "
        f"{readiness.get('ready_to_build_manual_state_write_proposal')}",
        f"preview_available: {preview.get('available')}",
        f"broker_order_id: {input_summary.get('broker_order_id')}",
        f"client_order_id: {input_summary.get('client_order_id')}",
        "suggested_internal_lifecycle_state: "
        f"{input_summary.get('suggested_internal_lifecycle_state')}",
        f"proposed_lifecycle_state: {preview.get('proposed_lifecycle_state')}",
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
