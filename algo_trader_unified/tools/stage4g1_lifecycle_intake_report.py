"""Build the Stage 4G-1 manual lifecycle intake report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4g1_lifecycle_intake_report import (
    build_stage4g1_lifecycle_intake_report,
)


def run_stage4g1_lifecycle_intake_report(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4g1_lifecycle_intake_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--stage4f-acceptance-json", required=True)
    parser.add_argument("--smoke-test-json", required=True)
    parser.add_argument("--submit-json", required=True)
    parser.add_argument("--order-control-json", action="append", default=[])
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: Stage 4G-1 lifecycle intake report requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    try:
        stage4f_acceptance_report = json.loads(args.stage4f_acceptance_json)
        smoke_test_report = json.loads(args.smoke_test_json)
        submit_report = json.loads(args.submit_json)
        order_control_reports = [
            json.loads(value) for value in args.order_control_json
        ]
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "stage4g1_lifecycle_intake_report": True,
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
        stage4f_acceptance_report=stage4f_acceptance_report,
        smoke_test_report=smoke_test_report,
        submit_report=submit_report,
        order_control_reports=order_control_reports,
    )

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4g2", {})
    return (
        0
        if readiness.get("ready_to_build_manual_lifecycle_state_preview") is True
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    return run_stage4g1_lifecycle_intake_report(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4g2", {})
    candidate = report.get("lifecycle_intake_candidate", {})
    lines = [
        "Stage 4G-1 manual lifecycle intake report",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_build_manual_lifecycle_state_preview: "
        f"{readiness.get('ready_to_build_manual_lifecycle_state_preview')}",
        f"candidate_available: {candidate.get('available')}",
        f"broker_order_id: {candidate.get('broker_order_id')}",
        f"client_order_id: {candidate.get('client_order_id')}",
        f"last_known_broker_status: {candidate.get('last_known_broker_status')}",
        "suggested_internal_lifecycle_state: "
        f"{candidate.get('suggested_internal_lifecycle_state')}",
        f"reconciliation_required: {candidate.get('reconciliation_required')}",
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
