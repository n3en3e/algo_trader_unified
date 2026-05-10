"""Build the Stage 4F-5 manual one-order paper smoke-test report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4f5_smoke_test_report import (
    build_stage4f5_smoke_test_report,
)


def run_stage4f5_smoke_test_report(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4f5_smoke_test_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--connection-preflight-json", required=True)
    parser.add_argument("--ticket-json", required=True)
    parser.add_argument("--submit-json", required=True)
    parser.add_argument("--order-control-json", action="append", default=[])
    parser.add_argument("--operator-note-json", default=None)
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: Stage 4F-5 smoke-test report requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    try:
        connection_preflight_report = json.loads(args.connection_preflight_json)
        ticket_report = json.loads(args.ticket_json)
        submit_report = json.loads(args.submit_json)
        order_control_reports = [
            json.loads(value) for value in args.order_control_json
        ]
        operator_notes = (
            json.loads(args.operator_note_json)
            if args.operator_note_json is not None
            else None
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "stage4f5_smoke_test_report": True,
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
        connection_preflight_report=connection_preflight_report,
        ticket_report=ticket_report,
        submit_report=submit_report,
        order_control_reports=order_control_reports,
        operator_notes=operator_notes,
    )

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4f6", {})
    return 0 if readiness.get("ready_for_stage4f_acceptance_report") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_stage4f5_smoke_test_report(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    smoke_test = report.get("smoke_test", {})
    readiness = report.get("readiness_for_stage4f6", {})
    order_control = report.get("order_control_summary", {})
    lines = [
        "Stage 4F-5 manual one-order smoke-test report",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"accepted: {smoke_test.get('accepted')}",
        f"ready_for_stage4f_acceptance_report: {readiness.get('ready_for_stage4f_acceptance_report')}",
        f"broker_order_id: {smoke_test.get('broker_order_id')}",
        f"client_order_id: {smoke_test.get('client_order_id')}",
        f"submitted: {smoke_test.get('submitted')}",
        f"status_seen: {smoke_test.get('status_seen')}",
        f"cancel_seen: {smoke_test.get('cancel_seen')}",
        f"terminal_or_safe_state_seen: {smoke_test.get('terminal_or_safe_state_seen')}",
        f"last_known_status: {order_control.get('last_known_status')}",
    ]
    for key in ("blockers", "warnings"):
        values = readiness.get(key)
        if values:
            lines.append(f"{key}:")
            for value in values:
                lines.append(f"  - {value}")
    for key in ("errors", "warnings"):
        values = report.get(key)
        if values:
            lines.append(f"report_{key}:")
            for value in values:
                lines.append(f"  - {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
