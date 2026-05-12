"""Build the Stage 4G-4 manual state write dry-run report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4g4_state_write_dry_run import (
    build_stage4g4_state_write_dry_run,
)


def run_stage4g4_state_write_dry_run(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4g4_state_write_dry_run,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4G-4 state write dry run requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--state-write-proposal-json", required=True)
    parser.add_argument("--state-snapshot-json")
    parser.add_argument("--ack", action="append", default=[])
    args = parser.parse_args(argv)

    try:
        state_write_proposal_report = json.loads(args.state_write_proposal_json)
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
                        "stage4g4_state_write_dry_run": True,
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
        state_write_proposal_report=state_write_proposal_report,
        existing_state_snapshot=existing_state_snapshot,
        operator_acknowledgements=args.ack,
    )

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4g5", {})
    return (
        0
        if readiness.get("ready_to_build_manual_state_write_executor") is True
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    return run_stage4g4_state_write_dry_run(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4g5", {})
    packet = report.get("dry_run_packet", {})
    lines = [
        "Stage 4G-4 manual state write dry run",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_build_manual_state_write_executor: "
        f"{readiness.get('ready_to_build_manual_state_write_executor')}",
        f"dry_run_packet_available: {packet.get('available')}",
        f"dry_run_operation_count: {len(packet.get('dry_run_operations') or [])}",
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
