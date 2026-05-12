"""Build the Stage 4G-5 manual state write executor report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.stage4g5_state_write_executor import (
    build_stage4g5_state_write_executor_report,
)


class _UnavailableCliStateWriter:
    def upsert_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("real StateStore writer execution is unavailable from CLI")

    def upsert_position(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("real StateStore writer execution is unavailable from CLI")


class _UnavailableCliLedgerWriter:
    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("real ledger writer execution is unavailable from CLI")


def run_stage4g5_state_write_executor(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_stage4g5_state_write_executor_report,
    state_store_writer: Any | None = None,
    ledger_writer: Any | None = None,
) -> int:
    if "--dry-run-only" not in argv:
        print(
            "ERROR: Stage 4G-5 state write executor requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--state-write-dry-run-json", required=True)
    parser.add_argument("--allow-state-write", action="store_true")
    parser.add_argument("--allow-ledger-write", action="store_true")
    parser.add_argument("--ack", action="append", default=[])
    args = parser.parse_args(argv)

    try:
        state_write_dry_run_report = json.loads(args.state_write_dry_run_json)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports parse type + text.
        message = f"ERROR: invalid JSON input: {type(exc).__name__}: {exc}"
        if args.json_output:
            print(
                json.dumps(
                    {
                        "dry_run": False,
                        "stage4g5_state_write_executor": True,
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
        state_write_dry_run_report=state_write_dry_run_report,
        state_store_writer=state_store_writer or _UnavailableCliStateWriter(),
        ledger_writer=ledger_writer or _UnavailableCliLedgerWriter(),
        operator_acknowledgements=args.ack,
        allow_state_write=args.allow_state_write,
        allow_ledger_write=args.allow_ledger_write,
    )

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    readiness = report.get("readiness_for_stage4g6", {})
    return (
        0
        if readiness.get("ready_to_build_manual_lifecycle_write_acceptance_report")
        is True
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    return run_stage4g5_state_write_executor(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    readiness = report.get("readiness_for_stage4g6", {})
    execution = report.get("execution", {})
    lines = [
        "Stage 4G-5 manual state write executor",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        "ready_to_build_manual_lifecycle_write_acceptance_report: "
        f"{readiness.get('ready_to_build_manual_lifecycle_write_acceptance_report')}",
        f"execution_attempted: {execution.get('attempted')}",
        f"completed: {execution.get('completed')}",
        "cli_real_writer_execution: unavailable",
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
