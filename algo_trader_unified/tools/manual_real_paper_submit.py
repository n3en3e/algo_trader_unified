"""Run one manually gated real IBKR paper submit from injected JSON reports."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.manual_real_paper_submit import (
    build_manual_real_paper_submit_report,
)


def run_manual_real_paper_submit(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_manual_real_paper_submit_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--allow-real-ibkr", action="store_true")
    parser.add_argument("--allow-real-paper-submit", action="store_true")
    parser.add_argument("--ticket-json", required=True)
    parser.add_argument("--preflight-json", required=True)
    parser.add_argument("--ack", action="append", default=[])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4004)
    parser.add_argument("--client-id", type=int, default=7)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--trading-mode", default="PAPER")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: manual real IBKR paper submit requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    try:
        ticket_report = json.loads(args.ticket_json)
        preflight_report = json.loads(args.preflight_json)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON input: {exc}", file=sys.stderr)
        return 1

    config = {
        "host": args.host,
        "port": args.port,
        "client_id": args.client_id,
        "account_id": args.account_id,
        "trading_mode": args.trading_mode,
        "readonly": False,
    }
    report = report_builder(
        ticket_report=ticket_report,
        connection_preflight_report=preflight_report,
        execution_client_factory=_create_execution_client,
        ib_factory=lambda: _create_real_ibkr_paper_ib(
            config=config,
            allow_real_ibkr=args.allow_real_ibkr,
        ),
        config=config,
        operator_acknowledgements=args.ack,
        allow_real_ibkr=args.allow_real_ibkr,
        allow_real_paper_submit=args.allow_real_paper_submit,
    )

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    submitted = report.get("submission", {}).get("submitted") is True
    return 0 if submitted and not report.get("errors") else 1


def _create_real_ibkr_paper_ib(*, config: dict[str, Any], allow_real_ibkr: bool) -> Any:
    from algo_trader_unified.core.ibkr_paper_factory import create_real_ibkr_paper_ib
    from algo_trader_unified.core.ibkr_paper_order_mapper import validate_ibkr_paper_config

    return create_real_ibkr_paper_ib(
        config=validate_ibkr_paper_config(config),
        allow_real_ibkr=allow_real_ibkr,
    )


def _create_execution_client(ib: Any, validated_config: Any) -> Any:
    from algo_trader_unified.core.ibkr_paper_execution_client import (
        IbkrPaperExecutionClient,
    )

    return IbkrPaperExecutionClient(ib=ib, config=validated_config)


def main(argv: list[str] | None = None) -> int:
    return run_manual_real_paper_submit(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    gates = report.get("gates", {})
    ticket = report.get("ticket", {})
    submission = report.get("submission", {})
    cleanup = report.get("cleanup", {})
    lines = [
        "Manual real IBKR paper submit",
        "single_ticket: True",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"gates_passed: {gates.get('passed')}",
        f"client_order_id: {ticket.get('client_order_id')}",
        f"symbol: {ticket.get('symbol')}",
        f"action: {ticket.get('action')}",
        f"quantity: {ticket.get('quantity')}",
        f"attempted: {submission.get('attempted')}",
        f"submitted: {submission.get('submitted')}",
        f"broker_order_id: {submission.get('broker_order_id')}",
        f"reason: {submission.get('reason')}",
        f"disconnect_attempted: {cleanup.get('disconnect_attempted')}",
        f"disconnect_ok: {cleanup.get('disconnect_ok')}",
    ]
    if gates.get("reasons"):
        lines.append("gate_reasons:")
        for reason in gates["reasons"]:
            lines.append(f"  - {reason}")
    for key in ("errors", "warnings"):
        values = report.get(key)
        if values:
            lines.append(f"{key}:")
            for value in values:
                lines.append(f"  - {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
