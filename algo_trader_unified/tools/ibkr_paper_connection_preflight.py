"""Run the Stage 4F-2 real IBKR paper connection preflight."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from algo_trader_unified.core.ibkr_paper_connection_preflight import (
    build_ibkr_paper_connection_preflight_report,
)


def run_ibkr_paper_connection_preflight(
    argv: list[str] | tuple[str, ...],
    *,
    report_builder: Callable[..., dict[str, Any]] = build_ibkr_paper_connection_preflight_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--allow-real-ibkr", action="store_true")
    parser.add_argument("--connect-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4004)
    parser.add_argument("--client-id", type=int, default=7)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--trading-mode", default="PAPER")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: IBKR paper connection preflight requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    if args.connect_timeout_seconds <= 0:
        print(
            "ERROR: --connect-timeout-seconds must be positive",
            file=sys.stderr,
        )
        return 1

    config = {
        "host": args.host,
        "port": args.port,
        "client_id": args.client_id,
        "account_id": args.account_id,
        "trading_mode": args.trading_mode,
        "readonly": True,
    }

    report = report_builder(
        config=config,
        ib_factory=lambda: _create_real_ibkr_paper_ib(
            config=config,
            allow_real_ibkr=args.allow_real_ibkr,
        ),
        client_factory=_create_readonly_client,
        allow_real_ibkr=args.allow_real_ibkr,
        connect_timeout_seconds=args.connect_timeout_seconds,
    )

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    return 0 if report.get("success") is True and not report.get("errors") else 1


def _create_real_ibkr_paper_ib(*, config: dict[str, Any], allow_real_ibkr: bool) -> Any:
    from algo_trader_unified.core.ibkr_paper_factory import create_real_ibkr_paper_ib
    from algo_trader_unified.core.ibkr_paper_order_mapper import (
        validate_ibkr_paper_readonly_config,
    )

    return create_real_ibkr_paper_ib(
        config=validate_ibkr_paper_readonly_config(config),
        allow_real_ibkr=allow_real_ibkr,
    )


def _create_readonly_client(ib: Any, validated_config: Any) -> Any:
    from algo_trader_unified.core.ibkr_paper_client import IbkrPaperReadOnlyClient

    return IbkrPaperReadOnlyClient(ib=ib, config=validated_config)


def main(argv: list[str] | None = None) -> int:
    return run_ibkr_paper_connection_preflight(sys.argv[1:] if argv is None else argv)


def _format_human(report: dict[str, Any]) -> str:
    config = report.get("config", {})
    connection = report.get("connection", {})
    readiness = report.get("readiness_for_stage4f3", {})
    lines = [
        "IBKR paper connection preflight",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"trading_mode: {config.get('trading_mode')}",
        f"host: {config.get('host')}",
        f"port: {config.get('port')}",
        f"client_id: {config.get('client_id')}",
        f"account_id: {config.get('account_id')}",
        f"paper_config_valid: {config.get('paper_config_valid')}",
        f"allow_real_ibkr: {connection.get('allow_real_ibkr')}",
        f"attempted: {connection.get('attempted')}",
        f"connected: {connection.get('connected')}",
        f"disconnected: {connection.get('disconnected')}",
    ]
    readiness_key = "ready_to_build_manual_real_paper_submit_command"
    if readiness_key in readiness:
        lines.append(f"{readiness_key}: {readiness.get(readiness_key)}")
    for key in ("errors", "warnings"):
        values = report.get(key)
        if values:
            lines.append(f"{key}:")
            for value in values:
                lines.append(f"  - {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
