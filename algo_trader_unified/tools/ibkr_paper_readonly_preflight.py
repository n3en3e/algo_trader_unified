"""Run the Stage 4E-2 read-only IBKR paper operator preflight."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Any, Callable

from algo_trader_unified.core.ibkr_paper_client import IbkrPaperReadOnlyClient
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    validate_ibkr_paper_readonly_config,
)
from algo_trader_unified.core.ibkr_paper_readonly_preflight import (
    build_ibkr_paper_readonly_preflight_report,
)


def run_ibkr_paper_readonly_preflight(
    argv: list[str] | tuple[str, ...],
    *,
    client_factory: Callable[..., Any] | None = None,
    report_builder: Callable[..., dict[str, Any]] = build_ibkr_paper_readonly_preflight_report,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4004)
    parser.add_argument("--client-id", type=int, default=7)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--trading-mode", default="PAPER")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print(
            "ERROR: IBKR paper read-only preflight requires --dry-run-only",
            file=sys.stderr,
        )
        return 1

    try:
        config = validate_ibkr_paper_readonly_config(
            {
                "host": args.host,
                "port": args.port,
                "client_id": args.client_id,
                "account_id": args.account_id,
                "trading_mode": args.trading_mode,
                "readonly": True,
            }
        )
        factory = client_factory or _default_client_factory
        client = factory(config=config)
        report = report_builder(client=client)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_format_human(report))
    return 0 if report.get("success") is True else 1


def main(argv: list[str] | None = None) -> int:
    return run_ibkr_paper_readonly_preflight(sys.argv[1:] if argv is None else argv)


def _default_client_factory(*, config: Any) -> IbkrPaperReadOnlyClient:
    from ib_insync import IB

    client = IbkrPaperReadOnlyClient(ib=IB(), config=replace(config, readonly=False))
    client.config = config
    return client


def _format_human(report: dict[str, Any]) -> str:
    config = report.get("config")
    config = config if isinstance(config, dict) else {}
    connection = report.get("connection")
    connection = connection if isinstance(connection, dict) else {}
    readonly_checks = report.get("readonly_checks")
    readonly_checks = readonly_checks if isinstance(readonly_checks, dict) else {}
    recommendations = report.get("recommendations")
    recommendations = recommendations if isinstance(recommendations, dict) else {}
    open_orders = report.get("open_orders")
    open_orders = open_orders if isinstance(open_orders, dict) else {}
    positions = report.get("positions")
    positions = positions if isinstance(positions, dict) else {}
    lines = [
        "IBKR paper read-only preflight",
        f"success: {report.get('success')}",
        f"generated_at: {report.get('generated_at')}",
        f"trading_mode: {config.get('trading_mode')}",
        f"host: {config.get('host')}",
        f"port: {config.get('port')}",
        f"client_id: {config.get('client_id')}",
        f"account_id: {config.get('account_id')}",
        f"readonly: {config.get('readonly')}",
        f"connected: {connection.get('connected')}",
        f"paper_mode: {connection.get('paper_mode')}",
        f"current_time_ok: {readonly_checks.get('current_time_ok')}",
        f"account_snapshot_ok: {readonly_checks.get('account_snapshot_ok')}",
        f"open_orders_count: {open_orders.get('count')}",
        f"positions_count: {positions.get('count')}",
        f"disconnect_ok: {readonly_checks.get('disconnect_ok')}",
    ]
    reason = connection.get("reason")
    if reason:
        lines.append(f"connection_reason: {reason}")
    for key in ("errors", "warnings"):
        values = report.get(key)
        if values:
            lines.append(f"{key}:")
            for value in values:
                lines.append(f"  - {value}")
    steps = recommendations.get("ordered_next_steps")
    if steps:
        lines.append("ordered_next_steps:")
        for step in steps:
            lines.append(f"  - {step}")
    do_not = recommendations.get("do_not_do_yet")
    if do_not:
        lines.append("do_not_do_yet:")
        for item in do_not:
            lines.append(f"  - {item}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
