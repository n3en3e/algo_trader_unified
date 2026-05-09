"""Stage 4F-1 real IBKR paper factory preflight.

This module describes a deliberately gated real-paper factory path. Importing
it must not import IBKR runtime libraries, connect to IB Gateway, submit orders,
cancel orders, request market data, or qualify contracts.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    IbkrPaperConfig,
    validate_ibkr_paper_readonly_config,
)
from algo_trader_unified.core.paper_broker_adapter import sanitize_json_safe


if TYPE_CHECKING:
    from ib_insync import IB as IbInsyncIB


ORDERED_NEXT_STEPS = [
    "Keep the factory unwired and add a manual real-paper submit command only in Stage 4F-2.",
    "Continue using injected probes and fake IB modules in tests.",
    "Require explicit operator gates before any future real paper submission path.",
]
DO_NOT_DO_YET = [
    "Do not connect to IB Gateway.",
    "Do not submit or cancel paper orders.",
    "Do not request market data or qualify contracts.",
    "Do not wire this factory into scheduler, daemon, or lifecycle jobs.",
    "Do not add any live order override.",
]


def build_ibkr_paper_factory_preflight_report(
    *,
    config: dict[str, Any] | IbkrPaperConfig,
    import_probe: Callable[[str], Any] | None = None,
    now_provider: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe Stage 4F-1 real-paper factory preflight report.

    The preflight accepts injected config and an optional import probe. It does
    not read global config, instantiate IB, connect, mutate state, or write
    files.
    """

    generated_at = _iso_now(now_provider)
    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []

    raw_config = _config_input_dict(config)
    validated_config: IbkrPaperConfig | None = None
    validation_reason = "valid"
    try:
        validated_config = validate_ibkr_paper_factory_config(raw_config)
        config_payload = _config_payload(validated_config, True, validation_reason)
    except Exception as exc:  # noqa: BLE001
        validation_reason = _exception_reason(exc)
        errors.append(f"validate_ibkr_paper_factory_config failed: {validation_reason}")
        blockers.append("IBKR paper factory config is not valid.")
        config_payload = _config_payload(raw_config, False, validation_reason)

    import_available = False
    import_probe_error: str | None = None
    try:
        probe = import_probe or importlib.util.find_spec
        import_available = probe("ib_insync") is not None
    except Exception as exc:  # noqa: BLE001
        import_probe_error = _exception_reason(exc)
        warnings.append(f"ib_insync availability probe failed: {import_probe_error}")

    factory = {
        "factory_present": True,
        "allow_real_ibkr_default": False,
        "would_import_ib_insync": False,
        "would_connect": False,
        "would_submit_orders": False,
        "would_cancel_orders": False,
        "would_request_market_data": False,
        "would_qualify_contracts": False,
        "ib_insync_available": import_available,
        "ib_insync_probe_error": import_probe_error,
    }
    import_safety = {
        "ib_insync_required_at_module_import": False,
        "type_checking_only_imports": True,
        "runtime_import_gated": True,
        "preflight_uses_find_spec_or_injected_probe": True,
    }
    safety = {
        "real_ibkr_enabled": False,
        "paper_order_submission_enabled": False,
        "cancel_enabled": False,
        "live_orders_enabled": False,
        "market_data_enabled": False,
        "contract_qualification_enabled": False,
        "scheduler_changes_enabled": False,
        "lifecycle_wiring_enabled": False,
    }

    ready = _ready_for_stage4f2(
        config_valid=validated_config is not None,
        import_safety=import_safety,
        factory=factory,
        safety=safety,
        blockers=blockers,
    )

    report = {
        "dry_run": True,
        "ibkr_paper_factory_preflight": True,
        "generated_at": generated_at,
        "config": config_payload,
        "factory": factory,
        "import_safety": import_safety,
        "safety": safety,
        "readiness_for_stage4f2": {
            "ready_to_build_manual_real_paper_submit_command": ready,
            "blockers": sorted(blockers),
            "warnings": sorted(warnings),
        },
        "recommendations": {
            "ordered_next_steps": list(ORDERED_NEXT_STEPS),
            "do_not_do_yet": list(DO_NOT_DO_YET),
        },
        "success": True,
        "errors": list(errors),
        "warnings": list(warnings),
    }
    safe = sanitize_json_safe(report)
    return safe if isinstance(safe, dict) else {"success": False, "errors": ["report serialization failed"]}


def validate_ibkr_paper_factory_config(config: dict[str, Any]) -> IbkrPaperConfig:
    """Validate the Stage 4F-1 factory config as read-only PAPER on port 4004."""

    return validate_ibkr_paper_readonly_config(config)


def create_real_ibkr_paper_ib(
    *,
    config: IbkrPaperConfig,
    allow_real_ibkr: bool = False,
) -> "IbInsyncIB":
    """Create an unconnected ib_insync.IB instance only behind explicit gates."""

    if allow_real_ibkr is not True:
        raise PermissionError("real IBKR paper factory requires allow_real_ibkr=True")

    validated_config = validate_ibkr_paper_factory_config(asdict(config))
    if validated_config.trading_mode != "PAPER" or validated_config.port != IBKR_PAPER_PORT:
        raise ValueError("real IBKR paper factory requires PAPER on port 4004")

    module = importlib.import_module("ib_insync")
    return module.IB()


def _ready_for_stage4f2(
    *,
    config_valid: bool,
    import_safety: dict[str, bool],
    factory: dict[str, Any],
    safety: dict[str, bool],
    blockers: list[str],
) -> bool:
    checks = [
        config_valid,
        import_safety["ib_insync_required_at_module_import"] is False,
        import_safety["preflight_uses_find_spec_or_injected_probe"] is True,
        import_safety["runtime_import_gated"] is True,
        factory["allow_real_ibkr_default"] is False,
        factory["would_connect"] is False,
        factory["would_submit_orders"] is False,
        factory["would_cancel_orders"] is False,
        factory["would_request_market_data"] is False,
        factory["would_qualify_contracts"] is False,
        safety["scheduler_changes_enabled"] is False,
        safety["lifecycle_wiring_enabled"] is False,
        safety["live_orders_enabled"] is False,
    ]
    if not all(checks) and "Stage 4F-1 safety invariants are incomplete." not in blockers:
        blockers.append("Stage 4F-1 safety invariants are incomplete.")
    return all(checks) and not blockers


def _config_input_dict(config: dict[str, Any] | IbkrPaperConfig) -> dict[str, Any]:
    if isinstance(config, IbkrPaperConfig):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise ValueError("config must be a dict or IbkrPaperConfig")


def _config_payload(
    config: dict[str, Any] | IbkrPaperConfig,
    paper_config_valid: bool,
    validation_reason: str,
) -> dict[str, Any]:
    payload = asdict(config) if isinstance(config, IbkrPaperConfig) else dict(config)
    return {
        "trading_mode": payload.get("trading_mode"),
        "host": payload.get("host"),
        "port": payload.get("port"),
        "client_id": payload.get("client_id"),
        "account_id": payload.get("account_id"),
        "readonly": payload.get("readonly"),
        "paper_config_valid": paper_config_valid,
        "validation_reason": validation_reason,
    }


def _iso_now(now_provider: Callable[[], datetime] | None) -> str:
    now = now_provider() if now_provider else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.isoformat()


def _exception_reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
