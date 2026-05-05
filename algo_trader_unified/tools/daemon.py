"""Dry-run daemon startup gate and shutdown safety."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from algo_trader_unified.config.scheduler import SCHEDULER_SHUTDOWN_TIMEOUT_SEC
from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.readiness import ReadinessManager
from algo_trader_unified.core.readiness_provider import DefaultHealthSnapshotProvider
from algo_trader_unified.core.scheduler import UnifiedScheduler
from algo_trader_unified.core.scheduler_cadence import (
    run_bounded_dry_run_smoke,
    run_bounded_foreground_scheduler,
)
from algo_trader_unified.core.state_store import (
    CURRENT_SCHEMA_VERSION,
    StateStore,
    StateStoreCorruptError,
)


@dataclass(frozen=True)
class DiagnosticResult:
    passed: bool
    reason: str | None = None


HaltState = dict[str, Any] | None


class DiagnosticProvider(Protocol):
    def run(self) -> DiagnosticResult:
        ...


class SchedulerLike(Protocol):
    def start(self) -> None:
        ...

    def shutdown(self, wait: bool = True) -> None:
        ...


class DiagnosticProviderFactory(Protocol):
    def __call__(self, state_store: Any, halt_state: HaltState) -> DiagnosticProvider:
        ...


class SchedulerFactory(Protocol):
    def __call__(self, *, state_store: Any, ledger: Any) -> SchedulerLike:
        ...


class ReadinessProviderFactory(Protocol):
    def __call__(
        self,
        *,
        state_store: Any,
        snapshots_dir: Path,
        halt_state_path: Path,
    ) -> Callable[[], Any]:
        ...


class SmokeRunner(Protocol):
    def __call__(
        self,
        *,
        state_store: Any,
        ledger: Any,
        readiness_provider: Callable[[], Any],
        snapshots_dir: Path,
        halt_state_path: Path,
        cycles: int,
        include_lifecycle_pipeline: bool,
    ) -> dict[str, Any]:
        ...


class ForegroundRunner(Protocol):
    def __call__(
        self,
        *,
        state_store: Any,
        ledger: Any,
        readiness_provider: Callable[[], Any],
        snapshots_dir: Path,
        halt_state_path: Path,
        runtime_seconds: float,
        enable_triggers: bool,
        include_lifecycle_pipeline: bool,
    ) -> dict[str, Any]:
        ...


class StartupDiagnosticProvider:
    """Local-only startup checks for dry-run daemon mode."""

    def __init__(self, state_store: Any, halt_state: HaltState) -> None:
        self.state_store = state_store
        self.halt_state = halt_state

    def run(self) -> DiagnosticResult:
        state = getattr(self.state_store, "state", None)
        if not isinstance(state, dict):
            return DiagnosticResult(False, "StateStore is not loadable")
        found = state.get("schema_version")
        if found != CURRENT_SCHEMA_VERSION:
            return DiagnosticResult(
                False,
                f"StateStore schema_version mismatch: found {found!r}, expected {CURRENT_SCHEMA_VERSION!r}",
            )
        state_halt = state.get("halt_state")
        if _halt_is_active(self.halt_state) != _halt_is_active(state_halt):
            return DiagnosticResult(
                False,
                "halt_state.json conflicts with StateStore halt_state",
            )
        if _has_needs_reconciliation(state):
            return DiagnosticResult(False, "NEEDS_RECONCILIATION records exist")
        return DiagnosticResult(True)


def load_env() -> None:
    from algo_trader_unified.config.env import get_required_env
    from algo_trader_unified.tools.validate_env import DEFAULT_REQUIRED_VARS

    for name in DEFAULT_REQUIRED_VARS:
        get_required_env(name)


def load_config() -> None:
    from algo_trader_unified.tools.validate_config import validate

    errors = validate()
    if errors:
        raise ValueError("; ".join(errors))


def load_halt_state(root_dir: str | Path = ".") -> HaltState:
    path = Path(root_dir) / "data" / "state" / "halt_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_readiness_provider(
    *,
    state_store: Any,
    snapshots_dir: Path,
    halt_state_path: Path,
) -> DefaultHealthSnapshotProvider:
    return DefaultHealthSnapshotProvider(
        state_store=state_store,
        halt_state_path=halt_state_path,
        snapshots_dir=snapshots_dir,
        max_staleness_minutes=15,
        strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
    )


def build_scheduler(
    *,
    state_store: Any,
    ledger: Any,
    enable_triggers: bool = True,
) -> UnifiedScheduler:
    scheduler = UnifiedScheduler(
        state_store=state_store,
        ledger=ledger,
        readiness_manager=ReadinessManager(state_store, ledger),
    )
    if enable_triggers:
        scheduler.build_scheduler()
    return scheduler


def run_daemon(
    argv: list[str] | tuple[str, ...],
    *,
    env_loader: Callable[[], None],
    config_loader: Callable[[], None],
    ledger_factory: Callable[[Path], Any],
    state_store_factory: Callable[[Path], Any],
    halt_loader: Callable[[Path], HaltState],
    diagnostic_provider: DiagnosticProviderFactory,
    scheduler_factory: SchedulerFactory,
    readiness_provider_factory: ReadinessProviderFactory | None = None,
    smoke_runner: SmokeRunner = run_bounded_dry_run_smoke,
    foreground_runner: ForegroundRunner = run_bounded_foreground_scheduler,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--root", default=".")
    parser.add_argument("--smoke-cycles", type=int)
    parser.add_argument("--foreground-runtime-seconds", type=float)
    parser.add_argument("--enable-triggers", action="store_true")
    parser.add_argument("--enable-lifecycle-pipeline", action="store_true")
    args = parser.parse_args(argv)

    if not args.dry_run_only:
        print("ERROR: daemon mode requires --dry-run-only", file=sys.stderr)
        return 1
    if args.smoke_cycles is not None and args.foreground_runtime_seconds is not None:
        print(
            "ERROR: choose either --smoke-cycles or --foreground-runtime-seconds",
            file=sys.stderr,
        )
        return 1
    if args.smoke_cycles is not None and args.smoke_cycles <= 0:
        print("ERROR: --smoke-cycles must be a positive finite integer", file=sys.stderr)
        return 1
    if (
        args.foreground_runtime_seconds is not None
        and (
            args.foreground_runtime_seconds <= 0
            or not math.isfinite(args.foreground_runtime_seconds)
        )
    ):
        print(
            "ERROR: --foreground-runtime-seconds must be a positive finite number",
            file=sys.stderr,
        )
        return 1

    root = Path(args.root)
    snapshots_dir = root / "data" / "snapshots"
    halt_state_path = root / "data" / "state" / "halt_state.json"
    try:
        env_loader()
        config_loader()
        ledger = ledger_factory(root)
        state_store = state_store_factory(root / "data" / "state" / "portfolio_state.json")
    except StateStoreCorruptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    halt_state = halt_loader(root)
    if _halt_is_active(halt_state):
        _append_startup_blocked(ledger, halt_state)
        print("ERROR: startup blocked by active halt_state.json", file=sys.stderr)
        return 1

    provider = diagnostic_provider(state_store, halt_state)
    result = provider.run()
    if not result.passed:
        reason = result.reason or "startup diagnostics failed"
        print(f"ERROR: startup diagnostics failed: {reason}", file=sys.stderr)
        return 1

    if args.smoke_cycles is not None:
        provider_factory = readiness_provider_factory or build_readiness_provider
        readiness_provider = provider_factory(
            state_store=state_store,
            snapshots_dir=snapshots_dir,
            halt_state_path=halt_state_path,
        )
        summary = smoke_runner(
            state_store=state_store,
            ledger=ledger,
            readiness_provider=readiness_provider,
            snapshots_dir=snapshots_dir,
            halt_state_path=halt_state_path,
            cycles=args.smoke_cycles,
            include_lifecycle_pipeline=args.enable_lifecycle_pipeline,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary.get("success") is True else 1

    if args.foreground_runtime_seconds is not None:
        provider_factory = readiness_provider_factory or build_readiness_provider
        readiness_provider = provider_factory(
            state_store=state_store,
            snapshots_dir=snapshots_dir,
            halt_state_path=halt_state_path,
        )
        summary = foreground_runner(
            state_store=state_store,
            ledger=ledger,
            readiness_provider=readiness_provider,
            snapshots_dir=snapshots_dir,
            halt_state_path=halt_state_path,
            runtime_seconds=args.foreground_runtime_seconds,
            enable_triggers=args.enable_triggers,
            include_lifecycle_pipeline=args.enable_lifecycle_pipeline,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary.get("success") is True else 1

    scheduler = scheduler_factory(state_store=state_store, ledger=ledger)
    scheduler.start()
    _install_shutdown_handlers(scheduler)
    wait_until_stopped = getattr(scheduler, "wait_until_stopped", None)
    if callable(wait_until_stopped):
        wait_until_stopped()
        return 0
    while True:
        signal.pause()


def main(argv: list[str] | None = None) -> int:
    return run_daemon(
        sys.argv[1:] if argv is None else argv,
        env_loader=load_env,
        config_loader=load_config,
        ledger_factory=LedgerAppender,
        state_store_factory=StateStore,
        halt_loader=load_halt_state,
        diagnostic_provider=StartupDiagnosticProvider,
        scheduler_factory=build_scheduler,
        readiness_provider_factory=build_readiness_provider,
    )


def _install_shutdown_handlers(scheduler: Any) -> None:
    def handler(signum, frame):
        raise SystemExit(_shutdown_scheduler(scheduler))

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _shutdown_scheduler(
    scheduler: Any,
    *,
    timeout_sec: float = SCHEDULER_SHUTDOWN_TIMEOUT_SEC,
    thread_factory: Callable[..., threading.Thread] = threading.Thread,
) -> int:
    thread = thread_factory(
        target=scheduler.shutdown,
        kwargs={"wait": True},
        daemon=True,
    )
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        print(
            f"WARNING: scheduler shutdown exceeded {timeout_sec} seconds",
            file=sys.stderr,
        )
        return 1
    return 0


def _halt_is_active(halt_state: Any) -> bool:
    if not isinstance(halt_state, dict):
        return False
    if halt_state.get("resumed") is True:
        return False
    return halt_state.get("tier") in {"soft", "hard"}


def _append_startup_blocked(ledger: Any, halt_state: dict[str, Any]) -> None:
    payload = {
        "event_detail": "STARTUP_BLOCKED_HALT_ACTIVE",
        "scope": "startup_gate",
        "reason": "halt_state_active_at_startup",
        "halt_scope": halt_state.get("scope"),
        "halt_id": halt_state.get("id"),
        "halt_tier": halt_state.get("tier"),
        "halt_reason": halt_state.get("reason"),
        "halt_event_id": halt_state.get("halt_event_id"),
    }
    ledger.append(
        event_type="MANUAL_STATUS_UPDATED",
        strategy_id=halt_state.get("id") if halt_state.get("scope") == "strategy" else "ACCOUNT",
        execution_mode="disabled",
        source_module="tools.daemon",
        payload=payload,
    )


def _has_needs_reconciliation(state: dict[str, Any]) -> bool:
    for key in ("positions", "order_intents", "close_intents"):
        collection = state.get(key, {})
        if isinstance(collection, dict):
            records = collection.values()
        elif isinstance(collection, list):
            records = collection
        else:
            continue
        for record in records:
            if isinstance(record, dict) and record.get("status") == "NEEDS_RECONCILIATION":
                return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
