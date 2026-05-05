"""Dry-run daily digest built from local state, ledgers, and snapshots."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.ledger import KNOWN_EVENT_TYPES
from algo_trader_unified.core.ledger_reader import LedgerReader


NY_TZ = ZoneInfo("America/New_York")
UNKNOWN_SKIP_REASON = "UNKNOWN_SKIP_REASON"
UNKNOWN_STRATEGY = "UNKNOWN_STRATEGY"

SIGNAL_EVENT_TYPES = {"SIGNAL_GENERATED", "SIGNAL_SKIPPED"} & KNOWN_EVENT_TYPES
INTENT_EVENT_TYPES = {
    "ORDER_INTENT_CREATED",
    "ORDER_INTENT_EXPIRED",
    "ORDER_INTENT_CANCELLED",
    "ORDER_SUBMITTED",
} & KNOWN_EVENT_TYPES
LIFECYCLE_EVENT_TYPES = {
    "ORDER_INTENT_CREATED",
    "ORDER_INTENT_EXPIRED",
    "ORDER_INTENT_CANCELLED",
    "ORDER_SUBMITTED",
    "FILL_CONFIRMED",
    "POSITION_CLOSED",
} & KNOWN_EVENT_TYPES
RECONCILIATION_EVENT_TYPE = "RECONCILIATION_FAILED"
_ACTIVE_INTENT_STATUSES = {"created", "submitted", "confirmed", "filled"}
_SNAPSHOT_TIMESTAMP_FIELDS = ("timestamp", "captured_at", "snapshot_at", "generated_at")


@dataclass(frozen=True)
class DigestContent:
    session_date: date
    strategy_ids: tuple[str, ...]
    account_summary: dict[str, Any]
    strategy_summary: dict[str, dict[str, int]]
    signals_generated_count: int
    signals_skipped_by_reason: dict[str, int]
    intent_counts: dict[str, int]
    active_intents: int
    lifecycle_status_counts: dict[str, int]
    ledger_consistency: dict[str, int]
    session_health: dict[str, Any]
    reconciliation_failed_events: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""


def build_digest_content(
    events: list[dict],
    state_snapshot: dict,
    session_date: date,
    strategy_ids: list[str],
) -> DigestContent:
    strategy_keys = tuple(strategy_ids)
    strategy_summary = {
        strategy_id: {
            "signals_generated": 0,
            "signals_skipped": 0,
            "intents_created": 0,
            "active_intents": int(
                state_snapshot.get("active_intents_by_strategy", {}).get(strategy_id, 0)
            ),
            "position_closed": 0,
        }
        for strategy_id in strategy_keys
    }
    strategy_summary[UNKNOWN_STRATEGY] = {
        "signals_generated": 0,
        "signals_skipped": 0,
        "intents_created": 0,
        "active_intents": int(
            state_snapshot.get("active_intents_by_strategy", {}).get(UNKNOWN_STRATEGY, 0)
        ),
        "position_closed": 0,
    }

    signals_generated = 0
    signals_skipped_by_reason: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    lifecycle_counts: Counter[str] = Counter()
    reconciliation_failed: list[dict[str, Any]] = []
    malformed_timestamp_count = 0
    unknown_event_type_count = 0
    missing_required_fields_count = 0
    event_ids: list[str] = []

    for event in events:
        event_type = event.get("event_type")
        if event_type not in KNOWN_EVENT_TYPES:
            unknown_event_type_count += 1
        for field_name in ("event_id", "event_type", "timestamp"):
            if not event.get(field_name):
                missing_required_fields_count += 1
        event_id = event.get("event_id")
        if isinstance(event_id, str) and event_id:
            event_ids.append(event_id)

        event_date = _event_session_date(event)
        if event_date is None:
            malformed_timestamp_count += 1
            continue
        if event_date != session_date:
            continue
        if event_type not in KNOWN_EVENT_TYPES:
            continue

        strategy_id = _strategy_id(event)
        strategy_bucket = strategy_id if strategy_id in strategy_summary else UNKNOWN_STRATEGY

        if event_type == "SIGNAL_GENERATED":
            signals_generated += 1
            strategy_summary[strategy_bucket]["signals_generated"] += 1
        elif event_type == "SIGNAL_SKIPPED":
            reason = event.get("payload", {}).get("skip_reason") if isinstance(event.get("payload"), dict) else None
            signals_skipped_by_reason[str(reason or UNKNOWN_SKIP_REASON)] += 1
            strategy_summary[strategy_bucket]["signals_skipped"] += 1

        if event_type in INTENT_EVENT_TYPES:
            intent_counts[event_type] += 1
            if event_type == "ORDER_INTENT_CREATED":
                strategy_summary[strategy_bucket]["intents_created"] += 1
        if event_type in LIFECYCLE_EVENT_TYPES:
            lifecycle_counts[event_type] += 1
        if event_type == "POSITION_CLOSED":
            strategy_summary[strategy_bucket]["position_closed"] += 1
        if event_type == RECONCILIATION_EVENT_TYPE:
            reconciliation_failed.append(event)

    duplicate_event_id_count = sum(
        count - 1 for count in Counter(event_ids).values() if count > 1
    )
    active_intents = int(state_snapshot.get("active_intents", 0))
    account_summary = {
        "open_positions": int(state_snapshot.get("open_positions", 0)),
        "active_intents": active_intents,
        "total_positions": int(state_snapshot.get("total_positions", 0)),
    }
    ledger_consistency = {
        "malformed_timestamp_count": malformed_timestamp_count,
        "unknown_event_type_count": unknown_event_type_count,
        "duplicate_event_id_count": duplicate_event_id_count,
        "missing_required_fields_count": missing_required_fields_count,
    }
    session_health = {
        "halt_state": state_snapshot.get("halt_state", "inactive"),
        "active_halt_conditions": int(state_snapshot.get("active_halt_conditions", 0)),
        "account_snapshot_fresh": bool(state_snapshot.get("account_snapshot_fresh", False)),
        "nlv_valid": bool(state_snapshot.get("nlv_valid", False)),
        "generated_at": state_snapshot.get("generated_at"),
    }
    content = DigestContent(
        session_date=session_date,
        strategy_ids=strategy_keys,
        account_summary=account_summary,
        strategy_summary=strategy_summary,
        signals_generated_count=signals_generated,
        signals_skipped_by_reason=dict(sorted(signals_skipped_by_reason.items())),
        intent_counts={key: intent_counts.get(key, 0) for key in sorted(INTENT_EVENT_TYPES)},
        active_intents=active_intents,
        lifecycle_status_counts={key: lifecycle_counts.get(key, 0) for key in sorted(LIFECYCLE_EVENT_TYPES)},
        ledger_consistency=ledger_consistency,
        session_health=session_health,
        reconciliation_failed_events=reconciliation_failed,
    )
    return DigestContent(**{**content.__dict__, "text": render_digest_text(content)})


def write_digest(
    content: DigestContent,
    snapshots_dir: Path,
    telegram_sender,
    now: datetime,
) -> None:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    session_date = _to_ny(now).date()
    path = snapshots_dir / f"digest_{session_date.isoformat()}.txt"
    path.write_text(content.text, encoding="utf-8")
    print(content.text)
    if telegram_sender is None:
        print("WARNING: Telegram sender unavailable", file=sys.stderr)
        return
    try:
        telegram_sender(content.text)
    except Exception as exc:
        print(f"WARNING: Telegram digest send failed: {exc}", file=sys.stderr)


def run_daily_digest(
    *,
    state_store,
    snapshots_dir: str | Path,
    halt_state_path: str | Path,
    ledger_reader: LedgerReader | None = None,
    ledger=None,
    telegram_sender=None,
    now: datetime | None = None,
    strategy_ids: list[str] | None = None,
) -> DigestContent:
    current = now or datetime.now(NY_TZ)
    strategy_ids = strategy_ids or [S01_VOL_BASELINE, S02_VOL_ENHANCED]
    if ledger_reader is None:
        if ledger is None or not hasattr(ledger, "root_dir"):
            events: list[dict] = []
        else:
            ledger_reader = LedgerReader.from_root(ledger.root_dir)
            events = ledger_reader.read_events()
    else:
        events = ledger_reader.read_events()
    snapshots_path = Path(snapshots_dir)
    state_snapshot = build_state_snapshot(
        state_store=state_store,
        snapshots_dir=snapshots_path,
        halt_state_path=Path(halt_state_path),
        now=current,
        strategy_ids=strategy_ids,
    )
    content = build_digest_content(
        events=events,
        state_snapshot=state_snapshot,
        session_date=_to_ny(current).date(),
        strategy_ids=strategy_ids,
    )
    write_digest(
        content=content,
        snapshots_dir=snapshots_path,
        telegram_sender=telegram_sender,
        now=current,
    )
    return content


def build_state_snapshot(
    *,
    state_store,
    snapshots_dir: Path,
    halt_state_path: Path,
    now: datetime,
    strategy_ids: list[str],
) -> dict[str, Any]:
    positions = _safe_call(state_store, "list_positions")
    order_intents = _safe_call(state_store, "list_order_intents")
    close_intents = _safe_call(state_store, "list_close_intents")
    active_intents_by_strategy = {strategy_id: 0 for strategy_id in strategy_ids}
    active_intents_by_strategy[UNKNOWN_STRATEGY] = 0
    active_intents = 0
    for intent in order_intents + close_intents:
        if not isinstance(intent, dict) or intent.get("status") not in _ACTIVE_INTENT_STATUSES:
            continue
        active_intents += 1
        strategy_id = intent.get("strategy_id")
        key = strategy_id if strategy_id in active_intents_by_strategy else UNKNOWN_STRATEGY
        active_intents_by_strategy[key] += 1
    halt_state = _halt_state_summary(halt_state_path)
    snapshot_fresh = _latest_snapshot_fresh(snapshots_dir, now)
    return {
        "generated_at": _to_ny(now).isoformat(),
        "open_positions": sum(1 for position in positions if isinstance(position, dict) and position.get("status") == "open"),
        "total_positions": len(positions),
        "active_intents": active_intents,
        "active_intents_by_strategy": active_intents_by_strategy,
        "halt_state": halt_state,
        "active_halt_conditions": 0 if halt_state == "inactive" else 1,
        "account_snapshot_fresh": snapshot_fresh,
        "nlv_valid": snapshot_fresh,
    }


def render_digest_text(content: DigestContent) -> str:
    lines = [
        f"Dry-run daily digest: {content.session_date.isoformat()}",
        "",
        "Account summary",
        f"- Open positions: {content.account_summary['open_positions']}",
        f"- Active intents: {content.account_summary['active_intents']}",
        f"- Total positions: {content.account_summary['total_positions']}",
        "",
        "Strategy summary",
    ]
    for strategy_id in (*content.strategy_ids, UNKNOWN_STRATEGY):
        summary = content.strategy_summary[strategy_id]
        lines.extend(
            [
                f"- {strategy_id}: signals_generated={summary['signals_generated']}, "
                f"signals_skipped={summary['signals_skipped']}, "
                f"intents_created={summary['intents_created']}, "
                f"active_intents={summary['active_intents']}, "
                f"position_closed={summary['position_closed']}",
            ]
        )
    lines.extend(
        [
            "",
            "Signals",
            f"- Generated: {content.signals_generated_count}",
            "- Skipped by reason: "
            + _format_counts(content.signals_skipped_by_reason),
            "",
            "Intents",
            "- Today: " + _format_counts(content.intent_counts),
            f"- Active at session end: {content.active_intents}",
            "",
            "Lifecycle status counts",
            "- " + _format_counts(content.lifecycle_status_counts),
            "",
            "Ledger consistency",
            f"- Malformed/unknown timestamps: {content.ledger_consistency['malformed_timestamp_count']}",
            f"- Unknown/missing event types: {content.ledger_consistency['unknown_event_type_count']}",
            f"- Duplicate event ids: {content.ledger_consistency['duplicate_event_id_count']}",
            f"- Missing required fields: {content.ledger_consistency['missing_required_fields_count']}",
            "",
            "Session health",
            f"- Halt: {content.session_health['halt_state']}",
            f"- Active halt conditions: {content.session_health['active_halt_conditions']}",
            f"- Account snapshot fresh: {content.session_health['account_snapshot_fresh']}",
            f"- NLV valid: {content.session_health['nlv_valid']}",
            f"- Generated at: {content.session_health['generated_at']}",
            "",
            "RECONCILIATION_FAILED",
            f"- Count: {len(content.reconciliation_failed_events)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _event_session_date(event: dict[str, Any]) -> date | None:
    timestamp = event.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _to_ny(parsed).date()


def _strategy_id(event: dict[str, Any]) -> str:
    strategy_id = event.get("strategy_id")
    if isinstance(strategy_id, str) and strategy_id:
        return strategy_id
    payload = event.get("payload")
    if isinstance(payload, dict):
        payload_strategy_id = payload.get("strategy_id")
        if isinstance(payload_strategy_id, str) and payload_strategy_id:
            return payload_strategy_id
    return UNKNOWN_STRATEGY


def _to_ny(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=NY_TZ)
    return value.astimezone(NY_TZ)


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _safe_call(state_store, name: str) -> list[dict]:
    method = getattr(state_store, name, None)
    if not callable(method):
        return []
    value = method()
    return value if isinstance(value, list) else []


def _halt_state_summary(halt_state_path: Path) -> str:
    try:
        if not halt_state_path.exists():
            return "inactive"
        payload = json.loads(halt_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable"
    if not isinstance(payload, dict):
        return "unreadable"
    if payload.get("resumed") is True or payload.get("tier") not in {"soft", "hard"}:
        return "inactive"
    scope = payload.get("scope") or "unknown"
    tier = payload.get("tier")
    halted_id = payload.get("id") or payload.get("scope_id")
    if halted_id:
        return f"active:{scope}:{halted_id}:{tier}"
    return f"active:{scope}:{tier}"


def _latest_snapshot_fresh(snapshots_dir: Path, now: datetime) -> bool:
    try:
        snapshots = [
            path
            for path in snapshots_dir.iterdir()
            if path.is_file() and path.suffix == ".json"
        ]
    except OSError:
        return False
    if not snapshots:
        return False
    latest = max(snapshots, key=lambda path: path.stat().st_mtime)
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    timestamp = _snapshot_timestamp(payload)
    if timestamp is None:
        timestamp = datetime.fromtimestamp(latest.stat().st_mtime, NY_TZ)
    age_seconds = (_to_ny(now) - _to_ny(timestamp)).total_seconds()
    return age_seconds <= 15 * 60


def _snapshot_timestamp(payload: dict[str, Any]) -> datetime | None:
    for field_name in _SNAPSHOT_TIMESTAMP_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
    return None
