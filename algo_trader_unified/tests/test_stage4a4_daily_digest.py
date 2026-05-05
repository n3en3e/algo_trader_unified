from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import JOB_DAILY_DIGEST
from algo_trader_unified.config.scheduler import JOB_SPECS, SCHEDULER_TIMEZONE
from algo_trader_unified.core.ledger import KNOWN_EVENT_TYPES
from algo_trader_unified.jobs.daily_digest import (
    UNKNOWN_SKIP_REASON,
    UNKNOWN_STRATEGY,
    DigestContent,
    build_digest_content,
    build_state_snapshot,
    run_daily_digest,
    write_digest,
)


NY = ZoneInfo("America/New_York")


class FakeStateStore:
    def __init__(self) -> None:
        self.positions = [
            {"position_id": "pos1", "strategy_id": S01_VOL_BASELINE, "status": "open"},
            {"position_id": "pos2", "strategy_id": S02_VOL_ENHANCED, "status": "closed"},
        ]
        self.order_intents = [
            {"intent_id": "i1", "strategy_id": S01_VOL_BASELINE, "status": "created"},
            {"intent_id": "i2", "strategy_id": S02_VOL_ENHANCED, "status": "cancelled"},
        ]
        self.close_intents = [
            {"close_intent_id": "c1", "strategy_id": S02_VOL_ENHANCED, "status": "submitted"},
        ]
        self.saved = False

    def list_positions(self):
        return deepcopy(self.positions)

    def list_order_intents(self):
        return deepcopy(self.order_intents)

    def list_close_intents(self):
        return deepcopy(self.close_intents)

    def save(self):
        self.saved = True
        raise AssertionError("digest must not save StateStore")


class FakeLedgerReader:
    def __init__(self, events):
        self.events = events

    def read_events(self):
        return deepcopy(self.events)


def event(event_type, timestamp, strategy_id=S01_VOL_BASELINE, payload=None, event_id=None):
    return {
        "event_id": event_id or f"evt_{event_type}_{timestamp}",
        "event_type": event_type,
        "timestamp": timestamp,
        "strategy_id": strategy_id,
        "execution_mode": "paper_only",
        "source_module": "test",
        "position_id": None,
        "opportunity_id": None,
        "payload": payload or {},
    }


def base_state_snapshot(**overrides):
    snapshot = {
        "generated_at": "2026-05-05T17:00:00-04:00",
        "open_positions": 1,
        "total_positions": 2,
        "active_intents": 2,
        "active_intents_by_strategy": {
            S01_VOL_BASELINE: 1,
            S02_VOL_ENHANCED: 1,
            UNKNOWN_STRATEGY: 0,
        },
        "halt_state": "inactive",
        "active_halt_conditions": 0,
        "account_snapshot_fresh": True,
        "nlv_valid": True,
    }
    snapshot.update(overrides)
    return snapshot


class DigestContentTests(unittest.TestCase):
    def test_builder_is_pure_and_counts_expected_event_categories(self) -> None:
        session = date(2026, 5, 5)
        events = [
            event("SIGNAL_GENERATED", "2026-05-05T13:40:00+00:00", S01_VOL_BASELINE, event_id="evt1"),
            event("SIGNAL_SKIPPED", "2026-05-05T14:00:00+00:00", S01_VOL_BASELINE, {"skip_reason": "SKIP_X"}, "evt2"),
            event("SIGNAL_SKIPPED", "2026-05-05T14:01:00+00:00", "", {}, "evt3"),
            event("ORDER_INTENT_CREATED", "2026-05-05T15:00:00+00:00", S02_VOL_ENHANCED, event_id="evt4"),
            event("ORDER_INTENT_EXPIRED", "2026-05-05T15:10:00+00:00", S02_VOL_ENHANCED, event_id="evt5"),
            event("ORDER_INTENT_CANCELLED", "2026-05-05T15:15:00+00:00", S02_VOL_ENHANCED, event_id="evt6"),
            event("ORDER_SUBMITTED", "2026-05-05T15:20:00+00:00", S02_VOL_ENHANCED, event_id="evt7"),
            event("FILL_CONFIRMED", "2026-05-05T15:25:00+00:00", S02_VOL_ENHANCED, event_id="evt8"),
            event("POSITION_CLOSED", "2026-05-05T15:30:00+00:00", S02_VOL_ENHANCED, event_id="evt9"),
            event("RECONCILIATION_FAILED", "2026-05-05T15:35:00+00:00", S02_VOL_ENHANCED, event_id="evt10"),
        ]
        before_events = deepcopy(events)
        state = base_state_snapshot()
        before_state = deepcopy(state)
        content = build_digest_content(
            events=events,
            state_snapshot=state,
            session_date=session,
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )
        self.assertIsInstance(content, DigestContent)
        self.assertEqual(events, before_events)
        self.assertEqual(state, before_state)
        self.assertEqual(content.signals_generated_count, 1)
        self.assertEqual(content.signals_skipped_by_reason["SKIP_X"], 1)
        self.assertEqual(content.signals_skipped_by_reason[UNKNOWN_SKIP_REASON], 1)
        self.assertEqual(content.intent_counts["ORDER_INTENT_CREATED"], 1)
        self.assertEqual(content.intent_counts["ORDER_INTENT_EXPIRED"], 1)
        self.assertEqual(content.intent_counts["ORDER_INTENT_CANCELLED"], 1)
        self.assertEqual(content.intent_counts["ORDER_SUBMITTED"], 1)
        self.assertEqual(content.lifecycle_status_counts["FILL_CONFIRMED"], 1)
        self.assertEqual(content.lifecycle_status_counts["POSITION_CLOSED"], 1)
        self.assertEqual(len(content.reconciliation_failed_events), 1)
        self.assertIn("Account summary", content.text)
        self.assertIn("Strategy summary", content.text)

    def test_session_date_is_injected_not_derived_from_old_events(self) -> None:
        content = build_digest_content(
            events=[
                event("SIGNAL_GENERATED", "2026-05-01T15:00:00+00:00", event_id="old"),
            ],
            state_snapshot=base_state_snapshot(),
            session_date=date(2026, 5, 5),
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )
        self.assertEqual(content.session_date, date(2026, 5, 5))
        self.assertEqual(content.signals_generated_count, 0)

    def test_utc_midnight_events_are_assigned_to_new_york_session_date(self) -> None:
        content = build_digest_content(
            events=[
                event("SIGNAL_GENERATED", "2026-05-06T02:30:00+00:00", event_id="ny_may5"),
                event("SIGNAL_GENERATED", "2026-05-06T04:30:00+00:00", event_id="ny_may6"),
            ],
            state_snapshot=base_state_snapshot(),
            session_date=date(2026, 5, 5),
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )
        self.assertEqual(content.signals_generated_count, 1)

    def test_dst_transition_uses_zoneinfo_conversion(self) -> None:
        content = build_digest_content(
            events=[
                event("SIGNAL_GENERATED", "2026-03-08T04:30:00+00:00", event_id="before_dst_day"),
                event("SIGNAL_GENERATED", "2026-03-08T06:30:00+00:00", event_id="dst_day"),
            ],
            state_snapshot=base_state_snapshot(),
            session_date=date(2026, 3, 8),
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )
        self.assertEqual(content.signals_generated_count, 1)

    def test_missing_and_malformed_timestamps_count_without_crashing(self) -> None:
        malformed = event("SIGNAL_GENERATED", "not-a-time", event_id="bad")
        missing = event("SIGNAL_GENERATED", "", event_id="missing")
        missing.pop("timestamp")
        content = build_digest_content(
            events=[malformed, missing],
            state_snapshot=base_state_snapshot(),
            session_date=date(2026, 5, 5),
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )
        self.assertEqual(content.ledger_consistency["malformed_timestamp_count"], 2)
        self.assertEqual(content.signals_generated_count, 0)

    def test_unknown_missing_event_types_and_duplicate_ids_are_reported(self) -> None:
        events = [
            event("NOT_REAL", "2026-05-05T15:00:00+00:00", event_id="dup"),
            event("SIGNAL_GENERATED", "2026-05-05T15:01:00+00:00", event_id="dup"),
            {"event_id": "missing_type", "timestamp": "2026-05-05T15:02:00+00:00"},
        ]
        content = build_digest_content(
            events=events,
            state_snapshot=base_state_snapshot(),
            session_date=date(2026, 5, 5),
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )
        self.assertEqual(content.ledger_consistency["unknown_event_type_count"], 2)
        self.assertEqual(content.ledger_consistency["duplicate_event_id_count"], 1)
        self.assertGreater(content.ledger_consistency["missing_required_fields_count"], 0)

    def test_non_counted_close_event_names_are_not_lifecycle_counts(self) -> None:
        content = build_digest_content(
            events=[
                event("CLOSE_INTENT_CREATED", "2026-05-05T15:00:00+00:00"),
                event("CLOSE_ORDER_SUBMITTED", "2026-05-05T15:00:00+00:00"),
                event("CLOSE_ORDER_CONFIRMED", "2026-05-05T15:00:00+00:00"),
                event("CLOSE_FILL_CONFIRMED", "2026-05-05T15:00:00+00:00"),
            ],
            state_snapshot=base_state_snapshot(),
            session_date=date(2026, 5, 5),
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )
        for name in (
            "CLOSE_INTENT_CREATED",
            "CLOSE_ORDER_SUBMITTED",
            "CLOSE_ORDER_CONFIRMED",
            "CLOSE_FILL_CONFIRMED",
        ):
            self.assertNotIn(name, content.lifecycle_status_counts)

    def test_event_categories_are_existing_known_event_types(self) -> None:
        for name in (
            "SIGNAL_GENERATED",
            "SIGNAL_SKIPPED",
            "ORDER_INTENT_CREATED",
            "ORDER_INTENT_EXPIRED",
            "ORDER_INTENT_CANCELLED",
            "ORDER_SUBMITTED",
            "FILL_CONFIRMED",
            "POSITION_CLOSED",
            "RECONCILIATION_FAILED",
        ):
            self.assertIn(name, KNOWN_EVENT_TYPES)


class DigestWriteTests(unittest.TestCase):
    def content(self) -> DigestContent:
        return build_digest_content(
            events=[],
            state_snapshot=base_state_snapshot(),
            session_date=date(2026, 5, 5),
            strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
        )

    def test_write_digest_writes_file_prints_stdout_and_sends_telegram(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sent = []
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                write_digest(
                    content=self.content(),
                    snapshots_dir=Path(tmp) / "data/snapshots",
                    telegram_sender=sent.append,
                    now=datetime(2026, 5, 5, 17, 0, tzinfo=NY),
                )
            path = Path(tmp) / "data/snapshots/digest_2026-05-05.txt"
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), self.content().text)
            self.assertIn("Dry-run daily digest: 2026-05-05", stdout.getvalue())
            self.assertEqual(sent, [self.content().text])

    def test_telegram_unavailable_or_failure_does_not_fail_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                write_digest(
                    content=self.content(),
                    snapshots_dir=Path(tmp) / "data/snapshots",
                    telegram_sender=None,
                    now=datetime(2026, 5, 5, 17, 0, tzinfo=NY),
                )
            self.assertIn("WARNING", stderr.getvalue())

            def fail_sender(text):
                raise RuntimeError("no telegram")

            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                write_digest(
                    content=self.content(),
                    snapshots_dir=Path(tmp) / "data/snapshots",
                    telegram_sender=fail_sender,
                    now=datetime(2026, 5, 5, 17, 0, tzinfo=NY),
                )
            self.assertIn("WARNING", stderr.getvalue())

    def test_file_write_errors_are_not_swallowed_as_telegram_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshots_path = Path(tmp) / "data/snapshots"
            snapshots_path.parent.mkdir(parents=True)
            snapshots_path.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    write_digest(
                        content=self.content(),
                        snapshots_dir=snapshots_path,
                        telegram_sender=None,
                        now=datetime(2026, 5, 5, 17, 0, tzinfo=NY),
                    )


class DigestIntegrationTests(unittest.TestCase):
    def test_run_daily_digest_reads_local_inputs_without_mutating_state_or_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_dir = root / "data/snapshots"
            snapshots_dir.mkdir(parents=True)
            snapshot_time = datetime(2026, 5, 5, 16, 59, tzinfo=NY)
            (snapshots_dir / "account.json").write_text(
                json.dumps({"timestamp": snapshot_time.isoformat()}),
                encoding="utf-8",
            )
            halt_path = root / "data/state/halt_state.json"
            state_store = FakeStateStore()
            before_state = deepcopy(state_store.__dict__)
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                content = run_daily_digest(
                    state_store=state_store,
                    snapshots_dir=snapshots_dir,
                    halt_state_path=halt_path,
                    ledger_reader=FakeLedgerReader(
                        [event("SIGNAL_GENERATED", "2026-05-05T15:00:00+00:00")]
                    ),
                    telegram_sender=None,
                    now=datetime(2026, 5, 5, 17, 0, tzinfo=NY),
                )
            self.assertEqual(content.signals_generated_count, 1)
            self.assertFalse(state_store.saved)
            self.assertEqual(before_state, state_store.__dict__)
            self.assertTrue((snapshots_dir / "digest_2026-05-05.txt").exists())
            self.assertIn("Dry-run daily digest", stdout.getvalue())

    def test_state_snapshot_includes_halt_and_snapshot_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_dir = root / "data/snapshots"
            snapshots_dir.mkdir(parents=True)
            (snapshots_dir / "account.json").write_text(
                json.dumps({"timestamp": datetime(2026, 5, 5, 16, 59, tzinfo=NY).isoformat()}),
                encoding="utf-8",
            )
            halt_path = root / "data/state/halt_state.json"
            halt_path.parent.mkdir(parents=True)
            halt_path.write_text(json.dumps({"scope": "account", "tier": "hard"}), encoding="utf-8")
            snapshot = build_state_snapshot(
                state_store=FakeStateStore(),
                snapshots_dir=snapshots_dir,
                halt_state_path=halt_path,
                now=datetime(2026, 5, 5, 17, 0, tzinfo=NY),
                strategy_ids=[S01_VOL_BASELINE, S02_VOL_ENHANCED],
            )
            self.assertEqual(snapshot["halt_state"], "active:account:hard")
            self.assertEqual(snapshot["active_halt_conditions"], 1)
            self.assertTrue(snapshot["account_snapshot_fresh"])
            self.assertTrue(snapshot["nlv_valid"])

    def test_daily_digest_job_is_wired_and_cadence_constant_unchanged(self) -> None:
        source = Path("algo_trader_unified/core/scheduler_cadence.py").read_text(encoding="utf-8")
        self.assertIn("run_daily_digest_job", source)
        self.assertIn("JOB_DAILY_DIGEST", source)
        self.assertEqual(JOB_DAILY_DIGEST, "daily_digest")
        spec = JOB_SPECS[JOB_DAILY_DIGEST]
        self.assertEqual(spec.trigger_type, "cron")
        self.assertEqual(
            spec.trigger_kwargs,
            {
                "day_of_week": "mon-fri",
                "hour": 17,
                "minute": 0,
                "timezone": SCHEDULER_TIMEZONE,
            },
        )
        self.assertEqual(spec.max_instances, 1)
        self.assertFalse(spec.coalesce)

    def test_digest_source_has_no_broker_systemd_or_pipeline_imports(self) -> None:
        source = Path("algo_trader_unified/jobs/daily_digest.py").read_text(encoding="utf-8")
        forbidden = (
            "ib_insync",
            "reqMktData",
            "placeOrder",
            "core.broker",
            "systemd",
            "run_intent_submission_job",
            "run_intent_confirmation_job",
            "run_intent_fill_confirmation_job",
            "run_position_transitions_job",
            ".jsonl",
        )
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
