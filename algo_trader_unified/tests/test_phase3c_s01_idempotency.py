from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE
from algo_trader_unified.config.scheduler import JOB_S01_VOL_SCAN
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader, LedgerReadError
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.scheduler import UnifiedScheduler
from algo_trader_unified.core.skip_reasons import (
    SKIP_ALREADY_SIGNALED_TODAY,
    SKIP_IV_RANK_BELOW_MIN,
    SKIP_NLV_DEGRADED,
)
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs.vol import run_s01_vol_scan
from algo_trader_unified.strategies.vol.signals import VolSignalInput


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def source_path(*parts: str) -> Path:
    return PACKAGE_ROOT.joinpath(*parts)


def ledger_event(
    *,
    event_type: str = "SIGNAL_GENERATED",
    strategy_id: str | None = S01_VOL_BASELINE,
    timestamp: str = "2026-04-27T13:40:00+00:00",
    payload: dict | None = None,
) -> dict:
    event = {
        "event_id": "evt_test",
        "event_type": event_type,
        "timestamp": timestamp,
        "execution_mode": "paper_only",
        "source_module": "test",
        "position_id": None,
        "opportunity_id": None,
        "payload": payload or {"strategy_id": strategy_id},
    }
    if strategy_id is not None:
        event["strategy_id"] = strategy_id
    return event


def order_ledger_event() -> dict:
    return ledger_event(
        event_type="ORDER_SUBMITTED",
        strategy_id=S01_VOL_BASELINE,
        timestamp="2026-04-27T13:41:00+00:00",
        payload={"orderRef": "S01|P0427XSP|OPEN"},
    )


class TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = LedgerAppender(self.root)
        self.manager = ReadinessManager(self.state_store, self.ledger)
        self.execution_path = self.root / "data/ledger/execution_ledger.jsonl"
        self.order_path = self.root / "data/ledger/order_ledger.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def reader(self) -> LedgerReader:
        return LedgerReader(
            execution_ledger_path=self.execution_path,
            order_ledger_path=self.order_path,
        )

    def write_execution_events(self, *events: dict) -> None:
        self.execution_path.parent.mkdir(parents=True, exist_ok=True)
        self.execution_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    def write_order_events(self, *events: dict) -> None:
        self.order_path.parent.mkdir(parents=True, exist_ok=True)
        self.order_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    def execution_events(self) -> list[dict]:
        text = self.execution_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line]

    def set_s01_readiness(self, *, ready_for_entries: bool, reason: str | None = None) -> None:
        self.manager.update_readiness(
            ReadinessStatus(
                strategy_id=S01_VOL_BASELINE,
                ready_for_entries=ready_for_entries,
                reason=reason,
                checked_at="2026-04-27T13:35:00+00:00",
                dirty_state=False,
                unknown_broker_exposure=False,
                nlv_degraded=reason == SKIP_NLV_DEGRADED,
                halt_active=False,
                calendar_expired=False,
                iv_baseline_available=True,
            )
        )

    def clean_input(self, **overrides) -> VolSignalInput:
        values = {
            "symbol": "XSP",
            "current_date": date(2026, 4, 27),
            "vix": 18.0,
            "iv_rank": 45.0,
            "target_dte": 45,
            "blackout_dates": (),
            "order_ref_candidate": "S01|P0427XSP|OPEN",
        }
        values.update(overrides)
        return VolSignalInput(**values)

    def run_s01(self, **kwargs):
        broker = kwargs.pop("broker", mock.Mock())
        result = run_s01_vol_scan(
            readiness_manager=kwargs.pop("readiness_manager", self.manager),
            state_store=self.state_store,
            ledger=self.ledger,
            broker=broker,
            current_time=kwargs.pop(
                "current_time",
                datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            ),
            **kwargs,
        )
        return result, broker


class LedgerReaderTests(TmpCase):
    def test_explicit_paths_missing_and_empty_files_return_empty(self) -> None:
        missing_reader = LedgerReader(
            execution_ledger_path=self.root / "missing_execution.jsonl",
            order_ledger_path=self.root / "missing_order.jsonl",
        )
        now = datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc)
        self.assertEqual(
            missing_reader.read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", now),
            [],
        )
        self.execution_path.write_text("", encoding="utf-8")
        self.assertEqual(self.reader().read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", now), [])

    def test_same_day_et_match_and_exclusions(self) -> None:
        now = datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc)
        event = ledger_event(timestamp="2026-04-27T09:35:00-04:00")
        self.write_execution_events(event)
        self.assertEqual(
            self.reader().read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", now),
            [event],
        )

        self.write_execution_events(ledger_event(timestamp="2026-04-26T23:59:00-04:00"))
        self.assertEqual(self.reader().read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", now), [])

        self.write_execution_events(ledger_event(strategy_id="S99_OTHER"))
        self.assertEqual(self.reader().read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", now), [])

        self.write_execution_events(ledger_event(event_type="SIGNAL_SKIPPED"))
        self.assertEqual(self.reader().read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", now), [])

    def test_utc_boundary_behavior(self) -> None:
        now = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)
        included = ledger_event(timestamp="2026-04-27T04:30:00+00:00")
        excluded = ledger_event(timestamp="2026-04-27T03:30:00+00:00")
        self.write_execution_events(included, excluded)
        self.assertEqual(
            self.reader().read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", now),
            [included],
        )

    def test_naive_timestamp_is_treated_as_local_timezone(self) -> None:
        now = datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc)
        included = ledger_event(timestamp="2026-04-27T09:35:00")
        excluded = ledger_event(timestamp="2026-04-26T23:59:00")
        self.write_execution_events(included, excluded)
        self.assertEqual(
            self.reader().read_today(
                S01_VOL_BASELINE,
                "SIGNAL_GENERATED",
                now,
                timezone="America/New_York",
            ),
            [included],
        )

    def test_payload_strategy_id_does_not_control_filtering(self) -> None:
        now = datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc)
        event = ledger_event(payload={"strategy_id": "S99_OTHER"})
        self.write_execution_events(event)
        self.assertEqual(
            self.reader().read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", now),
            [event],
        )

    def test_missing_top_level_strategy_id_is_skipped(self) -> None:
        now = datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc)
        self.write_execution_events(
            ledger_event(strategy_id=None, payload={"strategy_id": S01_VOL_BASELINE})
        )
        self.assertEqual(self.reader().read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", now), [])

    def test_invalid_json_raises_with_path_and_line_context(self) -> None:
        self.execution_path.write_text('{"event_id":"ok"}\nnot-json\n', encoding="utf-8")
        with self.assertRaisesRegex(LedgerReadError, r"execution_ledger\.jsonl.*line 2"):
            self.reader().read_events("execution")

    def test_invalid_now_raises_ledger_read_error(self) -> None:
        with self.assertRaisesRegex(LedgerReadError, "now must be a datetime"):
            self.reader().read_today(S01_VOL_BASELINE, "SIGNAL_GENERATED", "bad-now")

    def test_read_events_none_returns_both_ledgers(self) -> None:
        execution_event = ledger_event()
        order_event = order_ledger_event()
        self.write_execution_events(execution_event)
        self.write_order_events(order_event)
        before_execution = self.execution_path.read_text(encoding="utf-8")
        before_order = self.order_path.read_text(encoding="utf-8")

        events = self.reader().read_events()

        self.assertIn(execution_event, events)
        self.assertIn(order_event, events)
        self.assertEqual(self.execution_path.read_text(encoding="utf-8"), before_execution)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), before_order)

    def test_unknown_ledger_name_raises(self) -> None:
        with self.assertRaisesRegex(LedgerReadError, "Unknown ledger_name: bad"):
            self.reader().read_events("bad")

    def test_read_only_invariants(self) -> None:
        before = self.execution_path.read_text(encoding="utf-8")
        self.reader().read_events()
        self.assertEqual(self.execution_path.read_text(encoding="utf-8"), before)
        self.assertFalse(hasattr(LedgerReader, "append"))
        self.assertFalse(hasattr(LedgerAppender, "read_events"))
        self.assertFalse(hasattr(LedgerAppender, "read_today"))


class SkipReasonTests(unittest.TestCase):
    def test_skip_reason_is_core_constant_and_not_hardcoded_in_job(self) -> None:
        self.assertEqual(SKIP_ALREADY_SIGNALED_TODAY, "SKIP_ALREADY_SIGNALED_TODAY")
        source = source_path("jobs", "vol.py").read_text(encoding="utf-8")
        self.assertIn("algo_trader_unified.core.skip_reasons import", source)
        self.assertIn("SKIP_ALREADY_SIGNALED_TODAY", source)
        self.assertNotIn('"SKIP_ALREADY_SIGNALED_TODAY"', source)
        self.assertNotIn("'SKIP_ALREADY_SIGNALED_TODAY'", source)


class S01IdempotencyTests(TmpCase):
    def test_duplicate_same_day_skips_before_signal_engine_and_orders(self) -> None:
        self.set_s01_readiness(ready_for_entries=True)
        self.write_execution_events(ledger_event())
        provider = mock.Mock(return_value=self.clean_input())
        with mock.patch("algo_trader_unified.jobs.vol.VolSellingEngine") as engine_class:
            result, broker = self.run_s01(
                ledger_reader=self.reader(),
                signal_context_provider=provider,
            )
        events = self.execution_events()
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "already_signaled_today")
        self.assertIsNone(result.signal_result)
        provider.assert_not_called()
        engine_class.assert_not_called()
        self.assertEqual(events[-1]["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(events[-1]["payload"]["skip_reason"], SKIP_ALREADY_SIGNALED_TODAY)
        self.assertEqual(events[-1]["payload"]["gate_name"], "s01_vol_idempotency_gate")
        self.assertEqual(events[-1]["payload"]["matched_event_count"], 1)
        self.assertEqual([event["event_type"] for event in events].count("SIGNAL_GENERATED"), 1)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        broker.submit_order.assert_not_called()

    def test_duplicate_same_day_counts_multiple_matches(self) -> None:
        self.set_s01_readiness(ready_for_entries=True)
        self.write_execution_events(
            ledger_event(timestamp="2026-04-27T13:40:00+00:00"),
            ledger_event(timestamp="2026-04-27T14:40:00+00:00"),
        )
        result, _broker = self.run_s01(
            ledger_reader=self.reader(),
            signal_context_provider=lambda: self.clean_input(),
        )
        events = self.execution_events()
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "already_signaled_today")
        self.assertEqual(events[-1]["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(events[-1]["payload"]["skip_reason"], SKIP_ALREADY_SIGNALED_TODAY)
        self.assertEqual(events[-1]["payload"]["matched_event_count"], 2)
        self.assertEqual([event["event_type"] for event in events].count("SIGNAL_GENERATED"), 2)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_duplicate_check_wins_before_readiness(self) -> None:
        self.write_execution_events(ledger_event())
        readiness_manager = mock.Mock()
        provider = mock.Mock(return_value=self.clean_input())
        result, _broker = self.run_s01(
            readiness_manager=readiness_manager,
            ledger_reader=self.reader(),
            signal_context_provider=provider,
        )
        events = self.execution_events()
        self.assertEqual(result.detail, "already_signaled_today")
        readiness_manager.get_readiness.assert_not_called()
        provider.assert_not_called()
        self.assertEqual(events[-1]["payload"]["skip_reason"], SKIP_ALREADY_SIGNALED_TODAY)
        self.assertNotEqual(events[-1]["payload"]["gate_name"], "s01_vol_readiness_gate")

    def test_no_duplicate_preserves_readiness_skip_signal_skip_and_signal_generated(self) -> None:
        self.set_s01_readiness(ready_for_entries=False, reason=SKIP_NLV_DEGRADED)
        readiness_result, _broker = self.run_s01(
            ledger_reader=self.reader(),
            signal_context_provider=lambda: self.clean_input(),
        )
        self.assertEqual(readiness_result.detail, "readiness_skipped")
        self.assertEqual(self.execution_events()[-1]["payload"]["skip_reason"], SKIP_NLV_DEGRADED)

        self.execution_path.write_text("", encoding="utf-8")
        self.set_s01_readiness(ready_for_entries=True)
        signal_skip_result, _broker = self.run_s01(
            ledger_reader=self.reader(),
            signal_context_provider=lambda: self.clean_input(iv_rank=0),
        )
        self.assertEqual(signal_skip_result.detail, "signal_skipped")
        self.assertEqual(self.execution_events()[-1]["payload"]["skip_reason"], SKIP_IV_RANK_BELOW_MIN)

        self.execution_path.write_text("", encoding="utf-8")
        generated_result, _broker = self.run_s01(
            ledger_reader=self.reader(),
            signal_context_provider=lambda: self.clean_input(),
        )
        self.assertEqual(generated_result.detail, "signal_generated")
        self.assertEqual(self.execution_events()[-1]["event_type"], "SIGNAL_GENERATED")

    def test_previous_date_other_strategy_and_other_event_type_do_not_block(self) -> None:
        self.set_s01_readiness(ready_for_entries=True)
        for event in (
            ledger_event(timestamp="2026-04-26T13:40:00-04:00"),
            ledger_event(strategy_id="S99_OTHER"),
            ledger_event(event_type="SIGNAL_SKIPPED"),
        ):
            self.execution_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            result, _broker = self.run_s01(
                ledger_reader=self.reader(),
                signal_context_provider=lambda: self.clean_input(),
            )
            self.assertEqual(result.detail, "signal_generated")

    def test_missing_root_dir_without_injected_reader_raises(self) -> None:
        ledger = mock.Mock(spec=["append"])
        with self.assertRaisesRegex(LedgerReadError, "ledger_reader is required"):
            run_s01_vol_scan(
                readiness_manager=self.manager,
                state_store=self.state_store,
                ledger=ledger,
                current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            )

    def test_injected_reader_works_when_ledger_has_no_root_dir(self) -> None:
        self.set_s01_readiness(ready_for_entries=True)
        ledger = mock.Mock(spec=["append"])
        result = run_s01_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=ledger,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(),
            ledger_reader=self.reader(),
        )
        self.assertEqual(result.detail, "signal_generated")
        ledger.append.assert_called_once()


class UnifiedSchedulerIdempotencyTests(TmpCase):
    def test_run_job_once_passes_ledger_reader_and_skips_duplicate(self) -> None:
        self.write_execution_events(ledger_event())
        scheduler = UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=mock.Mock(),
        )
        provider = mock.Mock(return_value=self.clean_input())
        broker = mock.Mock()
        result = scheduler.run_job_once(
            JOB_S01_VOL_SCAN,
            ledger_reader=self.reader(),
            signal_context_provider=provider,
            broker=broker,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
        )
        events = self.execution_events()
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "already_signaled_today")
        self.assertEqual(events[-1]["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(events[-1]["payload"]["skip_reason"], SKIP_ALREADY_SIGNALED_TODAY)
        provider.assert_not_called()
        scheduler.readiness_manager.get_readiness.assert_not_called()
        broker.submit_order.assert_not_called()
        broker.placeOrder.assert_not_called()
        broker.cancelOrder.assert_not_called()
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")


class Phase3CRegressionTests(unittest.TestCase):
    def test_forbidden_scope_strings_and_calls_absent(self) -> None:
        source = "\n".join(
            [
                source_path("core", "ledger.py").read_text(encoding="utf-8"),
                source_path("core", "ledger_reader.py").read_text(encoding="utf-8"),
                source_path("core", "scheduler.py").read_text(encoding="utf-8"),
                source_path("jobs", "vol.py").read_text(encoding="utf-8"),
            ]
        )
        self.assertNotIn("OPPORTUNITY" + "_IDENTIFIED", source)
        for forbidden in (
            "create_pending_position(",
            "execute_close(",
            "confirm_close_fill(",
            "broker.submit_order(",
            "placeOrder(",
            "cancelOrder(",
            "yfinance",
            "requests",
            "ib_insync",
        ):
            self.assertNotIn(forbidden, source)

    def test_s02_remains_disabled_and_paper_only(self) -> None:
        from algo_trader_unified.config.scheduler import JOB_S02_VOL_SCAN, JOB_SPECS
        from algo_trader_unified.config.variants import S02_CONFIG

        self.assertFalse(JOB_SPECS[JOB_S02_VOL_SCAN].enabled)
        self.assertEqual(S02_CONFIG.execution_mode, "paper_only")
