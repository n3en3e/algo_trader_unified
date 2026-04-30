from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import JOB_S01_VOL_SCAN, JOB_S02_VOL_SCAN, JOB_SPECS
from algo_trader_unified.config.variants import S01_CONFIG, S02_CONFIG, StrategyVariantConfig
from algo_trader_unified.core.ledger import KNOWN_EVENT_TYPES, LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.readiness import ReadinessManager, ReadinessStatus
from algo_trader_unified.core.scheduler import UnifiedScheduler
from algo_trader_unified.core.skip_reasons import (
    SKIP_ALREADY_SIGNALED_TODAY,
    SKIP_NEEDS_RECONCILIATION,
    SKIP_NLV_DEGRADED,
    SKIP_VIX_GATE,
)
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.jobs import vol
from algo_trader_unified.jobs.vol import run_s01_vol_scan, run_s02_vol_scan, run_vol_scan
from algo_trader_unified.strategies.vol.signals import SignalResult, VolSignalInput


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def source_path(*parts: str) -> Path:
    return PACKAGE_ROOT.joinpath(*parts)


def ledger_event(
    *,
    strategy_id: str,
    event_type: str = "SIGNAL_GENERATED",
    timestamp: str = "2026-04-27T13:40:00+00:00",
    payload: dict | None = None,
) -> dict:
    return {
        "event_id": f"evt_{strategy_id}",
        "event_type": event_type,
        "timestamp": timestamp,
        "strategy_id": strategy_id,
        "execution_mode": "paper_only",
        "source_module": "test",
        "position_id": None,
        "opportunity_id": None,
        "payload": payload or {"strategy_id": strategy_id},
    }


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
        self.execution_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    def execution_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.execution_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def clean_input(self, strategy_id: str = S02_VOL_ENHANCED, **overrides) -> VolSignalInput:
        values = {
            "symbol": "XSP",
            "current_date": date(2026, 4, 27),
            "vix": 18.0,
            "iv_rank": 45.0,
            "target_dte": 45,
            "blackout_dates": (),
            "order_ref_candidate": f"{strategy_id}|P0427XSP|OPEN",
        }
        values.update(overrides)
        return VolSignalInput(**values)

    def set_readiness(
        self,
        strategy_id: str,
        *,
        ready_for_entries: bool,
        reason: str | None = None,
    ) -> None:
        self.manager.update_readiness(
            ReadinessStatus(
                strategy_id=strategy_id,
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

    def run_s02(self, **kwargs):
        broker = kwargs.pop("broker", mock.Mock())
        result = run_s02_vol_scan(
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


class SharedVolScanArchitectureTests(unittest.TestCase):
    def test_s01_wrapper_is_thin_and_uses_s01_config(self) -> None:
        with mock.patch.object(vol, "run_vol_scan", return_value=mock.sentinel.result) as shared:
            result = run_s01_vol_scan(
                readiness_manager=mock.Mock(),
                state_store=mock.Mock(),
                ledger=mock.Mock(root_dir="."),
            )
        self.assertIs(result, mock.sentinel.result)
        self.assertIs(shared.call_args.kwargs["config"], S01_CONFIG)

    def test_s02_wrapper_is_thin_and_uses_s02_config(self) -> None:
        with mock.patch.object(vol, "run_vol_scan", return_value=mock.sentinel.result) as shared:
            result = run_s02_vol_scan(
                readiness_manager=mock.Mock(),
                state_store=mock.Mock(),
                ledger=mock.Mock(root_dir="."),
            )
        self.assertIs(result, mock.sentinel.result)
        self.assertIs(shared.call_args.kwargs["config"], S02_CONFIG)

    def test_shared_run_vol_scan_source_is_not_hardcoded_to_s01(self) -> None:
        source = inspect.getsource(run_vol_scan)
        self.assertIn("config.strategy_id", source)
        self.assertIn("config.sleeve_id", source)
        self.assertIn("config.execution_mode", source)
        self.assertNotIn("S01_VOL_BASELINE", source)

    def test_unknown_strategy_id_has_readable_job_id_error(self) -> None:
        unknown_config = StrategyVariantConfig(
            strategy_id="S99_UNKNOWN_VOL",
            display_name="Unknown Vol",
            legacy_source="test",
            engine_type="vol_selling",
            sleeve_id="VOL",
            nominal_research_allocation=1,
            execution_mode="paper_only",
            params={
                "iv_rank_min": 30,
                "vix_gate_min": None,
                "target_dte": 45,
            },
        )
        with self.assertRaisesRegex(
            KeyError,
            "No job_id registered for strategy_id=S99_UNKNOWN_VOL",
        ):
            run_vol_scan(
                config=unknown_config,
                readiness_manager=mock.Mock(),
                state_store=mock.Mock(),
                ledger=mock.Mock(root_dir="."),
            )


class SchedulerConfigPhase3DTests(unittest.TestCase):
    def test_s01_and_s02_standard_vol_jobs_enabled(self) -> None:
        self.assertTrue(JOB_SPECS[JOB_S01_VOL_SCAN].enabled)
        self.assertTrue(JOB_SPECS[JOB_S02_VOL_SCAN].enabled)
        for job_id in (JOB_S01_VOL_SCAN, JOB_S02_VOL_SCAN):
            self.assertEqual(JOB_SPECS[job_id].max_instances, 1)
            self.assertTrue(JOB_SPECS[job_id].coalesce)
        self.assertFalse(any("0dte" in job_id.lower() for job_id in JOB_SPECS))


class S02DryRunVolScanTests(TmpCase):
    def test_s02_readiness_skip_writes_signal_skipped_only(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=False, reason=SKIP_NLV_DEGRADED)
        provider = mock.Mock(return_value=self.clean_input())
        result, broker = self.run_s02(signal_context_provider=provider)
        events = self.execution_events()
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "readiness_skipped")
        provider.assert_not_called()
        self.assertEqual(events[-1]["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(events[-1]["strategy_id"], S02_VOL_ENHANCED)
        self.assertEqual(events[-1]["payload"]["strategy_id"], S02_VOL_ENHANCED)
        self.assertEqual(events[-1]["payload"]["sleeve_id"], S02_CONFIG.sleeve_id)
        self.assertEqual(events[-1]["payload"]["gate_name"], "vol_readiness_gate")
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        broker.submit_order.assert_not_called()

    def test_run_vol_scan_uses_named_default_provider_when_omitted(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=True)
        current_time = datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc)
        with mock.patch.object(
            vol,
            "default_vol_signal_context_provider",
            wraps=vol.default_vol_signal_context_provider,
        ) as default_provider:
            result = run_vol_scan(
                config=S02_CONFIG,
                readiness_manager=self.manager,
                state_store=self.state_store,
                ledger=self.ledger,
                broker=mock.Mock(),
                current_time=current_time,
            )
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "signal_skipped")
        default_provider.assert_called_once_with(S02_CONFIG, current_time)

    def test_s02_signal_skip_uses_variant_vix_gate(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=True)
        provider = mock.Mock(return_value=self.clean_input(vix=10.0, iv_rank=45.0))
        result, broker = self.run_s02(signal_context_provider=provider)
        events = self.execution_events()
        provider.assert_called_once()
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "signal_skipped")
        self.assertEqual(events[-1]["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(events[-1]["strategy_id"], S02_VOL_ENHANCED)
        self.assertEqual(events[-1]["payload"]["skip_reason"], SKIP_VIX_GATE)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        broker.submit_order.assert_not_called()
        broker.placeOrder.assert_not_called()
        broker.cancelOrder.assert_not_called()

    def test_s02_clean_signal_creates_order_intent(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=True)
        result, broker = self.run_s02(signal_context_provider=lambda: self.clean_input())
        events = self.execution_events()
        self.assertEqual(result.status, "success")
        self.assertEqual(result.detail, "order_intent_created")
        self.assertEqual(events[-1]["event_type"], "SIGNAL_GENERATED")
        self.assertEqual(events[-1]["strategy_id"], S02_VOL_ENHANCED)
        self.assertEqual(events[-1]["payload"]["event_detail"], "S02_VOL_SIGNAL_GENERATED")
        self.assertIn("sizing_context", events[-1]["payload"])
        self.assertIn("risk_context", events[-1]["payload"])
        self.assertIn("ORDER_INTENT_CREATED", self.order_path.read_text(encoding="utf-8"))
        broker.submit_order.assert_not_called()

    def test_s02_default_reader_path_without_injected_reader(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=True)
        broker = mock.Mock()
        result = run_s02_vol_scan(
            readiness_manager=self.manager,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=broker,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
            signal_context_provider=lambda: self.clean_input(),
        )
        events = self.execution_events()
        self.assertEqual(result.status, "success")
        self.assertEqual(result.detail, "order_intent_created")
        self.assertEqual([event["event_type"] for event in events], ["SIGNAL_GENERATED"])
        self.assertEqual(events[-1]["strategy_id"], S02_VOL_ENHANCED)
        self.assertIn("ORDER_INTENT_CREATED", self.order_path.read_text(encoding="utf-8"))
        broker.submit_order.assert_not_called()

    def test_s02_duplicate_same_day_skips_before_signal_provider(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=True)
        self.write_execution_events(
            ledger_event(strategy_id=S02_VOL_ENHANCED),
            ledger_event(strategy_id=S02_VOL_ENHANCED, timestamp="2026-04-27T14:40:00+00:00"),
        )
        provider = mock.Mock(return_value=self.clean_input())
        with mock.patch("algo_trader_unified.jobs.vol.VolSellingEngine") as engine_class:
            result, broker = self.run_s02(
                ledger_reader=self.reader(),
                signal_context_provider=provider,
            )
        events = self.execution_events()
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "already_signaled_today")
        provider.assert_not_called()
        engine_class.assert_not_called()
        self.assertEqual(events[-1]["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(events[-1]["payload"]["skip_reason"], SKIP_ALREADY_SIGNALED_TODAY)
        self.assertEqual(events[-1]["payload"]["matched_event_count"], 2)
        self.assertEqual([event["event_type"] for event in events].count("SIGNAL_GENERATED"), 2)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")
        broker.submit_order.assert_not_called()

    def test_s01_s02_idempotency_is_isolated(self) -> None:
        self.set_readiness(S01_VOL_BASELINE, ready_for_entries=True)
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=True)
        self.write_execution_events(ledger_event(strategy_id=S01_VOL_BASELINE))
        s02_result, _broker = self.run_s02(
            ledger_reader=self.reader(),
            signal_context_provider=lambda: self.clean_input(S02_VOL_ENHANCED),
        )
        self.assertEqual(s02_result.detail, "order_intent_created")

        self.write_execution_events(ledger_event(strategy_id=S02_VOL_ENHANCED))
        s01_result, _broker = self.run_s01(
            ledger_reader=self.reader(),
            signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE),
        )
        self.assertEqual(s01_result.detail, "order_intent_created")

    def test_s01_s02_readiness_is_isolated(self) -> None:
        self.set_readiness(S01_VOL_BASELINE, ready_for_entries=True)
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=True)
        s02_before = self.state_store.get_readiness(S02_VOL_ENHANCED)
        self.run_s01(signal_context_provider=lambda: self.clean_input(S01_VOL_BASELINE))
        self.assertEqual(self.state_store.get_readiness(S02_VOL_ENHANCED), s02_before)

        s01_before = self.state_store.get_readiness(S01_VOL_BASELINE)
        self.run_s02(signal_context_provider=lambda: self.clean_input(S02_VOL_ENHANCED))
        self.assertEqual(self.state_store.get_readiness(S01_VOL_BASELINE), s01_before)


class UnifiedSchedulerS02Tests(TmpCase):
    def scheduler(self, readiness_manager=None) -> UnifiedScheduler:
        return UnifiedScheduler(
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_manager=readiness_manager or self.manager,
        )

    def test_run_job_once_executes_s02_readiness_skip(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=False, reason=SKIP_NEEDS_RECONCILIATION)
        result = self.scheduler().run_job_once(
            JOB_S02_VOL_SCAN,
            signal_context_provider=mock.Mock(return_value=self.clean_input()),
            broker=mock.Mock(),
        )
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "readiness_skipped")
        event = self.execution_events()[-1]
        self.assertEqual(event["event_type"], "SIGNAL_SKIPPED")
        self.assertEqual(event["strategy_id"], S02_VOL_ENHANCED)

    def test_run_job_once_executes_s02_clean_signal(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=True)
        broker = mock.Mock()
        result = self.scheduler().run_job_once(
            JOB_S02_VOL_SCAN,
            signal_context_provider=lambda: self.clean_input(),
            broker=broker,
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.detail, "order_intent_created")
        self.assertEqual(self.execution_events()[-1]["event_type"], "SIGNAL_GENERATED")
        broker.submit_order.assert_not_called()
        broker.placeOrder.assert_not_called()
        broker.cancelOrder.assert_not_called()

    def test_run_job_once_s02_duplicate_uses_injected_reader(self) -> None:
        self.write_execution_events(ledger_event(strategy_id=S02_VOL_ENHANCED))
        readiness_manager = mock.Mock()
        provider = mock.Mock(return_value=self.clean_input())
        broker = mock.Mock()
        result = self.scheduler(readiness_manager=readiness_manager).run_job_once(
            JOB_S02_VOL_SCAN,
            ledger_reader=self.reader(),
            signal_context_provider=provider,
            broker=broker,
            current_time=datetime(2026, 4, 27, 13, 40, tzinfo=timezone.utc),
        )
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.detail, "already_signaled_today")
        self.assertEqual(self.execution_events()[-1]["payload"]["skip_reason"], SKIP_ALREADY_SIGNALED_TODAY)
        provider.assert_not_called()
        readiness_manager.get_readiness.assert_not_called()
        broker.submit_order.assert_not_called()

    def test_run_job_once_s02_kwargs_passthrough_engine(self) -> None:
        self.set_readiness(S02_VOL_ENHANCED, ready_for_entries=True)
        engine = mock.Mock()
        signal_result = SignalResult(
            should_enter=False,
            skip_reason=SKIP_VIX_GATE,
            skip_detail="mocked",
            sizing_context={"capital": 90000.0, "allocation_pct": 0.225},
            risk_context={"execution_mode": "paper_only", "strategy_id": S02_VOL_ENHANCED},
        )
        engine.generate_standard_strangle_signal.return_value = signal_result
        result = self.scheduler().run_job_once(
            JOB_S02_VOL_SCAN,
            signal_context_provider=lambda: self.clean_input(),
            broker=mock.Mock(),
            engine=engine,
        )
        self.assertEqual(result.detail, "signal_skipped")
        engine.generate_standard_strangle_signal.assert_called_once()


class VolEventTaxonomyTests(TmpCase):
    def test_vol_scans_do_not_write_reserved_event_types(self) -> None:
        for strategy_id, runner in (
            (S01_VOL_BASELINE, self.run_s01),
            (S02_VOL_ENHANCED, self.run_s02),
        ):
            self.set_readiness(strategy_id, ready_for_entries=True)
            runner(signal_context_provider=lambda strategy_id=strategy_id: self.clean_input(strategy_id))
        event_types = {event["event_type"] for event in self.execution_events()}
        self.assertNotIn("OPPORTUNITY_SCORED", event_types)
        self.assertNotIn("OPPORTUNITY" + "_IDENTIFIED", event_types)
        self.assertNotIn("SIGNAL_SIZED", event_types)
        self.assertNotIn("SIGNAL_SIZED", KNOWN_EVENT_TYPES)

    def test_no_forbidden_job_or_scheduler_calls_added(self) -> None:
        source = "\n".join(
            [
                source_path("core", "scheduler.py").read_text(encoding="utf-8"),
                source_path("jobs", "vol.py").read_text(encoding="utf-8"),
            ]
        )
        for forbidden in (
            "create_pending_position(",
            "execute_close(",
            "confirm_close_fill(",
            "record_close(",
            "broker.submit_order(",
            "placeOrder(",
            "cancelOrder(",
            "yfinance",
            "requests",
            "ib_insync",
        ):
            self.assertNotIn(forbidden, source)

    def test_s02_remains_paper_only_and_non_vol_strategy_absent(self) -> None:
        self.assertEqual(S02_CONFIG.execution_mode, "paper_only")
        all_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in PACKAGE_ROOT.rglob("*.py")
            if "__pycache__" not in str(path)
        )
        self.assertNotIn("commodity" + "_vrp", all_sources.lower())
