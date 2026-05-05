from __future__ import annotations

import inspect
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest import mock
from pathlib import Path

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.scheduler import (
    JOB_DAILY_DIGEST,
    JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
    JOB_DRY_RUN_CONFIRM_FILLS,
    JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
    JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
    JOB_EOD_REVIEW,
    JOB_HEARTBEAT,
    JOB_INTENT_CONFIRMATION,
    JOB_INTENT_FILL_CONFIRMATION,
    JOB_INTENT_SUBMISSION,
    JOB_KEEPALIVE,
    JOB_MARKET_OPEN_SCAN,
    JOB_POSITION_TRANSITIONS,
    JOB_RISK_MONITOR,
    JOB_S01_VOL_SCAN,
    JOB_S02_VOL_SCAN,
    JOB_SPECS,
    SCHEDULER_TIMEZONE,
)
from algo_trader_unified.config.variants import S01_CONFIG
from algo_trader_unified.core import scheduler_cadence
from algo_trader_unified.core.readiness import ReadinessStatus
from algo_trader_unified.core.skip_reasons import (
    SKIP_READINESS_NOT_EVALUATED,
    SKIP_NLV_DEGRADED,
    SKIP_STATESTORE_UNREADABLE,
)


class FakeScheduler:
    def __init__(self) -> None:
        self.jobs = []

    def add_job(self, func, **kwargs) -> None:
        self.jobs.append({"func": func, **kwargs})


class FakeLedger:
    def __init__(self) -> None:
        self.events = []

    def append(self, **kwargs):
        self.events.append(kwargs)
        return "evt_fake"


class FakeStateStore:
    def __init__(self) -> None:
        self.state = {
            "positions": {
                "pos_open": {"status": "open"},
                "pos_closed": {"status": "closed"},
            },
            "order_intents": {
                "intent_1": {"status": "created"},
                "intent_2": {"status": "cancelled"},
            },
            "close_intents": {
                "close_1": {"status": "submitted"},
            },
        }
        self.readiness = {}

    def list_positions(self, status=None):
        positions = list(self.state["positions"].values())
        if status is None:
            return positions
        return [position for position in positions if position.get("status") == status]

    def list_order_intents(self, strategy_id=None):
        records = list(self.state["order_intents"].values())
        if strategy_id is None:
            return records
        return [record for record in records if record.get("strategy_id") == strategy_id]

    def list_close_intents(self, strategy_id=None):
        records = list(self.state["close_intents"].values())
        if strategy_id is None:
            return records
        return [record for record in records if record.get("strategy_id") == strategy_id]

    def get_order_intent(self, intent_id):
        return self.state["order_intents"].get(intent_id)

    def get_close_intent(self, close_intent_id):
        return self.state["close_intents"].get(close_intent_id)

    def get_readiness(self, strategy_id):
        return self.readiness.get(strategy_id)


class FakeReadinessManager:
    def __init__(self, readiness=None, exc=None):
        self.readiness = readiness
        self.exc = exc

    def get_readiness(self, strategy_id):
        if self.exc is not None:
            raise self.exc
        return self.readiness


class SchedulerCadenceCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snapshots_dir = self.root / "data/snapshots"
        self.halt_state_path = self.root / "data/state/halt_state.json"
        self.state_store = FakeStateStore()
        self.ledger = FakeLedger()
        self.scheduler = FakeScheduler()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build(self, *, enable_triggers=True, enable_lifecycle_pipeline=False):
        return scheduler_cadence.build_scheduler(
            enable_triggers=enable_triggers,
            enable_lifecycle_pipeline=enable_lifecycle_pipeline,
            state_store=self.state_store,
            ledger=self.ledger,
            readiness_provider=lambda: None,
            snapshots_dir=self.snapshots_dir,
            halt_state_path=self.halt_state_path,
            scheduler_factory=lambda: self.scheduler,
        )


class BuilderSignatureAndRegistrationTests(SchedulerCadenceCase):
    def test_build_scheduler_signature_accepts_required_dependencies(self) -> None:
        params = inspect.signature(scheduler_cadence.build_scheduler).parameters
        for name in (
            "enable_triggers",
            "enable_lifecycle_pipeline",
            "state_store",
            "ledger",
            "readiness_provider",
            "snapshots_dir",
            "halt_state_path",
            "scheduler_factory",
        ):
            self.assertIn(name, params)

    def test_enable_triggers_false_registers_zero_jobs(self) -> None:
        for enable_lifecycle_pipeline in (False, True):
            self.scheduler.jobs = []
            scheduler = self.build(
                enable_triggers=False,
                enable_lifecycle_pipeline=enable_lifecycle_pipeline,
            )
            self.assertIs(scheduler, self.scheduler)
            self.assertEqual(self.scheduler.jobs, [])

    def test_enable_triggers_true_registers_exact_stage4a_jobs(self) -> None:
        self.build(enable_triggers=True, enable_lifecycle_pipeline=False)
        self.assertEqual(
            [job["id"] for job in self.scheduler.jobs],
            [
                JOB_KEEPALIVE,
                JOB_RISK_MONITOR,
                JOB_HEARTBEAT,
                JOB_MARKET_OPEN_SCAN,
                JOB_S01_VOL_SCAN,
                JOB_S02_VOL_SCAN,
                JOB_EOD_REVIEW,
                JOB_DAILY_DIGEST,
            ],
        )
        forbidden = {
            JOB_INTENT_SUBMISSION,
            JOB_INTENT_CONFIRMATION,
            JOB_INTENT_FILL_CONFIRMATION,
            JOB_POSITION_TRANSITIONS,
        }
        self.assertFalse(forbidden & {job["id"] for job in self.scheduler.jobs})

    def test_enable_lifecycle_pipeline_registers_stage4a_plus_stage4b_jobs(self) -> None:
        self.build(enable_triggers=True, enable_lifecycle_pipeline=True)
        self.assertEqual(
            [job["id"] for job in self.scheduler.jobs],
            [
                JOB_KEEPALIVE,
                JOB_RISK_MONITOR,
                JOB_HEARTBEAT,
                JOB_MARKET_OPEN_SCAN,
                JOB_S01_VOL_SCAN,
                JOB_S02_VOL_SCAN,
                JOB_EOD_REVIEW,
                JOB_DAILY_DIGEST,
                JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
                JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
                JOB_DRY_RUN_CONFIRM_FILLS,
                JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
            ],
        )

    def test_interval_jobs_have_expected_cadence_and_options(self) -> None:
        self.build(enable_lifecycle_pipeline=True)
        by_id = {job["id"]: job for job in self.scheduler.jobs}
        self.assertEqual(by_id[JOB_KEEPALIVE]["trigger"], "interval")
        self.assertEqual(by_id[JOB_KEEPALIVE]["seconds"], 60)
        self.assertEqual(by_id[JOB_RISK_MONITOR]["minutes"], 5)
        self.assertEqual(by_id[JOB_HEARTBEAT]["minutes"], 5)
        for job_id in (JOB_KEEPALIVE, JOB_RISK_MONITOR, JOB_HEARTBEAT):
            self.assertEqual(by_id[job_id]["max_instances"], 1)
            self.assertTrue(by_id[job_id]["coalesce"])
        for job_id in (
            JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
            JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
            JOB_DRY_RUN_CONFIRM_FILLS,
            JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
        ):
            self.assertEqual(by_id[job_id]["trigger"], "interval")
            self.assertEqual(by_id[job_id]["seconds"], 60)
            self.assertEqual(by_id[job_id]["max_instances"], 1)
            self.assertTrue(by_id[job_id]["coalesce"])

    def test_cron_jobs_have_expected_cadence_options_and_timezone(self) -> None:
        self.build()
        by_id = {job["id"]: job for job in self.scheduler.jobs}
        expected = {
            JOB_MARKET_OPEN_SCAN: (9, 35, True),
            JOB_S01_VOL_SCAN: (9, 40, True),
            JOB_S02_VOL_SCAN: (9, 45, True),
            JOB_EOD_REVIEW: (16, 5, True),
            JOB_DAILY_DIGEST: (17, 0, False),
        }
        for job_id, (hour, minute, coalesce) in expected.items():
            job = by_id[job_id]
            self.assertEqual(job["trigger"], "cron")
            self.assertEqual(job["day_of_week"], "mon-fri")
            self.assertEqual(job["hour"], hour)
            self.assertEqual(job["minute"], minute)
            self.assertEqual(job["timezone"], SCHEDULER_TIMEZONE)
            self.assertEqual(job["max_instances"], 1)
            self.assertEqual(job["coalesce"], coalesce)

    def test_lifecycle_scheduled_callables_map_to_existing_stage3_wrappers(self) -> None:
        self.build(enable_lifecycle_pipeline=True)
        by_id = {job["id"]: job for job in self.scheduler.jobs}
        expected_by_id = {
            JOB_DRY_RUN_SUBMIT_PENDING_INTENTS: "run_intent_submission_job",
            JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS: "run_intent_confirmation_job",
            JOB_DRY_RUN_CONFIRM_FILLS: "run_intent_fill_confirmation_job",
            JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS: "run_position_transitions_job",
        }
        for job_id, wrapper_name in expected_by_id.items():
            wrapper = mock.Mock(return_value={"dry_run": True, "job_id": job_id})
            with mock.patch.object(scheduler_cadence, wrapper_name, wrapper):
                result = by_id[job_id]["func"]()
            self.assertEqual(result, {"dry_run": True, "job_id": job_id})
            wrapper.assert_called_once()
            kwargs = wrapper.call_args.kwargs
            self.assertIs(kwargs["state_store"], self.state_store)
            self.assertIs(kwargs["ledger"], self.ledger)
            self.assertIsInstance(kwargs["now"], str)

    def test_lifecycle_scheduled_callables_noop_safely_and_return_json_safe_dicts(self) -> None:
        self.state_store.state["order_intents"] = {
            "intent_1": {"intent_id": "intent_1", "status": "cancelled"}
        }
        self.state_store.state["close_intents"] = {
            "close_1": {"close_intent_id": "close_1", "status": "cancelled"}
        }
        before = deepcopy(self.state_store.state)
        self.build(enable_lifecycle_pipeline=True)
        lifecycle_ids = (
            JOB_DRY_RUN_CONFIRM_FILLS,
            JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
            JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
            JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
        )
        by_id = {job["id"]: job for job in self.scheduler.jobs}
        for job_id in lifecycle_ids:
            result = by_id[job_id]["func"]()
            self.assertIs(result["dry_run"], True)
            json.dumps(result)
        self.assertEqual(self.state_store.state, before)
        self.assertEqual(self.ledger.events, [])

    def test_lifecycle_pipeline_job_specs_are_disabled_by_default(self) -> None:
        for job_id in (
            JOB_DRY_RUN_SUBMIT_PENDING_INTENTS,
            JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS,
            JOB_DRY_RUN_CONFIRM_FILLS,
            JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS,
        ):
            spec = JOB_SPECS[job_id]
            self.assertFalse(spec.enabled)
            self.assertEqual(spec.trigger_type, "interval")
            self.assertEqual(spec.trigger_kwargs, {"seconds": 60})
            self.assertEqual(spec.max_instances, 1)
            self.assertTrue(spec.coalesce)


class LocalJobBehaviorTests(SchedulerCadenceCase):
    def test_keepalive_does_not_need_broker_or_write(self) -> None:
        self.assertEqual(scheduler_cadence.run_keepalive(), {"status": "alive"})

    def test_risk_monitor_reads_local_state_and_logs_without_mutation(self) -> None:
        self.snapshots_dir.mkdir(parents=True)
        (self.snapshots_dir / "account.json").write_text(
            json.dumps({"timestamp": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )
        self.halt_state_path.parent.mkdir(parents=True)
        self.halt_state_path.write_text(
            json.dumps({"scope": "account", "tier": "hard"}),
            encoding="utf-8",
        )
        before = deepcopy(self.state_store.state)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = scheduler_cadence.run_risk_monitor(
                state_store=self.state_store,
                snapshots_dir=self.snapshots_dir,
                halt_state_path=self.halt_state_path,
            )
        self.assertIn("Risk monitor.", stdout.getvalue())
        self.assertEqual(result["open_positions"], 1)
        self.assertEqual(result["active_intents"], 2)
        self.assertTrue(result["snapshot_fresh"])
        self.assertIn("active:account:hard", result["halt"])
        self.assertEqual(before, self.state_store.state)
        self.assertEqual(self.ledger.events, [])

    def test_risk_monitor_marks_stale_snapshot_unfresh(self) -> None:
        self.snapshots_dir.mkdir(parents=True)
        stale = datetime.now(timezone.utc) - timedelta(minutes=30)
        (self.snapshots_dir / "account.json").write_text(
            json.dumps({"timestamp": stale.isoformat()}),
            encoding="utf-8",
        )
        with redirect_stdout(io.StringIO()):
            result = scheduler_cadence.run_risk_monitor(
                state_store=self.state_store,
                snapshots_dir=self.snapshots_dir,
                halt_state_path=self.halt_state_path,
            )
        self.assertFalse(result["snapshot_fresh"])

    def test_heartbeat_prints_line_and_writes_snapshot_without_state_mutation(self) -> None:
        before = deepcopy(self.state_store.state)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = scheduler_cadence.run_heartbeat(
                state_store=self.state_store,
                snapshots_dir=self.snapshots_dir,
                halt_state_path=self.halt_state_path,
            )
        self.assertIn(
            "Daemon alive. Open positions: 1, Active intents: 2, Halt: inactive",
            stdout.getvalue(),
        )
        snapshots = list(self.snapshots_dir.glob("heartbeat_*.json"))
        self.assertEqual(len(snapshots), 1)
        payload = json.loads(snapshots[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["open_positions"], 1)
        self.assertEqual(payload["active_intents"], 2)
        self.assertEqual(result["halt"], "inactive")
        self.assertEqual(before, self.state_store.state)
        self.assertEqual(self.ledger.events, [])


class VolScanSafetyTests(SchedulerCadenceCase):
    def test_vol_scan_skip_when_readiness_not_evaluated(self) -> None:
        result = scheduler_cadence.run_safe_vol_scan(
            config=S01_CONFIG,
            readiness_manager=FakeReadinessManager(readiness=None),
            state_store=self.state_store,
            ledger=self.ledger,
        )
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["detail"], "readiness_not_evaluated")
        self.assertEqual(
            self.ledger.events[-1]["payload"]["skip_reason"],
            SKIP_READINESS_NOT_EVALUATED,
        )

    def test_vol_scan_skip_when_statestore_unreadable(self) -> None:
        result = scheduler_cadence.run_safe_vol_scan(
            config=S01_CONFIG,
            readiness_manager=FakeReadinessManager(exc=RuntimeError("corrupt")),
            state_store=self.state_store,
            ledger=self.ledger,
        )
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["detail"], "statestore_unreadable")
        self.assertEqual(
            self.ledger.events[-1]["payload"]["skip_reason"],
            SKIP_STATESTORE_UNREADABLE,
        )

    def test_vol_scan_uses_existing_readiness_skip_reason(self) -> None:
        readiness = ReadinessStatus(
            strategy_id=S01_VOL_BASELINE,
            ready_for_entries=False,
            reason=SKIP_NLV_DEGRADED,
            checked_at="2026-05-05T13:35:00+00:00",
            dirty_state=False,
            unknown_broker_exposure=False,
            nlv_degraded=True,
            halt_active=False,
            calendar_expired=False,
            iv_baseline_available=True,
        )
        result = scheduler_cadence.run_safe_vol_scan(
            config=S01_CONFIG,
            readiness_manager=FakeReadinessManager(readiness=readiness),
            state_store=self.state_store,
            ledger=self.ledger,
        )
        self.assertEqual(result["detail"], "readiness_skipped")
        self.assertEqual(
            self.ledger.events[-1]["payload"]["skip_reason"],
            SKIP_NLV_DEGRADED,
        )


class BrokerBoundaryTests(unittest.TestCase):
    def test_stage4b1_source_has_no_broker_systemd_or_live_scheduler_imports(self) -> None:
        source = Path("algo_trader_unified/core/scheduler_cadence.py").read_text(
            encoding="utf-8"
        )
        forbidden = (
            "ib_insync",
            "reqMktData",
            "placeOrder",
            "core.broker",
            "systemd",
            "jobs.vol",
            "BackgroundScheduler.start",
        )
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
