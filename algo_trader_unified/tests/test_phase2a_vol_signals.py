from __future__ import annotations

import inspect
import re
import tempfile
import unittest
from copy import deepcopy
from datetime import date
from dataclasses import replace
from pathlib import Path
from typing import get_type_hints
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.config.variants import S01_CONFIG, S02_CONFIG
from algo_trader_unified.core.broker import IBKRBrokerWrapper
from algo_trader_unified.core.broker import (
    DiagnosticClientOrderError,
    MissingOrderRefError,
)
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.strategies.base import Phase2ARiskManagerStub, RiskManagerProtocol
from algo_trader_unified.strategies.vol.engine import VolSellingEngine
from algo_trader_unified.strategies.vol.engine import InvalidSignalError, LifecycleTransitionError
from algo_trader_unified.strategies.vol.order_manager import (
    CloseFillValidationError,
    InvalidManagementResultError,
    StaleManagementResultError,
    StaleCloseFillError,
    VolOrderManager,
)
from algo_trader_unified.strategies.vol.signals import (
    ACTION_CLOSE_DTE,
    ACTION_CLOSE_MANUAL,
    ACTION_CLOSE_PROFIT_TARGET,
    ACTION_CLOSE_STOP_LOSS,
    ACTION_HOLD,
    SKIP_BLACKOUT_DATE,
    SKIP_EXISTING_POSITION,
    SKIP_HALTED,
    SKIP_IV_RANK_BELOW_MIN,
    SKIP_NEEDS_RECONCILIATION,
    SKIP_ORDERREF_MISSING,
    SKIP_VIX_GATE,
    has_existing_open_position,
    has_needs_reconciliation,
    ManagementInputError,
)


class BlockingRiskManager:
    def __init__(self, *, halted: bool = False, can_enter: bool = True) -> None:
        self._halted = halted
        self._can_enter = can_enter

    def can_enter(self, strategy_id: str) -> bool:
        return self._can_enter

    def is_halted(self, strategy_id: str) -> bool:
        return self._halted


class VolCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = LedgerAppender(self.root)
        self.client = mock.Mock()
        self.broker = IBKRBrokerWrapper(self.client)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def engine(self, config=S01_CONFIG, risk_manager=None) -> VolSellingEngine:
        return VolSellingEngine(
            config=config,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=self.broker,
            risk_manager=risk_manager or Phase2ARiskManagerStub(),
        )

    def run_signal(self, engine: VolSellingEngine, **overrides):
        kwargs = {
            "symbol": "XSP",
            "current_date": date(2026, 4, 27),
            "vix": 18.0,
            "iv_rank": 45.0,
            "target_dte": 45,
            "blackout_dates": (),
            "order_ref_candidate": "S01|P0427XSP|OPEN",
        }
        kwargs.update(overrides)
        return engine.generate_standard_strangle_signal(**kwargs)

    def execution_ledger_text(self) -> str:
        return (self.root / "data/ledger/execution_ledger.jsonl").read_text()

    def assert_context_schema(self, result) -> None:
        self.assertIn("capital", result.sizing_context)
        self.assertIn("allocation_pct", result.sizing_context)
        self.assertIn("execution_mode", result.risk_context)
        self.assertIn("strategy_id", result.risk_context)
        self.assertIsNotNone(result.sizing_context["capital"])
        self.assertIsNotNone(result.sizing_context["allocation_pct"])
        self.assertTrue(result.risk_context["execution_mode"])
        self.assertTrue(result.risk_context["strategy_id"])

    def leg_specs(self):
        return [
            {
                "leg_id": "short_put",
                "symbol": "XSP",
                "expiry": "20260605",
                "strike": 480,
                "right": "P",
                "ratio": 1,
                "signed_qty": -1,
            },
            {
                "leg_id": "short_call",
                "symbol": "XSP",
                "expiry": "20260605",
                "strike": 560,
                "right": "C",
                "ratio": 1,
                "signed_qty": -1,
            },
        ]

    def clean_signal(self, engine: VolSellingEngine | None = None):
        return self.run_signal(engine or self.engine())


class Phase2AConfigTests(unittest.TestCase):
    def test_s01_s02_configs(self) -> None:
        for config in (S01_CONFIG, S02_CONFIG):
            self.assertEqual(config.execution_mode, "paper_only")
            self.assertEqual(config.engine_type, "vol_selling")
            self.assertEqual(config.sleeve_id, "VOL")
            self.assertEqual(config.nominal_research_allocation, 90_000)
            for key in (
                "iv_rank_min",
                "vix_gate_min",
                "target_dte",
                "profit_target_pct",
                "stop_loss_mult",
            ):
                self.assertIn(key, config.params)
        self.assertEqual(S01_CONFIG.strategy_id, S01_VOL_BASELINE)
        self.assertEqual(S02_CONFIG.strategy_id, S02_VOL_ENHANCED)
        self.assertNotEqual(S02_CONFIG.execution_mode, "paper_proxy_for_live")


class Phase2AVolGateTests(VolCase):
    def test_s01_vix_passthrough(self) -> None:
        self.assertIsNone(S01_CONFIG.params["vix_gate_min"])
        result = self.run_signal(self.engine(S01_CONFIG), vix=12.0)
        self.assertTrue(result.should_enter)
        self.assertIn("SIGNAL_GENERATED", self.execution_ledger_text())

    def test_s02_clean_path(self) -> None:
        result = self.run_signal(
            self.engine(S02_CONFIG),
            vix=18.0,
            iv_rank=45.0,
            order_ref_candidate="S02|P0427XSP|OPEN",
        )
        self.assertTrue(result.should_enter)
        self.assertIn("SIGNAL_GENERATED", self.execution_ledger_text())

    def test_disabled_mode_blocks_entry(self) -> None:
        config = replace(S01_CONFIG, execution_mode="disabled")
        result = self.run_signal(self.engine(config))
        self.assertFalse(result.should_enter)
        self.assertEqual(result.skip_reason, SKIP_HALTED)
        self.assertIn("SIGNAL_SKIPPED", self.execution_ledger_text())

    def test_shadow_mode_allows_clean_signal(self) -> None:
        config = replace(S01_CONFIG, execution_mode="shadow")
        result = self.run_signal(self.engine(config))
        self.assertTrue(result.should_enter)
        self.assertIn("SIGNAL_GENERATED", self.execution_ledger_text())

    def test_can_enter_false_writes_signal_skipped(self) -> None:
        result = self.run_signal(self.engine(risk_manager=BlockingRiskManager(can_enter=False)))
        self.assertFalse(result.should_enter)
        self.assertEqual(result.skip_reason, SKIP_HALTED)
        self.assertIn("SIGNAL_SKIPPED", self.execution_ledger_text())

    def test_existing_position_skip(self) -> None:
        self.state_store.state["positions"].append(
            {
                "strategy_id": S01_CONFIG.strategy_id,
                "status": "open",
                "contract_identity": {"underlying": "XSP"},
                "legs": [],
            }
        )
        self.state_store.save()
        result = self.run_signal(self.engine())
        self.assertFalse(result.should_enter)
        self.assertEqual(result.skip_reason, SKIP_EXISTING_POSITION)
        self.assert_context_schema(result)
        self.assertIn("SIGNAL_SKIPPED", self.execution_ledger_text())
        self.client.placeOrder.assert_not_called()

    def test_halted_strategy_skip(self) -> None:
        for risk_manager in (
            BlockingRiskManager(halted=True),
            BlockingRiskManager(can_enter=False),
        ):
            with self.subTest(risk_manager=risk_manager):
                result = self.run_signal(self.engine(risk_manager=risk_manager))
                self.assertFalse(result.should_enter)
                self.assertEqual(result.skip_reason, SKIP_HALTED)
                self.assert_context_schema(result)
                self.client.placeOrder.assert_not_called()
        source = Path("algo_trader_unified/strategies/vol/engine.py").read_text()
        self.assertNotIn("halt_state.json", source)

    def test_blackout_date_skip(self) -> None:
        result = self.run_signal(
            self.engine(),
            blackout_dates=(date(2026, 4, 27),),
        )
        self.assertFalse(result.should_enter)
        self.assertEqual(result.skip_reason, SKIP_BLACKOUT_DATE)
        self.assert_context_schema(result)
        self.assertIn("SIGNAL_SKIPPED", self.execution_ledger_text())

    def test_vix_gate_skip(self) -> None:
        result = self.run_signal(
            self.engine(S02_CONFIG),
            vix=12.5,
            order_ref_candidate="S02|P0427XSP|OPEN",
        )
        self.assertFalse(result.should_enter)
        self.assertEqual(result.skip_reason, SKIP_VIX_GATE)
        self.assert_context_schema(result)
        self.assertIn("SIGNAL_SKIPPED", self.execution_ledger_text())

    def test_iv_rank_skip(self) -> None:
        result = self.run_signal(self.engine(), iv_rank=20.0)
        self.assertFalse(result.should_enter)
        self.assertEqual(result.skip_reason, SKIP_IV_RANK_BELOW_MIN)
        self.assert_context_schema(result)
        self.assertIn("SIGNAL_SKIPPED", self.execution_ledger_text())

    def test_needs_reconciliation_skip(self) -> None:
        self.state_store.state["positions"].append(
            {
                "strategy_id": S01_CONFIG.strategy_id,
                "status": "NEEDS_RECONCILIATION",
                "contract_identity": {"underlying": "XSP"},
                "legs": [],
            }
        )
        self.state_store.save()
        result = self.run_signal(self.engine())
        self.assertFalse(result.should_enter)
        self.assertEqual(result.skip_reason, SKIP_NEEDS_RECONCILIATION)
        self.assert_context_schema(result)
        self.assertIn("SIGNAL_SKIPPED", self.execution_ledger_text())

    def test_missing_order_ref_skip(self) -> None:
        for order_ref_candidate in (None, "", "   "):
            with self.subTest(order_ref_candidate=order_ref_candidate):
                result = self.run_signal(self.engine(), order_ref_candidate=order_ref_candidate)
                self.assertFalse(result.should_enter)
                self.assertEqual(result.skip_reason, SKIP_ORDERREF_MISSING)
                self.assert_context_schema(result)
                self.client.placeOrder.assert_not_called()
        self.assertIn("SIGNAL_SKIPPED", self.execution_ledger_text())

    def test_clean_path_signal_generated(self) -> None:
        result = self.run_signal(self.engine())
        self.assertTrue(result.should_enter)
        self.assertIsNone(result.skip_reason)
        self.assertIsNone(result.skip_detail)
        self.assert_context_schema(result)
        text = self.execution_ledger_text()
        self.assertIn("SIGNAL_GENERATED", text)
        self.assertNotIn("SIGNAL_SKIPPED", text)
        self.client.placeOrder.assert_not_called()

    def test_signal_events_route_to_execution_ledger(self) -> None:
        self.run_signal(self.engine())
        self.run_signal(self.engine(S02_CONFIG), vix=12.0, order_ref_candidate="S02|P|OPEN")
        order_text = (self.root / "data/ledger/order_ledger.jsonl").read_text()
        exec_text = self.execution_ledger_text()
        self.assertEqual(order_text, "")
        self.assertIn("SIGNAL_GENERATED", exec_text)
        self.assertIn("SIGNAL_SKIPPED", exec_text)

    def test_no_market_data_imports(self) -> None:
        for module_path in [
            "algo_trader_unified/strategies/vol/signals.py",
            "algo_trader_unified/strategies/vol/engine.py",
        ]:
            source = Path(module_path).read_text()
            self.assertNotIn("yfinance", source)
            self.assertNotIn("ib_insync", source)
            self.assertNotIn("requests", source)


class Phase2AProtocolTests(unittest.TestCase):
    def test_risk_manager_protocol_and_annotation(self) -> None:
        self.assertIsInstance(Phase2ARiskManagerStub(), RiskManagerProtocol)
        hints = get_type_hints(VolSellingEngine)
        self.assertIs(hints["risk_manager"], RiskManagerProtocol)
        signature = inspect.signature(VolSellingEngine.generate_standard_strangle_signal)
        self.assertIn("vix", signature.parameters)
        self.assertIn("iv_rank", signature.parameters)

    def test_no_submit_order_call_in_engine_source(self) -> None:
        source = Path("algo_trader_unified/strategies/vol/engine.py").read_text()
        self.assertNotIn("submit_order(", source)


class SpyLock:
    def __init__(self, name: str, acquisitions: list[str]) -> None:
        self.name = name
        self.acquisitions = acquisitions

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        self.acquisitions.append(self.name)
        return True

    def release(self) -> None:
        return None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


class StateStoreReadLockTests(VolCase):
    def test_has_existing_open_position_acquires_strategy_lock(self) -> None:
        acquisitions: list[str] = []
        self.state_store.strategy_state_locks[S01_CONFIG.strategy_id] = SpyLock(
            "strategy", acquisitions
        )
        has_existing_open_position(
            self.state_store,
            strategy_id=S01_CONFIG.strategy_id,
            symbol="XSP",
        )
        self.assertEqual(acquisitions, ["strategy"])

    def test_has_needs_reconciliation_acquires_strategy_lock(self) -> None:
        acquisitions: list[str] = []
        self.state_store.strategy_state_locks[S01_CONFIG.strategy_id] = SpyLock(
            "strategy", acquisitions
        )
        has_needs_reconciliation(
            self.state_store,
            strategy_id=S01_CONFIG.strategy_id,
            symbol="XSP",
        )
        self.assertEqual(acquisitions, ["strategy"])


class Phase2BLifecycleTests(VolCase):
    def install_spy_locks(self) -> list[str]:
        acquisitions: list[str] = []
        self.state_store.strategy_state_locks[S01_CONFIG.strategy_id] = SpyLock(
            "strategy", acquisitions
        )
        self.state_store._write_lock = SpyLock("state_write", acquisitions)
        return acquisitions

    def create_pending(self, engine: VolSellingEngine | None = None, position_id: str = "S01-POS-1"):
        engine = engine or self.engine()
        signal = self.clean_signal(engine)
        return engine.create_pending_position(
            signal_result=signal,
            position_id=position_id,
            leg_specs=self.leg_specs(),
        )

    def position(self, position_id: str = "S01-POS-1") -> dict:
        for position in self.state_store.state["positions"]:
            if position["position_id"] == position_id:
                return position
        raise AssertionError(f"missing position {position_id}")

    def test_create_pending_position_creates_record(self) -> None:
        record = self.create_pending()
        stored = self.position()
        self.assertEqual(record["status"], "pending_open")
        self.assertEqual(stored["strategy_id"], S01_CONFIG.strategy_id)
        self.assertEqual(stored["sleeve_id"], "VOL")
        self.assertEqual(stored["symbol"], "XSP")
        self.assertEqual(stored["position_id"], "S01-POS-1")
        self.assertEqual(stored["legs"], self.leg_specs())

    def test_create_pending_position_rejects_invalid_signal(self) -> None:
        engine = self.engine()
        invalid_signal = self.run_signal(engine, iv_rank=20.0)
        before_positions = deepcopy(self.state_store.state["positions"])
        before_ledger = self.execution_ledger_text()
        with self.assertRaises(InvalidSignalError):
            engine.create_pending_position(
                signal_result=invalid_signal,
                position_id="bad",
                leg_specs=self.leg_specs(),
            )
        self.assertEqual(before_positions, self.state_store.state["positions"])
        self.assertEqual(before_ledger, self.execution_ledger_text())
        self.client.placeOrder.assert_not_called()

    def test_create_pending_writes_position_adjusted_not_opened(self) -> None:
        self.create_pending()
        text = self.execution_ledger_text()
        self.assertIn("POSITION_ADJUSTED", text)
        self.assertIn("PENDING_OPEN_CREATED", text)
        self.assertNotIn("POSITION_OPENED", text)

    def test_create_pending_sets_execution_mode_fields(self) -> None:
        record = self.create_pending()
        self.assertEqual(record["execution_mode_at_entry"], S01_CONFIG.execution_mode)
        self.assertEqual(record["managed_under_mode"], S01_CONFIG.execution_mode)

    def test_mark_position_open_transition(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        record = engine.mark_position_open(position_id="S01-POS-1")
        self.assertEqual(record["status"], "open")
        self.assertEqual(self.position()["status"], "open")
        self.assertIn("POSITION_OPENED", self.execution_ledger_text())
        self.client.placeOrder.assert_not_called()

    def test_mark_position_pending_close_transition(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        engine.mark_position_open(position_id="S01-POS-1")
        record = engine.mark_position_pending_close(position_id="S01-POS-1")
        self.assertEqual(record["status"], "pending_close")
        text = self.execution_ledger_text()
        self.assertIn("POSITION_ADJUSTED", text)
        self.assertIn("PENDING_CLOSE_CREATED", text)
        self.client.placeOrder.assert_not_called()

    def test_record_close_transition_from_pending_close(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        engine.mark_position_open(position_id="S01-POS-1")
        engine.mark_position_pending_close(position_id="S01-POS-1")
        record = engine.record_close(
            position_id="S01-POS-1",
            close_reason="unit_test_close",
            realized_pnl=123.45,
        )
        self.assertEqual(record["status"], "closed")
        self.assertIn("closed_at", record)
        self.assertEqual(record["realized_pnl"], 123.45)
        self.assertEqual(record["lifecycle_reason"], "unit_test_close")
        self.assertIn("POSITION_CLOSED", self.execution_ledger_text())
        self.client.placeOrder.assert_not_called()

    def test_record_close_transition_from_open(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        engine.mark_position_open(position_id="S01-POS-1")
        record = engine.record_close(
            position_id="S01-POS-1",
            close_reason="direct_close",
            realized_pnl=-10.0,
        )
        self.assertEqual(record["status"], "closed")
        self.assertEqual(record["realized_pnl"], -10.0)

    def test_execution_mode_fields_unchanged_full_lifecycle(self) -> None:
        engine = self.engine()
        pending = self.create_pending(engine)
        entry_mode = pending["execution_mode_at_entry"]
        managed_mode = pending["managed_under_mode"]
        opened = engine.mark_position_open(position_id="S01-POS-1")
        pending_close = engine.mark_position_pending_close(position_id="S01-POS-1")
        closed = engine.record_close(
            position_id="S01-POS-1",
            close_reason="unit_test",
            realized_pnl=0.0,
        )
        for record in (opened, pending_close, closed):
            self.assertEqual(record["execution_mode_at_entry"], entry_mode)
            self.assertEqual(record["managed_under_mode"], managed_mode)

    def test_lock_order_create_pending(self) -> None:
        engine = self.engine()
        signal = self.clean_signal(engine)
        acquisitions = self.install_spy_locks()
        engine.create_pending_position(
            signal_result=signal,
            position_id="S01-POS-1",
            leg_specs=self.leg_specs(),
        )
        self.assertEqual(acquisitions[:2], ["strategy", "state_write"])

    def test_lock_order_mark_position_open(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        acquisitions = self.install_spy_locks()
        engine.mark_position_open(position_id="S01-POS-1")
        self.assertEqual(acquisitions[:2], ["strategy", "state_write"])

    def test_lock_order_mark_position_pending_close(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        engine.mark_position_open(position_id="S01-POS-1")
        acquisitions = self.install_spy_locks()
        engine.mark_position_pending_close(position_id="S01-POS-1")
        self.assertEqual(acquisitions[:2], ["strategy", "state_write"])

    def test_lock_order_record_close(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        engine.mark_position_open(position_id="S01-POS-1")
        acquisitions = self.install_spy_locks()
        engine.record_close(
            position_id="S01-POS-1",
            close_reason="unit_test",
            realized_pnl=0.0,
        )
        self.assertEqual(acquisitions[:2], ["strategy", "state_write"])

    def test_lifecycle_events_route_to_execution_ledger_only(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        engine.mark_position_open(position_id="S01-POS-1")
        engine.mark_position_pending_close(position_id="S01-POS-1")
        engine.record_close(
            position_id="S01-POS-1",
            close_reason="unit_test",
            realized_pnl=0.0,
        )
        order_text = (self.root / "data/ledger/order_ledger.jsonl").read_text()
        exec_text = self.execution_ledger_text()
        self.assertEqual(order_text, "")
        self.assertIn("POSITION_ADJUSTED", exec_text)
        self.assertIn("POSITION_OPENED", exec_text)
        self.assertIn("POSITION_CLOSED", exec_text)

    def test_no_direct_file_writes_in_vol_lifecycle(self) -> None:
        for module_path in [
            "algo_trader_unified/strategies/vol/engine.py",
            "algo_trader_unified/strategies/vol/signals.py",
            "algo_trader_unified/tools/halt.py",
            "algo_trader_unified/tools/resume_halt.py",
            "algo_trader_unified/tools/reconcile_check.py",
            "algo_trader_unified/core/reconciliation.py",
        ]:
            source = Path(module_path).read_text()
            if "strategies/vol" in module_path:
                self.assertNotIn(".jsonl", source, module_path)
                self.assertNotIn("write_text", source, module_path)
                self.assertNotIn(".open(", source, module_path)
                self.assertIsNone(re.search(r"(?<![\w.])open\(", source), module_path)
            else:
                self.assertNotIn(".jsonl", source, module_path)
                self.assertIsNone(re.search(r"(?<![\w.])open\(", source), module_path)

    def assert_invalid_transition_no_side_effects(self, action) -> None:
        before_state = deepcopy(self.state_store.state)
        before_ledger = self.execution_ledger_text()
        with self.assertRaises(LifecycleTransitionError):
            action()
        self.assertEqual(before_state, self.state_store.state)
        self.assertEqual(before_ledger, self.execution_ledger_text())
        self.client.placeOrder.assert_not_called()

    def test_mark_position_open_invalid_status(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        self.state_store.state["positions"][0]["status"] = "open"
        self.assert_invalid_transition_no_side_effects(
            lambda: engine.mark_position_open(position_id="S01-POS-1")
        )

    def test_mark_position_pending_close_invalid_status(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        self.assert_invalid_transition_no_side_effects(
            lambda: engine.mark_position_pending_close(position_id="S01-POS-1")
        )

    def test_record_close_invalid_pending_open_and_closed(self) -> None:
        engine = self.engine()
        self.create_pending(engine)
        self.assert_invalid_transition_no_side_effects(
            lambda: engine.record_close(
                position_id="S01-POS-1",
                close_reason="invalid",
                realized_pnl=0.0,
            )
        )
        engine.mark_position_open(position_id="S01-POS-1")
        engine.record_close(
            position_id="S01-POS-1",
            close_reason="valid",
            realized_pnl=0.0,
        )
        self.assert_invalid_transition_no_side_effects(
            lambda: engine.record_close(
                position_id="S01-POS-1",
                close_reason="invalid_again",
                realized_pnl=0.0,
            )
        )


class Phase2CManagementSignalTests(VolCase):
    def management(self, **overrides):
        kwargs = {
            "position_id": "S01-POS-1",
            "current_date": date(2026, 5, 1),
            "entry_date": date(2026, 4, 1),
            "expiry": date(2026, 6, 20),
            "entry_credit": 2.0,
            "current_mark_to_close": 1.2,
            "manual_close_requested": False,
        }
        kwargs.update(overrides)
        return self.engine().evaluate_management_signal(**kwargs)

    def test_hold_writes_no_event(self) -> None:
        before = self.execution_ledger_text()
        result = self.management()
        self.assertEqual(result.action, ACTION_HOLD)
        self.assertFalse(result.should_close)
        self.assertEqual(before, self.execution_ledger_text())

    def test_profit_target_signal(self) -> None:
        result = self.management(current_mark_to_close=1.0)
        self.assertEqual(result.action, ACTION_CLOSE_PROFIT_TARGET)
        self.assertTrue(result.should_close)
        text = self.execution_ledger_text()
        self.assertIn("SIGNAL_GENERATED", text)
        self.assertIn("MANAGEMENT_CLOSE_SIGNAL", text)

    def test_stop_loss_signal(self) -> None:
        result = self.management(current_mark_to_close=4.0)
        self.assertEqual(result.action, ACTION_CLOSE_STOP_LOSS)
        self.assertTrue(result.should_close)
        self.assertIn("SIGNAL_GENERATED", self.execution_ledger_text())

    def test_dte_close_signal(self) -> None:
        result = self.management(expiry=date(2026, 5, 15))
        self.assertEqual(result.action, ACTION_CLOSE_DTE)
        self.assertTrue(result.should_close)
        self.assertIn("SIGNAL_GENERATED", self.execution_ledger_text())

    def test_manual_close_signal(self) -> None:
        result = self.management(manual_close_requested=True)
        self.assertEqual(result.action, ACTION_CLOSE_MANUAL)
        self.assertTrue(result.should_close)
        self.assertIn("SIGNAL_GENERATED", self.execution_ledger_text())

    def test_priority_order(self) -> None:
        self.assertEqual(
            self.management(manual_close_requested=True, current_mark_to_close=4.0).action,
            ACTION_CLOSE_MANUAL,
        )
        self.assertEqual(
            self.management(entry_credit=2.0, current_mark_to_close=4.0).action,
            ACTION_CLOSE_STOP_LOSS,
        )
        self.assertEqual(
            self.management(current_mark_to_close=1.0, expiry=date(2026, 5, 15)).action,
            ACTION_CLOSE_PROFIT_TARGET,
        )
        self.assertEqual(
            self.management(current_mark_to_close=1.2, expiry=date(2026, 5, 15)).action,
            ACTION_CLOSE_DTE,
        )

    def test_params_present_and_used_in_context(self) -> None:
        for config in (S01_CONFIG, S02_CONFIG):
            self.assertIn("dte_close_threshold", config.params)
            engine = self.engine(config)
            result = engine.evaluate_management_signal(
                position_id="P",
                current_date=date(2026, 5, 1),
                entry_date=date(2026, 4, 1),
                expiry=date(2026, 6, 20),
                entry_credit=2.0,
                current_mark_to_close=1.2,
            )
            self.assertEqual(result.context["profit_target_pct"], config.params["profit_target_pct"])
            self.assertEqual(result.context["stop_loss_mult"], config.params["stop_loss_mult"])
            self.assertEqual(result.context["dte_close_threshold"], config.params["dte_close_threshold"])
        source = Path("algo_trader_unified/strategies/vol/signals.py").read_text()
        self.assertIn('config.params["profit_target_pct"]', source)
        self.assertIn('config.params["stop_loss_mult"]', source)
        self.assertIn('config.params["dte_close_threshold"]', source)

    def test_management_ledger_routing_and_no_forbidden_events(self) -> None:
        self.management(manual_close_requested=True)
        order_text = (self.root / "data/ledger/order_ledger.jsonl").read_text()
        exec_text = self.execution_ledger_text()
        self.assertEqual(order_text, "")
        self.assertIn("SIGNAL_GENERATED", exec_text)
        self.assertNotIn("POSITION_CLOSED", exec_text)
        self.assertNotIn("POSITION_ADJUSTED", exec_text)
        self.assertNotIn("ORDER_SUBMITTED", exec_text)
        self.assertNotIn("ORDER_CONFIRMED", exec_text)
        self.assertNotIn("FILL_CONFIRMED", exec_text)

    def test_no_state_store_mutation(self) -> None:
        before = deepcopy(self.state_store.state)
        self.management()
        self.management(manual_close_requested=True)
        self.assertEqual(before, self.state_store.state)

    def test_no_broker_calls(self) -> None:
        self.management(manual_close_requested=True)
        self.client.placeOrder.assert_not_called()
        self.client.cancelOrder.assert_not_called()
        self.broker.submit_order = mock.Mock()
        self.management(current_mark_to_close=4.0)
        self.broker.submit_order.assert_not_called()

    def test_management_payload_fields(self) -> None:
        self.management(manual_close_requested=True)
        payload = json_loads_last_line(self.execution_ledger_text())["payload"]
        self.assertEqual(payload["event_detail"], "MANAGEMENT_CLOSE_SIGNAL")
        for key in (
            "position_id",
            "strategy_id",
            "action",
            "reason",
            "entry_credit",
            "current_mark_to_close",
            "days_to_expiry",
        ):
            self.assertIn(key, payload)

    def test_management_input_errors_have_no_side_effects(self) -> None:
        for overrides in (
            {"entry_credit": 0},
            {"entry_credit": -1},
            {"current_mark_to_close": -0.01},
        ):
            with self.subTest(overrides=overrides):
                before_state = deepcopy(self.state_store.state)
                before_ledger = self.execution_ledger_text()
                with self.assertRaises(ManagementInputError):
                    self.management(**overrides)
                self.assertEqual(before_state, self.state_store.state)
                self.assertEqual(before_ledger, self.execution_ledger_text())
                self.client.placeOrder.assert_not_called()

    def test_should_close_invariant_for_all_actions(self) -> None:
        cases = [
            self.management(),
            self.management(manual_close_requested=True),
            self.management(current_mark_to_close=4.0),
            self.management(current_mark_to_close=1.0),
            self.management(expiry=date(2026, 5, 15)),
        ]
        for result in cases:
            self.assertEqual(result.should_close, result.action != ACTION_HOLD)

    def test_expired_dte_closes(self) -> None:
        result = self.management(expiry=date(2026, 4, 30))
        self.assertLess(result.days_to_expiry, 0)
        self.assertEqual(result.action, ACTION_CLOSE_DTE)
        self.assertTrue(result.should_close)
        text = self.execution_ledger_text()
        self.assertIn("SIGNAL_GENERATED", text)
        self.assertIn("MANAGEMENT_CLOSE_SIGNAL", text)

    def test_entry_date_reserved_context_and_no_priority_effect(self) -> None:
        older = self.management(
            entry_date=date(2026, 1, 1),
            manual_close_requested=True,
            current_mark_to_close=4.0,
        )
        newer = self.management(
            entry_date=date(2026, 4, 20),
            manual_close_requested=True,
            current_mark_to_close=4.0,
        )
        self.assertEqual(older.context["entry_date"], "2026-01-01")
        self.assertEqual(newer.context["entry_date"], "2026-04-20")
        self.assertEqual(older.action, newer.action)
        self.assertEqual(older.action, ACTION_CLOSE_MANUAL)


class Phase2DOrderManagerTests(VolCase):
    def setUp(self) -> None:
        super().setUp()
        self.submit_order_patch = mock.patch.object(
            self.broker,
            "submit_order",
            wraps=self.broker.submit_order,
        )
        self.submit_order_mock = self.submit_order_patch.start()
        self.addCleanup(self.submit_order_patch.stop)
        self.engine_obj = self.engine()
        self.order_manager = VolOrderManager(
            config=S01_CONFIG,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=self.broker,
            engine=self.engine_obj,
        )

    def create_open_position(self, position_id: str = "S01-POS-1") -> None:
        signal = self.clean_signal(self.engine_obj)
        self.engine_obj.create_pending_position(
            signal_result=signal,
            position_id=position_id,
            leg_specs=self.leg_specs(),
        )
        self.engine_obj.mark_position_open(position_id=position_id)

    def close_result(self, **overrides):
        kwargs = {
            "position_id": "S01-POS-1",
            "current_date": date(2026, 5, 1),
            "entry_date": date(2026, 4, 1),
            "expiry": date(2026, 6, 20),
            "entry_credit": 2.0,
            "current_mark_to_close": 4.0,
        }
        kwargs.update(overrides)
        return self.engine_obj.evaluate_management_signal(**kwargs)

    def execute_close(self, management_result=None, order_ref: str | None = "S01|CLOSE|1"):
        return self.order_manager.execute_close(
            management_result=management_result or self.close_result(),
            order_ref=order_ref,
            intent_id="intent-1",
        )

    def order_ledger_text(self) -> str:
        return (self.root / "data/ledger/order_ledger.jsonl").read_text()

    def test_execute_close_rejects_hold(self) -> None:
        self.create_open_position()
        hold = self.close_result(current_mark_to_close=1.2)
        before_state = deepcopy(self.state_store.state)
        before_order = self.order_ledger_text()
        with self.assertRaises(InvalidManagementResultError):
            self.execute_close(hold)
        self.assertEqual(before_state, self.state_store.state)
        self.assertEqual(before_order, self.order_ledger_text())
        self.client.placeOrder.assert_not_called()

    def test_execute_close_rejects_missing_order_ref(self) -> None:
        self.create_open_position()
        for order_ref in (None, "", "   "):
            with self.subTest(order_ref=order_ref):
                before_state = deepcopy(self.state_store.state)
                before_order = self.order_ledger_text()
                with self.assertRaises(MissingOrderRefError):
                    self.execute_close(order_ref=order_ref)
                self.assertEqual(before_state, self.state_store.state)
                self.assertEqual(before_order, self.order_ledger_text())
                self.client.placeOrder.assert_not_called()

    def test_execute_close_stale_status(self) -> None:
        self.create_open_position()
        self.state_store.state["positions"][0]["status"] = "pending_close"
        management = self.close_result()
        before_state = deepcopy(self.state_store.state)
        before_exec = self.execution_ledger_text()
        before_order = self.order_ledger_text()
        with mock.patch.object(
            self.engine_obj,
            "mark_position_pending_close",
            wraps=self.engine_obj.mark_position_pending_close,
        ) as pending_close:
            with self.assertRaises(StaleManagementResultError):
                self.execute_close(management)
        pending_close.assert_not_called()
        self.assertEqual(before_state, self.state_store.state)
        self.assertEqual(before_exec, self.execution_ledger_text())
        self.assertEqual(before_order, self.order_ledger_text())
        self.assertNotIn("ORDER_SUBMITTED", self.order_ledger_text())
        self.assertNotIn("POSITION_ADJUSTED", self.execution_ledger_text()[len(before_exec):])
        self.submit_order_mock.assert_not_called()
        self.client.placeOrder.assert_not_called()

    def test_execute_close_stale_closed_status(self) -> None:
        self.create_open_position()
        management = self.close_result()
        self.state_store.state["positions"][0]["status"] = "closed"
        before_state = deepcopy(self.state_store.state)
        before_exec = self.execution_ledger_text()
        before_order = self.order_ledger_text()
        with self.assertRaises(StaleManagementResultError):
            self.execute_close(management)
        self.assertEqual(before_state, self.state_store.state)
        self.assertEqual(before_exec, self.execution_ledger_text())
        self.assertEqual(before_order, self.order_ledger_text())
        self.assertNotIn("ORDER_SUBMITTED", self.order_ledger_text())
        self.assertNotIn("POSITION_ADJUSTED", self.execution_ledger_text()[len(before_exec):])
        self.submit_order_mock.assert_not_called()

    def test_execute_close_success(self) -> None:
        self.create_open_position()
        management = self.close_result()
        with mock.patch.object(
            self.engine_obj,
            "mark_position_pending_close",
            wraps=self.engine_obj.mark_position_pending_close,
        ) as pending_close:
            result = self.execute_close(management)
        pending_close.assert_called_once()
        position = self.state_store.state["positions"][0]
        self.assertEqual(position["status"], "pending_close")
        self.assertEqual(position["source_management_action"], management.action)
        self.assertEqual(position["close_reason"], management.reason)
        self.assertEqual(position["close_intent_id"], "intent-1")
        self.assertEqual(position["close_order_ref"], "S01|CLOSE|1")
        exec_text = self.execution_ledger_text()
        order_text = self.order_ledger_text()
        self.assertIn("POSITION_ADJUSTED", exec_text)
        self.assertIn("PENDING_CLOSE_CREATED", exec_text)
        self.assertIn("ORDER_SUBMITTED", order_text)
        self.assertEqual(result.status, "close_intent_created")
        self.assertEqual(result.source_management_action, management.action)
        self.assertTrue(result.pending_close_created)
        self.assertIsNotNone(result.order_submitted_event_id)
        self.submit_order_mock.assert_not_called()
        self.client.placeOrder.assert_not_called()

    def test_order_submitted_payload_fields(self) -> None:
        self.create_open_position()
        self.execute_close()
        payload = json_loads_last_line(self.order_ledger_text())["payload"]
        for key in (
            "intent_id",
            "position_id",
            "strategy_id",
            "sleeve_id",
            "symbol",
            "action",
            "reason",
            "order_ref",
            "dry_run",
            "source_management_action",
            "current_mark_to_close",
            "entry_credit",
            "days_to_expiry",
        ):
            self.assertIn(key, payload)
        self.assertTrue(payload["dry_run"])

    def test_close_order_intent_payload_matches_order_submitted(self) -> None:
        management = self.close_result()
        intent = self.order_manager._build_close_intent(
            management_result=management,
            order_ref="S01|CLOSE|1",
            intent_id="intent-1",
            current_time=None,
        )
        self.create_open_position()
        self.execute_close(management)
        ledger_payload = json_loads_last_line(self.order_ledger_text())["payload"]
        for key, value in intent.payload.items():
            self.assertEqual(ledger_payload[key], value)

    def test_routing_and_no_forbidden_events(self) -> None:
        self.create_open_position()
        self.execute_close()
        order_text = self.order_ledger_text()
        exec_text = self.execution_ledger_text()
        self.assertIn("ORDER_SUBMITTED", order_text)
        self.assertNotIn("ORDER_SUBMITTED", exec_text)
        self.assertIn("POSITION_ADJUSTED", exec_text)
        self.assertNotIn("POSITION_ADJUSTED", order_text)
        self.assertNotIn("ORDER_CONFIRMED", order_text + exec_text)
        self.assertNotIn("FILL_CONFIRMED", order_text + exec_text)
        self.assertNotIn("POSITION_CLOSED", order_text + exec_text)

    def test_toctou_recheck_rejects_changed_status(self) -> None:
        self.create_open_position()
        management = self.close_result()
        self.state_store.state["positions"][0]["status"] = "pending_close"
        before_state = deepcopy(self.state_store.state)
        before_exec = self.execution_ledger_text()
        before_order = self.order_ledger_text()
        with self.assertRaises(StaleManagementResultError):
            self.execute_close(management)
        self.assertEqual(before_state, self.state_store.state)
        self.assertEqual(before_exec, self.execution_ledger_text())
        self.assertEqual(before_order, self.order_ledger_text())
        self.assertNotIn("ORDER_SUBMITTED", self.order_ledger_text())
        self.assertNotIn("POSITION_ADJUSTED", self.execution_ledger_text()[len(before_exec):])
        self.submit_order_mock.assert_not_called()

    def test_diagnostic_client_rejected_before_order_submitted(self) -> None:
        self.create_open_position()
        manager = VolOrderManager(
            config=S01_CONFIG,
            state_store=self.state_store,
            ledger=self.ledger,
            broker=self.broker,
            engine=self.engine_obj,
            client_id=95,
        )
        management = self.close_result()
        before_state = deepcopy(self.state_store.state)
        before_exec = self.execution_ledger_text()
        before_order = self.order_ledger_text()
        with self.assertRaises(DiagnosticClientOrderError):
            manager.execute_close(
                management_result=management,
                order_ref="S01|CLOSE|1",
                intent_id="intent-1",
            )
        self.assertEqual(before_state, self.state_store.state)
        self.assertEqual(before_exec, self.execution_ledger_text())
        self.assertEqual(before_order, self.order_ledger_text())
        self.submit_order_mock.assert_not_called()
        self.client.placeOrder.assert_not_called()

    def test_execute_close_position_not_found_no_side_effects(self) -> None:
        management = self.close_result(position_id="missing-position")
        before_state = deepcopy(self.state_store.state)
        before_exec = self.execution_ledger_text()
        before_order = self.order_ledger_text()
        with self.assertRaises(StaleManagementResultError):
            self.execute_close(management)
        self.assertEqual(before_state, self.state_store.state)
        self.assertEqual(before_exec, self.execution_ledger_text())
        self.assertEqual(before_order, self.order_ledger_text())
        self.assertNotIn("ORDER_SUBMITTED", self.order_ledger_text())
        self.assertNotIn("POSITION_ADJUSTED", self.execution_ledger_text()[len(before_exec):])
        self.submit_order_mock.assert_not_called()
        self.client.placeOrder.assert_not_called()

    def test_no_engine_order_layer_inversion(self) -> None:
        source = Path("algo_trader_unified/strategies/vol/engine.py").read_text()
        self.assertNotIn("from algo_trader_unified.strategies.vol.order_manager", source)
        self.assertNotIn("VolOrderManager(", source)
        self.assertNotIn("CloseOrderIntent", source)
        self.assertNotIn("CloseExecutionResult", source)
        self.assertNotIn("execute_close", source)

    def test_order_manager_no_direct_file_writes(self) -> None:
        source = Path("algo_trader_unified/strategies/vol/order_manager.py").read_text()
        self.assertNotIn(".jsonl", source)
        self.assertNotIn("write_text", source)
        self.assertNotIn(".open(", source)
        self.assertNotIn("state_store.save(", source)
        self.assertIsNone(re.search(r"(?<![\w.])open\(", source))


class Phase2ECloseFillConfirmationTests(Phase2DOrderManagerTests):
    def position(self, position_id: str = "S01-POS-1") -> dict:
        for position in self.state_store.state["positions"]:
            if position["position_id"] == position_id:
                return position
        raise AssertionError(f"missing position {position_id}")

    def pending_close_position(self) -> None:
        self.create_open_position()
        self.execute_close()

    def fill_kwargs(self, **overrides):
        kwargs = {
            "position_id": "S01-POS-1",
            "order_ref": "S01|CLOSE|1",
            "realized_pnl": 125.5,
            "fill_price": 1.25,
            "fill_time": "2026-05-01T15:45:00+00:00",
            "fill_id": "fill-1",
        }
        kwargs.update(overrides)
        return kwargs

    def confirm_close_fill(self, **overrides):
        return self.order_manager.confirm_close_fill(**self.fill_kwargs(**overrides))

    def assert_no_fill_or_close_mutation(self, before_state, before_exec, before_order) -> None:
        self.assertEqual(before_state, self.state_store.state)
        self.assertEqual(before_exec, self.execution_ledger_text())
        self.assertEqual(before_order, self.order_ledger_text())
        self.assertNotIn("FILL_CONFIRMED", self.order_ledger_text())
        self.assertNotIn("POSITION_CLOSED", self.execution_ledger_text())
        self.submit_order_mock.assert_not_called()
        self.client.placeOrder.assert_not_called()

    def test_confirm_close_fill_rejects_missing_position(self) -> None:
        before_state = deepcopy(self.state_store.state)
        before_exec = self.execution_ledger_text()
        before_order = self.order_ledger_text()
        with mock.patch.object(self.engine_obj, "record_close") as record_close:
            with self.assertRaises(StaleCloseFillError):
                self.confirm_close_fill(position_id="missing-position")
        record_close.assert_not_called()
        self.assert_no_fill_or_close_mutation(before_state, before_exec, before_order)

    def test_confirm_close_fill_rejects_non_pending_close_status(self) -> None:
        self.create_open_position()
        for status in ("open", "pending_open", "closed"):
            with self.subTest(status=status):
                self.position()["status"] = status
                before_state = deepcopy(self.state_store.state)
                before_exec = self.execution_ledger_text()
                before_order = self.order_ledger_text()
                with mock.patch.object(self.engine_obj, "record_close") as record_close:
                    with self.assertRaises(StaleCloseFillError):
                        self.confirm_close_fill()
                record_close.assert_not_called()
                self.assert_no_fill_or_close_mutation(before_state, before_exec, before_order)

    def test_confirm_close_fill_rejects_missing_order_ref(self) -> None:
        self.pending_close_position()
        for order_ref in (None, "", "   "):
            with self.subTest(order_ref=order_ref):
                before_state = deepcopy(self.state_store.state)
                before_exec = self.execution_ledger_text()
                before_order = self.order_ledger_text()
                with mock.patch.object(self.engine_obj, "record_close") as record_close:
                    with self.assertRaises(MissingOrderRefError):
                        self.confirm_close_fill(order_ref=order_ref)
                record_close.assert_not_called()
                self.assert_no_fill_or_close_mutation(before_state, before_exec, before_order)

    def test_confirm_close_fill_rejects_order_ref_mismatch(self) -> None:
        self.pending_close_position()
        self.state_store.state["positions"][0]["close_order_ref"] = "A"
        before_state = deepcopy(self.state_store.state)
        before_exec = self.execution_ledger_text()
        before_order = self.order_ledger_text()
        with mock.patch.object(self.engine_obj, "record_close") as record_close:
            with self.assertRaises(CloseFillValidationError):
                self.confirm_close_fill(order_ref="B")
        record_close.assert_not_called()
        self.assert_no_fill_or_close_mutation(before_state, before_exec, before_order)

    def test_confirm_close_fill_rejects_absent_close_order_ref(self) -> None:
        self.pending_close_position()
        del self.state_store.state["positions"][0]["close_order_ref"]
        before_state = deepcopy(self.state_store.state)
        before_exec = self.execution_ledger_text()
        before_order = self.order_ledger_text()
        with mock.patch.object(self.engine_obj, "record_close") as record_close:
            with self.assertRaises(CloseFillValidationError):
                self.confirm_close_fill(order_ref="S01|CLOSE|1")
        record_close.assert_not_called()
        self.assert_no_fill_or_close_mutation(before_state, before_exec, before_order)

    def test_confirm_close_fill_rejects_invalid_fill_data(self) -> None:
        cases = (
            {"fill_price": -0.01},
            {"fill_time": None},
            {"fill_time": ""},
            {"realized_pnl": "not-a-number"},
        )
        self.pending_close_position()
        for overrides in cases:
            with self.subTest(overrides=overrides):
                before_state = deepcopy(self.state_store.state)
                before_exec = self.execution_ledger_text()
                before_order = self.order_ledger_text()
                with mock.patch.object(self.engine_obj, "record_close") as record_close:
                    with self.assertRaises(CloseFillValidationError):
                        self.confirm_close_fill(**overrides)
                record_close.assert_not_called()
                self.assert_no_fill_or_close_mutation(before_state, before_exec, before_order)

    def test_confirm_close_fill_rejects_missing_close_metadata(self) -> None:
        self.pending_close_position()
        original_position = deepcopy(self.state_store.state["positions"][0])
        for missing_key in ("source_management_action", "close_reason"):
            with self.subTest(missing_key=missing_key):
                self.state_store.state["positions"][0] = deepcopy(original_position)
                del self.state_store.state["positions"][0][missing_key]
                before_state = deepcopy(self.state_store.state)
                before_exec = self.execution_ledger_text()
                before_order = self.order_ledger_text()
                with mock.patch.object(self.engine_obj, "record_close") as record_close:
                    with self.assertRaises(CloseFillValidationError):
                        self.confirm_close_fill()
                record_close.assert_not_called()
                self.assert_no_fill_or_close_mutation(before_state, before_exec, before_order)

    def test_confirm_close_fill_success(self) -> None:
        self.pending_close_position()
        with mock.patch.object(
            self.engine_obj,
            "record_close",
            wraps=self.engine_obj.record_close,
        ) as record_close:
            result = self.confirm_close_fill()
        record_close.assert_called_once()
        position = self.state_store.state["positions"][0]
        self.assertEqual(position["status"], "closed")
        self.assertIn("closed_at", position)
        self.assertEqual(position["realized_pnl"], 125.5)
        self.assertEqual(position["lifecycle_reason"], position["close_reason"])
        self.assertEqual(result.status, "close_confirmed")
        self.assertIsNotNone(result.fill_confirmed_event_id)
        self.assertIsNotNone(result.position_closed_event_id)
        self.assertEqual(result.close_reason, position["close_reason"])
        order_text = self.order_ledger_text()
        exec_text = self.execution_ledger_text()
        self.assertIn("FILL_CONFIRMED", order_text)
        self.assertIn("POSITION_CLOSED", exec_text)
        self.assertNotIn("FILL_CONFIRMED", exec_text)
        self.assertNotIn("POSITION_CLOSED", order_text)
        self.submit_order_mock.assert_not_called()
        self.client.placeOrder.assert_not_called()

    def test_confirm_close_fill_payload_fields(self) -> None:
        self.pending_close_position()
        self.confirm_close_fill()
        order_events = json_loads_lines(self.order_ledger_text())
        fill_events = [
            event for event in order_events if event["event_type"] == "FILL_CONFIRMED"
        ]
        self.assertEqual(len(fill_events), 1)
        payload = fill_events[0]["payload"]
        for key in (
            "position_id",
            "strategy_id",
            "sleeve_id",
            "symbol",
            "order_ref",
            "fill_id",
            "fill_price",
            "fill_time",
            "realized_pnl",
            "source_management_action",
            "close_reason",
        ):
            self.assertIn(key, payload)
        order_text = self.order_ledger_text()
        exec_text = self.execution_ledger_text()
        self.assertIn("FILL_CONFIRMED", order_text)
        self.assertNotIn("FILL_CONFIRMED", exec_text)
        self.assertNotIn("ORDER_CONFIRMED", order_text + exec_text)

    def test_full_close_round_trip_ledger_routing(self) -> None:
        self.pending_close_position()
        self.confirm_close_fill()
        order_events = [event["event_type"] for event in json_loads_lines(self.order_ledger_text())]
        exec_events = [event["event_type"] for event in json_loads_lines(self.execution_ledger_text())]
        self.assertIn("ORDER_SUBMITTED", order_events)
        self.assertIn("FILL_CONFIRMED", order_events)
        self.assertNotIn("POSITION_ADJUSTED", order_events)
        self.assertNotIn("POSITION_CLOSED", order_events)
        self.assertIn("POSITION_ADJUSTED", exec_events)
        self.assertIn("POSITION_CLOSED", exec_events)
        self.assertNotIn("ORDER_SUBMITTED", exec_events)
        self.assertNotIn("FILL_CONFIRMED", exec_events)
        self.assertNotIn("ORDER_CONFIRMED", order_events + exec_events)
        self.assertNotIn("LIVE_BROKER_EVENT", order_events + exec_events)

    def test_confirm_close_fill_no_broker_calls(self) -> None:
        self.pending_close_position()
        self.confirm_close_fill()
        self.submit_order_mock.assert_not_called()
        self.client.placeOrder.assert_not_called()
        self.client.cancelOrder.assert_not_called()


def json_loads_last_line(text: str) -> dict:
    import json

    return json.loads(text.strip().splitlines()[-1])


def json_loads_lines(text: str) -> list[dict]:
    import json

    return [json.loads(line) for line in text.splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
