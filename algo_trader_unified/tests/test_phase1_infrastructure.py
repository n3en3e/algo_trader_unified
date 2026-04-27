from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S02_VOL_ENHANCED
from algo_trader_unified.core.broker import (
    DiagnosticClientOrderError,
    IBKRBrokerWrapper,
    MissingOrderRefError,
)
from algo_trader_unified.core.ledger import (
    LedgerAppender,
    LedgerInitError,
    LedgerValidationError,
)
from algo_trader_unified.core.reconciliation import reconcile_check
from algo_trader_unified.core.state_store import StateStore, StateStoreCorruptError


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def source_path(*parts: str) -> Path:
    return PACKAGE_ROOT.joinpath(*parts)


class TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()


class LedgerTests(TmpCase):
    def test_routing(self) -> None:
        ledger = LedgerAppender(self.root)
        order_event = ledger.append(
            event_type="ORDER_SUBMITTED",
            strategy_id="S01_VOL_BASELINE",
            execution_mode="paper_only",
            source_module="test",
            payload={"orderRef": "S01|P1|OPEN"},
        )
        position_event = ledger.append(
            event_type="POSITION_OPENED",
            strategy_id="S01_VOL_BASELINE",
            execution_mode="paper_only",
            source_module="test",
            payload={},
        )
        resume_event = ledger.append(
            event_type="HALT_RESUMED",
            strategy_id="ACCOUNT",
            execution_mode="disabled",
            source_module="test",
            payload={"scope": "account"},
        )
        recon_event = ledger.append(
            event_type="RECONCILIATION_FAILED",
            strategy_id="ACCOUNT",
            execution_mode="disabled",
            source_module="test",
            payload={},
        )
        order_lines = (self.root / "data/ledger/order_ledger.jsonl").read_text()
        exec_lines = (self.root / "data/ledger/execution_ledger.jsonl").read_text()
        self.assertIn(order_event.event_id, order_lines)
        self.assertIn(position_event.event_id, exec_lines)
        self.assertIn(resume_event.event_id, exec_lines)
        self.assertIn(recon_event.event_id, exec_lines)

    def test_wrong_routing_raises_before_write(self) -> None:
        ledger = LedgerAppender(self.root)
        order_path = self.root / "data/ledger/order_ledger.jsonl"
        before = order_path.read_text()
        with self.assertRaises(LedgerValidationError):
            ledger.append(
                event_type="ORDER_SUBMITTED",
                strategy_id="S01_VOL_BASELINE",
                execution_mode="paper_only",
                source_module="test",
                payload={},
                expected_ledger="execution_ledger.jsonl",
            )
        self.assertEqual(before, order_path.read_text())

    def test_unknown_event_type_raises_before_write(self) -> None:
        ledger = LedgerAppender(self.root)
        exec_path = self.root / "data/ledger/execution_ledger.jsonl"
        before = exec_path.read_text()
        with self.assertRaises(LedgerValidationError):
            ledger.append(
                event_type="NOT_AN_EVENT",
                strategy_id="S01_VOL_BASELINE",
                execution_mode="paper_only",
                source_module="test",
                payload={},
            )
        self.assertEqual(before, exec_path.read_text())

    def test_append_does_not_read_existing_file(self) -> None:
        ledger = LedgerAppender(self.root)
        real_open = builtins.open

        def guarded_open(file, mode="r", *args, **kwargs):
            if str(file).endswith(".jsonl") and "r" in mode:
                raise AssertionError("ledger read attempted")
            return real_open(file, mode, *args, **kwargs)

        with mock.patch("builtins.open", guarded_open):
            ledger.append(
                event_type="POSITION_OPENED",
                strategy_id="S01_VOL_BASELINE",
                execution_mode="paper_only",
                source_module="test",
                payload={},
            )

    def test_init_failure(self) -> None:
        ledger_dir = self.root / "data" / "ledger"
        ledger_dir.parent.mkdir(parents=True)
        ledger_dir.write_text("not a dir")
        with self.assertRaises(LedgerInitError):
            LedgerAppender(self.root)


class StateStoreTests(TmpCase):
    def test_fresh_schema_and_readiness(self) -> None:
        store = StateStore(self.root / "data/state/portfolio_state.json")
        self.assertEqual(store.state["schema_version"], 1)
        readiness = store.get_readiness(S02_VOL_ENHANCED)
        self.assertIn("standard_strangle_clean_days", readiness)
        self.assertIn("last_clean_day_date", readiness)
        self.assertIn("last_reconciliation_check", readiness)
        self.assertIn("0dte_jobs_registered", readiness)

    def test_schema_version_validation(self) -> None:
        path = self.root / "state.json"
        path.write_text(json.dumps({"schema_version": 1}))
        StateStore(path)
        path.write_text(json.dumps({}))
        with self.assertRaisesRegex(StateStoreCorruptError, "found .*expected"):
            StateStore(path)
        path.write_text(json.dumps({"schema_version": 2}))
        with self.assertRaisesRegex(StateStoreCorruptError, "found .*expected"):
            StateStore(path)

    def test_corrupt_json_raises(self) -> None:
        path = self.root / "state.json"
        path.write_text("{bad")
        with self.assertRaises(StateStoreCorruptError):
            StateStore(path)

    def test_atomic_write_replace_failure_leaves_original(self) -> None:
        path = self.root / "state.json"
        store = StateStore(path)
        original = path.read_text()
        store.state["positions"].append({"position_id": "new"})
        with mock.patch("algo_trader_unified.core.state_store.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                store.save()
        self.assertEqual(original, path.read_text())
        StateStore(path)

    def test_corrupt_tmp_ignored_by_next_save(self) -> None:
        path = self.root / "state.json"
        store = StateStore(path)
        original = path.read_text()
        path.with_name(path.name + ".tmp").write_text("{bad")
        StateStore(path)
        self.assertEqual(original, path.read_text())
        store.save()
        StateStore(path)


class BrokerGuardTests(unittest.TestCase):
    def test_diagnostic_client_rejected_before_place_order(self) -> None:
        client = mock.Mock()
        wrapper = IBKRBrokerWrapper(client)
        with self.assertRaises(DiagnosticClientOrderError):
            wrapper.submit_order(client_id=95, contract=object(), order=object(), order_ref="X")
        client.placeOrder.assert_not_called()

    def test_missing_order_ref_rejected_before_place_order(self) -> None:
        client = mock.Mock()
        wrapper = IBKRBrokerWrapper(client)
        with self.assertRaises(MissingOrderRefError):
            wrapper.submit_order(client_id=20, contract=object(), order=object(), order_ref=None)
        client.placeOrder.assert_not_called()

    def test_empty_and_whitespace_order_ref_rejected_before_place_order(self) -> None:
        client = mock.Mock()
        wrapper = IBKRBrokerWrapper(client)
        for order_ref in ("", "   "):
            with self.assertRaises(MissingOrderRefError):
                wrapper.submit_order(client_id=20, contract=object(), order=object(), order_ref=order_ref)
        client.placeOrder.assert_not_called()


class ReconciliationTests(TmpCase):
    def test_unknown_exposure_writes_ledger_event(self) -> None:
        store = StateStore(self.root / "data/state/portfolio_state.json")
        ledger = LedgerAppender(self.root)
        broker = type("Broker", (), {"aggregate_exposure": {"FUT:MES": 1}})()
        with mock.patch.object(ledger, "append", wraps=ledger.append) as append:
            result = reconcile_check(broker, store, ledger)
        self.assertFalse(result.clean)
        self.assertEqual(result.unknown_exposure, {"FUT:MES": 1.0})
        self.assertEqual(result.dirty_strategies, ["ACCOUNT"])
        self.assertIsNotNone(result.event_id)
        append.assert_called()
        self.assertEqual(append.call_args.kwargs["event_type"], "RECONCILIATION_FAILED")
        exec_text = (self.root / "data/ledger/execution_ledger.jsonl").read_text()
        self.assertIn("RECONCILIATION_FAILED", exec_text)

    def test_clean_reconciliation_does_not_write_failure(self) -> None:
        store = StateStore(self.root / "data/state/portfolio_state.json")
        ledger = LedgerAppender(self.root)
        broker = type("Broker", (), {"aggregate_exposure": {}})()
        with mock.patch.object(ledger, "append", wraps=ledger.append) as append:
            result = reconcile_check(broker, store, ledger)
        self.assertTrue(result.clean)
        append.assert_not_called()

    def test_reconciliation_module_has_no_jsonl_io(self) -> None:
        for module_path in (
            source_path("tools", "halt.py"),
            source_path("tools", "resume_halt.py"),
            source_path("tools", "reconcile_check.py"),
            source_path("core", "reconciliation.py"),
        ):
            source = module_path.read_text()
            self.assertNotIn(".jsonl", source, module_path)
            self.assertNotIn("open(", source, module_path)


class ToolTests(TmpCase):
    def _run_module(self, module: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        package_root = str(Path.cwd())
        merged_env["PYTHONPATH"] = (
            package_root
            if not merged_env.get("PYTHONPATH")
            else package_root + os.pathsep + merged_env["PYTHONPATH"]
        )
        if env:
            merged_env.update(env)
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=self.root,
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_validate_env_missing_and_present(self) -> None:
        env = os.environ.copy()
        env.pop("PHASE1_REQUIRED_TEST", None)
        missing = subprocess.run(
            [sys.executable, "-m", "algo_trader_unified.tools.validate_env", "--required", "PHASE1_REQUIRED_TEST"],
            cwd=Path.cwd(),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(missing.returncode, 1)
        self.assertIn("PHASE1_REQUIRED_TEST", missing.stdout + missing.stderr)
        present = subprocess.run(
            [sys.executable, "-m", "algo_trader_unified.tools.validate_env", "--required", "PHASE1_REQUIRED_TEST"],
            cwd=Path.cwd(),
            env={**env, "PHASE1_REQUIRED_TEST": "ok"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(present.returncode, 0)

    def test_validate_env_dotenv_loads(self) -> None:
        (self.root / ".env").write_text("DOTENV_PHASE1_TEST=loaded\n")
        result = self._run_module(
            "algo_trader_unified.tools.validate_env",
            "--required",
            "DOTENV_PHASE1_TEST",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_validate_config_valid(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "algo_trader_unified.tools.validate_config"],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_validate_config_invalid_allocation_sum(self) -> None:
        from algo_trader_unified.tools import validate_config

        with mock.patch.dict(validate_config.STRATEGY_ALLOCATIONS, {"S01_VOL_BASELINE": 1}, clear=True):
            errors = validate_config.validate()
        self.assertTrue(any("allocations sum" in error for error in errors))

    def test_validate_config_invalid_strategy_id_format(self) -> None:
        from algo_trader_unified.tools import validate_config

        with mock.patch.dict(validate_config.STRATEGY_ALLOCATIONS, {"A1_VOL_BASELINE": 400_000}, clear=True):
            errors = validate_config.validate()
        self.assertTrue(any("invalid format" in error for error in errors))

    def test_halt_account_id_rejected_without_writes(self) -> None:
        result = self._run_module(
            "algo_trader_unified.tools.halt",
            "--scope",
            "account",
            "--id",
            "foo",
            "--tier",
            "soft",
            "--operator",
            "tester",
            "--reason",
            "test",
        )
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / "data/state/halt_state.json").exists())
        self.assertFalse((self.root / "data/ledger/execution_ledger.jsonl").exists())

    def test_halt_diagnostic_client_rejected_without_writes(self) -> None:
        result = self._run_module(
            "algo_trader_unified.tools.halt",
            "--scope",
            "account",
            "--tier",
            "soft",
            "--operator",
            "tester",
            "--reason",
            "test",
            "--client-id",
            "95",
        )
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / "data/state/halt_state.json").exists())
        self.assertFalse((self.root / "data/ledger/execution_ledger.jsonl").exists())

    def test_resume_diagnostic_client_rejected_without_writes(self) -> None:
        result = self._run_module(
            "algo_trader_unified.tools.resume_halt",
            "--scope",
            "account",
            "--operator",
            "tester",
            "--reason",
            "test",
            "--halt-event-id",
            "evt_test",
            "--client-id",
            "95",
        )
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / "data/ledger/execution_ledger.jsonl").exists())

    def test_halt_and_resume_event_types(self) -> None:
        halt = self._run_module(
            "algo_trader_unified.tools.halt",
            "--scope",
            "account",
            "--tier",
            "soft",
            "--operator",
            "tester",
            "--reason",
            "test",
        )
        self.assertEqual(halt.returncode, 0, halt.stderr)
        halt_state = json.loads((self.root / "data/state/halt_state.json").read_text())
        resume = self._run_module(
            "algo_trader_unified.tools.resume_halt",
            "--scope",
            "account",
            "--operator",
            "tester",
            "--reason",
            "done",
            "--halt-event-id",
            halt_state["halt_event_id"],
        )
        self.assertEqual(resume.returncode, 0, resume.stderr)
        text = (self.root / "data/ledger/execution_ledger.jsonl").read_text()
        self.assertIn("HALT_TRIGGERED", text)
        self.assertIn("HALT_RESUMED", text)
        self.assertNotIn("MANUAL_STATUS_UPDATED", text)


if __name__ == "__main__":
    unittest.main()
