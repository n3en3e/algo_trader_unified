from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.execution import DryRunExecutionAdapter
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.order_intents import submit_order_intent
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools import submit_order_intent as submit_tool


SUBMITTED_AT = "2026-04-27T14:10:00+00:00"
FORBIDDEN_ORDER_ID_FIELD = "broker" + "_order_id"


class Phase3HCliCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_store = StateStore(self.root / "data/state/portfolio_state.json")
        self.ledger = LedgerAppender(self.root)
        self.order_path = self.root / "data/ledger/order_ledger.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def create_intent(
        self,
        strategy_id: str,
        intent_id: str,
        *,
        dry_run: bool | None = True,
    ) -> dict:
        record = {
            "intent_id": intent_id,
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "execution_mode": "paper_only",
            "status": "created",
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_ref": f"{strategy_id}|P0427XSP|OPEN",
            "created_at": "2026-04-27T13:40:00+00:00",
            "updated_at": "2026-04-27T13:40:00+00:00",
            "sizing_context": {},
            "risk_context": {},
            "signal_payload_snapshot": {},
        }
        if dry_run is not None:
            record["dry_run"] = dry_run
        return self.state_store.create_order_intent(record)

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = submit_tool.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def order_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.order_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def load_state_store(self) -> StateStore:
        return StateStore(self.root / "data/state/portfolio_state.json")


class SubmitOrderIntentCliSuccessTests(Phase3HCliCase):
    def assert_cli_submits_created_intent(self, strategy_id: str, intent_id: str) -> None:
        self.create_intent(strategy_id, intent_id)
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--submitted-at",
                SUBMITTED_AT,
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertIn(intent_id, stdout)
        self.assertIn("submitted", stdout)
        self.assertIn("simulated_order_id", stdout)
        self.assertIn("order_submitted_event_id", stdout)

        stored = self.load_state_store().get_order_intent(intent_id)
        self.assertEqual(stored["status"], "submitted")
        self.assertIn("simulated_order_id", stored)
        self.assertNotIn(FORBIDDEN_ORDER_ID_FIELD, stored)

        events = self.order_events()
        self.assertEqual([event["event_type"] for event in events], ["ORDER_SUBMITTED"])
        ledger_text = self.order_path.read_text(encoding="utf-8")
        self.assertNotIn("ORDER_CONFIRMED", ledger_text)
        self.assertNotIn("FILL_CONFIRMED", ledger_text)
        self.assertNotIn("POSITION_", ledger_text)

    def test_created_s01_intent_submits_successfully(self) -> None:
        self.assert_cli_submits_created_intent(S01_VOL_BASELINE, "s01:cli")

    def test_created_s02_intent_submits_successfully(self) -> None:
        self.assert_cli_submits_created_intent(S02_VOL_ENHANCED, "s02:cli")

    def test_json_output_is_strict_json_only(self) -> None:
        intent_id = "s01:json"
        self.create_intent(S01_VOL_BASELINE, intent_id)
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--submitted-at",
                SUBMITTED_AT,
                "--json",
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(stdout, json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        self.assertEqual(
            set(payload),
            {
                "intent_id",
                "status",
                "simulated_order_id",
                "order_submitted_event_id",
                "dry_run",
            },
        )
        self.assertEqual(payload["intent_id"], intent_id)
        self.assertEqual(payload["status"], "submitted")
        self.assertIn("simulated_order_id", payload)
        self.assertIn("order_submitted_event_id", payload)
        self.assertIs(payload["dry_run"], True)
        self.assertIsInstance(payload["dry_run"], bool)

    def test_dry_run_flag_is_accepted(self) -> None:
        intent_id = "s01:dry-flag"
        self.create_intent(S01_VOL_BASELINE, intent_id)
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--submitted-at",
                SUBMITTED_AT,
                "--dry-run",
            ]
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("submitted", stdout)


class SubmitOrderIntentCliErrorTests(Phase3HCliCase):
    def test_missing_intent_id_uses_argparse_system_exit(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.run_cli(["--root-dir", str(self.root)])
        self.assertNotEqual(caught.exception.code, 0)

    def test_unknown_intent_id_exits_nonzero(self) -> None:
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                "missing:intent",
                "--submitted-at",
                SUBMITTED_AT,
            ]
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("missing:intent", stderr)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_already_submitted_intent_exits_nonzero_without_second_event(self) -> None:
        intent_id = "s01:already"
        self.create_intent(S01_VOL_BASELINE, intent_id)
        submit_order_intent(
            state_store=self.state_store,
            ledger=self.ledger,
            execution_adapter=DryRunExecutionAdapter(),
            intent_id=intent_id,
            submitted_at=SUBMITTED_AT,
        )
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--submitted-at",
                "2026-04-27T14:20:00+00:00",
            ]
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("not 'created'", stderr)
        self.assertEqual([event["event_type"] for event in self.order_events()], ["ORDER_SUBMITTED"])

    def test_expired_and_cancelled_intents_exit_nonzero_without_order_submitted(self) -> None:
        expired_id = "s01:expired"
        cancelled_id = "s01:cancelled"
        self.create_intent(S01_VOL_BASELINE, expired_id)
        self.create_intent(S01_VOL_BASELINE, cancelled_id)
        self.state_store.expire_order_intent(
            expired_id,
            expired_at=SUBMITTED_AT,
            expire_reason="ttl_expired",
            expired_event_id="evt_expired",
        )
        self.state_store.cancel_order_intent(
            cancelled_id,
            cancelled_at=SUBMITTED_AT,
            cancel_reason="operator_cancelled",
            cancelled_event_id="evt_cancelled",
        )
        for intent_id in (expired_id, cancelled_id):
            code, stdout, stderr = self.run_cli(
                [
                    "--root-dir",
                    str(self.root),
                    "--intent-id",
                    intent_id,
                    "--submitted-at",
                    "2026-04-27T14:20:00+00:00",
                ]
            )
            self.assertNotEqual(code, 0)
            self.assertEqual(stdout, "")
            self.assertIn("not 'created'", stderr)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_invalid_submitted_at_exits_nonzero_without_mutation(self) -> None:
        intent_id = "s01:bad-time"
        self.create_intent(S01_VOL_BASELINE, intent_id)
        before = self.load_state_store().get_order_intent(intent_id)
        with self.assertRaises(SystemExit) as caught:
            self.run_cli(
                [
                    "--root-dir",
                    str(self.root),
                    "--intent-id",
                    intent_id,
                    "--submitted-at",
                    "not-a-time",
                ]
            )
        self.assertNotEqual(caught.exception.code, 0)
        self.assertEqual(self.load_state_store().get_order_intent(intent_id), before)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")

    def test_missing_dry_run_on_legacy_intent_exits_nonzero(self) -> None:
        intent_id = "s01:legacy"
        self.create_intent(S01_VOL_BASELINE, intent_id)
        legacy = self.state_store.state["order_intents"][intent_id]
        del legacy["dry_run"]
        self.state_store.save()
        code, stdout, stderr = self.run_cli(
            [
                "--root-dir",
                str(self.root),
                "--intent-id",
                intent_id,
                "--submitted-at",
                SUBMITTED_AT,
            ]
        )
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("intent.dry_run", stderr)
        self.assertEqual(self.order_path.read_text(encoding="utf-8"), "")


class SubmitOrderIntentCliSafetyTests(Phase3HCliCase):
    def test_cli_source_stays_dry_run_and_helper_only(self) -> None:
        source = inspect.getsource(submit_tool)
        self.assertIn("argparse.ArgumentParser", source)
        self.assertIn("DryRunExecutionAdapter", source)
        self.assertIn("submit_order_intent(", source)
        self.assertNotIn("ib_insync", source)
        self.assertNotIn("yfinance", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("broker.submit_order", source)
        self.assertNotIn("placeOrder", source)
        self.assertNotIn("cancelOrder", source)
        self.assertNotIn("ledger.append", source)
        self.assertNotIn("StateStore.submit_order_intent", source)
        self.assertNotIn("scheduler", source.lower())
        self.assertNotIn("ORDER_CONFIRMED", source)
        self.assertNotIn("FILL_CONFIRMED", source)
        self.assertNotIn("POSITION_", source)

    def test_import_has_no_stdout_or_stderr_side_effects(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            importlib.reload(submit_tool)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_omitted_submitted_at_uses_timezone_aware_timestamp(self) -> None:
        intent_id = "s01:auto-time"
        self.create_intent(S01_VOL_BASELINE, intent_id)
        code, stdout, stderr = self.run_cli(
            ["--root-dir", str(self.root), "--intent-id", intent_id, "--json"]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["status"], "submitted")
        stored = self.load_state_store().get_order_intent(intent_id)
        parsed = datetime.fromisoformat(stored["submitted_at"])
        self.assertIsNotNone(parsed.tzinfo)


if __name__ == "__main__":
    unittest.main()
