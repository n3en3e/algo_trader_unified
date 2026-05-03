from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from algo_trader_unified.config.portfolio import S01_VOL_BASELINE, S02_VOL_ENHANCED
from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.ledger_reader import LedgerReader
from algo_trader_unified.core.management import (
    ManagementSignalResult,
    default_management_signal_provider,
    run_management_scan,
)
from algo_trader_unified.core.state_store import StateStore
from algo_trader_unified.tools import run_management as management_tool


NOW = "2026-04-28T15:10:00+00:00"


class LockSpy:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.depth = 0
        self.entries = 0

    @property
    def held(self) -> bool:
        return self.depth > 0

    def __enter__(self):
        self._lock.acquire()
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.depth -= 1
        self._lock.release()


class Phase3QCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_path = self.root / "data/state/portfolio_state.json"
        self.order_path = self.root / "data/ledger/order_ledger.jsonl"
        self.execution_path = self.root / "data/ledger/execution_ledger.jsonl"
        self.state_store = StateStore(self.state_path)
        self.ledger = LedgerAppender(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def reader(self) -> LedgerReader:
        return LedgerReader(
            execution_ledger_path=self.execution_path,
            order_ledger_path=self.order_path,
        )

    def position_record(
        self,
        strategy_id: str = S01_VOL_BASELINE,
        position_id: str = "position:1",
        *,
        status: str = "open",
        dry_run: object = True,
        quantity: object = 2,
        entry_price: object = 0.75,
    ) -> dict:
        return {
            "position_id": position_id,
            "intent_id": f"{position_id}:intent",
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": dry_run,
            "opened_at": "2026-04-28T14:07:00+00:00",
            "updated_at": "2026-04-28T14:07:00+00:00",
            "source_signal_event_id": "evt_signal",
            "order_intent_created_event_id": "evt_created",
            "order_submitted_event_id": "evt_submitted",
            "order_confirmed_event_id": "evt_confirmed",
            "fill_confirmed_event_id": "evt_fill",
            "position_opened_event_id": "evt_opened",
            "order_ref": f"{strategy_id}|P0428XSP|OPEN",
            "simulated_order_id": f"sim:{position_id}",
            "fill_id": f"fill:{position_id}",
            "entry_price": entry_price,
            "quantity": quantity,
            "action": "open",
        }

    def create_position(self, strategy_id: str, position_id: str, **kwargs) -> dict:
        record = self.position_record(strategy_id, position_id, **kwargs)
        self.state_store.state["positions"][position_id] = deepcopy(record)
        self.state_store.save()
        return record

    def close_record(
        self,
        strategy_id: str,
        position_id: str,
        *,
        status: str = "created",
        close_intent_id: str | None = None,
    ) -> dict:
        close_intent_id = close_intent_id or f"{strategy_id}:{position_id}:close"
        return {
            "close_intent_id": close_intent_id,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "sleeve_id": "VOL",
            "symbol": "XSP",
            "status": status,
            "execution_mode": "paper_only",
            "dry_run": True,
            "created_at": NOW,
            "updated_at": NOW,
            "close_reason": "management",
            "requested_by": "management",
            "position_opened_event_id": "evt_opened",
            "source_signal_event_id": "evt_signal",
            "fill_confirmed_event_id": "evt_fill",
            "close_intent_created_event_id": f"evt_{close_intent_id}",
            "quantity": 2,
            "entry_price": 0.75,
            "action": "close",
        }

    def attach_close_intent(self, strategy_id: str, position_id: str, status: str) -> None:
        close_intent = self.close_record(strategy_id, position_id, status=status)
        self.state_store.state["close_intents"][close_intent["close_intent_id"]] = close_intent
        self.state_store.state["positions"][position_id][
            "active_close_intent_id"
        ] = close_intent["close_intent_id"]
        self.state_store.save()

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = management_tool.main(["--root-dir", str(self.root), *argv])
            except SystemExit as exc:
                code = int(exc.code)
        return code, stdout.getvalue(), stderr.getvalue()

    def order_events(self) -> list[dict]:
        return self.reader().read_events("order")

    def execution_events(self) -> list[dict]:
        return self.reader().read_events("execution")


class ManagementSignalResultTests(Phase3QCase):
    def test_default_provider_never_closes_and_result_is_json_safe(self) -> None:
        result = default_management_signal_provider(position={}, now=NOW)
        self.assertEqual(
            asdict(result),
            {
                "should_close": False,
                "close_reason": None,
                "requested_by": "management",
                "details": {},
            },
        )
        json.dumps(result.to_dict())

    def test_default_provider_has_no_live_data_imports(self) -> None:
        source = Path("algo_trader_unified/core/management.py").read_text(encoding="utf-8")
        for forbidden in ("ib_insync", "yfinance", "requests"):
            self.assertNotIn(forbidden, source)


class ManagementRunnerTests(Phase3QCase):
    def test_no_action_does_not_write_ledger_or_mutate_state_for_s01_and_s02(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")
        self.create_position(S02_VOL_ENHANCED, "position:s02")
        before_state = deepcopy(self.state_store.state)

        result = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            management_signal_provider=lambda position, now: ManagementSignalResult(False),
            now=NOW,
        )

        self.assertEqual(result["evaluated_count"], 2)
        self.assertEqual(result["no_action_count"], 2)
        self.assertEqual(result["close_intents_created_count"], 0)
        self.assertEqual(self.state_store.state, before_state)
        self.assertEqual(self.order_events(), [])
        self.assertEqual(self.execution_events(), [])

    def test_close_signal_creates_close_intent_only_and_preserves_open_position(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")

        result = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            management_signal_provider=lambda position, now: ManagementSignalResult(
                True, close_reason="profit target", requested_by="test_provider"
            ),
            now=NOW,
        )

        self.assertEqual(result["evaluated_count"], 1)
        self.assertEqual(result["close_intents_created_count"], 1)
        close_intent = result["created_close_intents"][0]
        self.assertEqual(close_intent["status"], "created")
        self.assertEqual(close_intent["close_reason"], "profit target")
        self.assertEqual(close_intent["requested_by"], "test_provider")
        position = self.state_store.get_position("position:s01")
        self.assertEqual(position["status"], "open")
        self.assertEqual(position["active_close_intent_id"], close_intent["close_intent_id"])
        self.assertEqual([event["event_type"] for event in self.order_events()], ["CLOSE_INTENT_CREATED"])
        self.assertEqual(self.execution_events(), [])

    def test_close_reason_and_requested_by_default_for_blank_provider_values(self) -> None:
        self.create_position(S02_VOL_ENHANCED, "position:s02")

        result = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            management_signal_provider=lambda position, now: {
                "should_close": True,
                "close_reason": " ",
                "requested_by": "",
            },
            now=NOW,
        )

        close_intent = result["created_close_intents"][0]
        self.assertEqual(close_intent["close_reason"], "management_signal")
        self.assertEqual(close_intent["requested_by"], "management")

    def test_provider_is_called_without_strategy_lock_held(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:lock")
        lock = LockSpy()
        self.state_store.strategy_state_locks[S01_VOL_BASELINE] = lock

        def provider(position, now):
            self.assertFalse(lock.held)
            return ManagementSignalResult(True, close_reason="lock test")

        result = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            management_signal_provider=provider,
            now=NOW,
        )

        self.assertEqual(result["close_intents_created_count"], 1)
        self.assertGreaterEqual(lock.entries, 1)

    def test_skips_active_close_intents_without_calling_provider(self) -> None:
        for status in ("created", "submitted", "confirmed", "filled"):
            position_id = f"position:{status}"
            self.create_position(S01_VOL_BASELINE, position_id)
            self.attach_close_intent(S01_VOL_BASELINE, position_id, status)
        self.create_position(S01_VOL_BASELINE, "position:closed-link")
        self.attach_close_intent(S01_VOL_BASELINE, "position:closed-link", "position_closed")
        calls = []

        result = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            management_signal_provider=lambda position, now: calls.append(position["position_id"])
            or ManagementSignalResult(False),
            now=NOW,
        )

        self.assertEqual(result["skipped_active_close_intent_count"], 4)
        self.assertEqual(calls, ["position:closed-link"])
        self.assertEqual(result["evaluated_count"], 1)

    def test_strategy_filter_limits_evaluation(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")
        self.create_position(S02_VOL_ENHANCED, "position:s02")
        calls = []

        result = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            strategy_id=S02_VOL_ENHANCED,
            management_signal_provider=lambda position, now: calls.append(position["position_id"])
            or ManagementSignalResult(False),
            now=NOW,
        )

        self.assertEqual(calls, ["position:s02"])
        self.assertEqual(result["evaluated_count"], 1)
        unknown = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            strategy_id="UNKNOWN",
            management_signal_provider=lambda position, now: ManagementSignalResult(True),
            now=NOW,
        )
        self.assertEqual(unknown["evaluated_count"], 0)
        self.assertEqual(unknown["close_intents_created_count"], 0)

    def test_provider_error_is_json_safe_and_other_positions_continue(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:error")
        self.create_position(S02_VOL_ENHANCED, "position:ok")

        def provider(position, now):
            if position["position_id"] == "position:error":
                raise RuntimeError("provider failed")
            return ManagementSignalResult(False)

        result = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            management_signal_provider=provider,
            now=NOW,
        )

        self.assertEqual(result["errors_count"], 1)
        self.assertEqual(result["no_action_count"], 1)
        json.dumps(result["errors"])
        self.assertEqual(self.state_store.get_active_close_intent("position:error"), None)

    def test_close_intent_creation_error_continues_to_next_position(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:bad", dry_run=False)
        self.create_position(S02_VOL_ENHANCED, "position:good")

        result = run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            management_signal_provider=lambda position, now: ManagementSignalResult(True),
            now=NOW,
        )

        self.assertEqual(result["errors_count"], 1)
        self.assertEqual(result["close_intents_created_count"], 1)
        self.assertEqual([event["event_type"] for event in self.order_events()], ["CLOSE_INTENT_CREATED"])


class ManagementCliTests(Phase3QCase):
    def test_cli_default_provider_human_and_json_no_action(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")

        code, stdout, stderr = self.run_cli(["--now", NOW])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("no close intents created", stdout)
        self.assertEqual(self.order_events(), [])

        code, stdout, stderr = self.run_cli(["--now", NOW, "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["dry_run"], True)
        self.assertEqual(payload["close_intents_created_count"], 0)

    def test_cli_injected_provider_can_create_close_intent(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:s01")
        provider = lambda position, now: ManagementSignalResult(True, close_reason="cli test")

        with mock.patch.object(management_tool, "DEFAULT_MANAGEMENT_SIGNAL_PROVIDER", provider):
            code, stdout, stderr = self.run_cli(["--now", NOW, "--json"])

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["close_intents_created_count"], 1)
        self.assertEqual([event["event_type"] for event in self.order_events()], ["CLOSE_INTENT_CREATED"])

    def test_cli_timestamp_z_normalizes_and_invalid_now_exits_nonzero(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:z")
        provider = lambda position, now: ManagementSignalResult(True)
        with mock.patch.object(management_tool, "DEFAULT_MANAGEMENT_SIGNAL_PROVIDER", provider):
            code, stdout, stderr = self.run_cli(["--now", "2026-04-28T15:10:00Z", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["created_close_intents"][0]["created_at"], NOW)

        before_state = deepcopy(self.state_store.state)
        code, stdout, stderr = self.run_cli(["--now", "not-a-date"])
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("--now must be a parseable ISO timestamp", stderr)
        self.assertEqual(self.state_store.state, before_state)

    def test_cli_omitted_now_is_timezone_aware(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:auto-now")
        seen = []

        def provider(position, now):
            seen.append(now)
            return ManagementSignalResult(False)

        with mock.patch.object(management_tool, "DEFAULT_MANAGEMENT_SIGNAL_PROVIDER", provider):
            code, stdout, stderr = self.run_cli(["--json"])

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("+00:00", seen[0])
        json.loads(stdout)


class ManagementSafetyTests(Phase3QCase):
    def test_management_path_creates_no_later_lifecycle_events(self) -> None:
        self.create_position(S01_VOL_BASELINE, "position:safe")
        run_management_scan(
            state_store=self.state_store,
            ledger=self.ledger,
            management_signal_provider=lambda position, now: ManagementSignalResult(True),
            now=NOW,
        )
        all_events = self.order_events() + self.execution_events()
        self.assertEqual([event["event_type"] for event in all_events], ["CLOSE_INTENT_CREATED"])
        forbidden = {
            "CLOSE_ORDER_SUBMITTED",
            "CLOSE_ORDER_CONFIRMED",
            "CLOSE_FILL_CONFIRMED",
            "POSITION_CLOSED",
            "POSITION_ADJUSTED",
        }
        self.assertTrue(forbidden.isdisjoint({event["event_type"] for event in all_events}))

    def test_source_safety_scans(self) -> None:
        sources = {
            "core": Path("algo_trader_unified/core/management.py").read_text(encoding="utf-8"),
            "tool": Path("algo_trader_unified/tools/run_management.py").read_text(encoding="utf-8"),
        }
        forbidden_imports = ("ib_insync", "yfinance", "requests")
        forbidden_calls = (
            "broker.submit_order",
            "placeOrder",
            "cancelOrder",
            "scheduler start",
            "submit_close_intent(",
            "confirm_close_order(",
            "confirm_close_fill(",
            "close_position_from_filled_intent(",
        )
        forbidden_fields = (
            "target_price",
            "limit_price",
            "order_type",
            "time_in_force",
            "side",
            "direction",
            "multiplier",
            "option_legs",
            "spread_legs",
        )
        for source in sources.values():
            for forbidden in forbidden_imports + forbidden_calls + forbidden_fields:
                self.assertNotIn(forbidden, source)
            self.assertNotIn("except:", source)
            self.assertNotIn(".jsonl", source)
        combined = "\n".join(sources.values())
        self.assertNotIn("0DTE", combined)
        self.assertNotIn("commodity" + "_vrp", combined.lower())
