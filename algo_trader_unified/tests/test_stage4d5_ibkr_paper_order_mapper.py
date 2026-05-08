from __future__ import annotations

import copy
import inspect
import json
import unittest
from dataclasses import asdict, is_dataclass
from pathlib import Path

from algo_trader_unified.core import ibkr_paper_order_mapper
from algo_trader_unified.core.ibkr_paper_order_mapper import (
    IBKR_PAPER_PORT,
    IbkrPaperConfig,
    build_ibkr_paper_order_plan,
    validate_ibkr_paper_config,
)
from algo_trader_unified.core.paper_broker_adapter import BrokerOrderRequest


ROOT = Path(__file__).resolve().parents[1]
STAGE4D5_FILES = [
    ROOT / "core/ibkr_paper_order_mapper.py",
]
UNWIRED_RUNTIME_FILES = [
    ROOT / "core/paper_broker_adapter.py",
    ROOT / "tools/daemon.py",
    ROOT / "core/scheduler.py",
    ROOT / "core/scheduler_cadence.py",
]


class UnsafeThing:
    pass


def valid_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "host": "127.0.0.1",
        "port": IBKR_PAPER_PORT,
        "client_id": 7,
        "account_id": "DU1234567",
        "trading_mode": "PAPER",
        "readonly": False,
    }
    config.update(overrides)
    return config


def valid_request(**overrides: object) -> BrokerOrderRequest:
    request = {
        "client_order_id": "intent-stage4d5-001",
        "strategy_id": "S01_VOL_BASELINE",
        "symbol": "XSP",
        "asset_type": "OPTION",
        "side": "BUY",
        "quantity": 1,
        "order_type": "LIMIT",
        "limit_price": 1.25,
        "time_in_force": "DAY",
        "intent_id": "intent-stage4d5-001",
        "metadata": {"expiry": "20260619", "strike": 525.0, "right": "C"},
    }
    request.update(overrides)
    return BrokerOrderRequest(**request)  # type: ignore[arg-type]


class IbkrPaperConfigValidationTests(unittest.TestCase):
    def test_valid_paper_config_is_accepted(self) -> None:
        config = validate_ibkr_paper_config(valid_config())

        self.assertIsInstance(config, IbkrPaperConfig)
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 4004)
        self.assertEqual(config.client_id, 7)
        self.assertEqual(config.account_id, "DU1234567")
        self.assertEqual(config.trading_mode, "PAPER")
        self.assertFalse(config.readonly)

    def test_rejects_live_and_unknown_trading_modes(self) -> None:
        cases = [
            ("LIVE", "LIVE is rejected"),
            ("DRY_RUN", 'trading_mode must be exactly "PAPER"'),
            ("UNKNOWN", 'trading_mode must be exactly "PAPER"'),
            (None, 'trading_mode must be exactly "PAPER"'),
        ]
        for mode, message in cases:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, message):
                    validate_ibkr_paper_config(valid_config(trading_mode=mode))

    def test_port_validation_is_paper_only_and_rejects_likely_live_port(self) -> None:
        self.assertEqual(validate_ibkr_paper_config(valid_config(port=4004)).port, 4004)

        with self.assertRaisesRegex(ValueError, "4002 is rejected"):
            validate_ibkr_paper_config(valid_config(port=4002))

        with self.assertRaisesRegex(ValueError, "port must be 4004"):
            validate_ibkr_paper_config(valid_config(port=7497))

    def test_rejects_invalid_host_client_id_readonly_and_account_id(self) -> None:
        cases = [
            ({"host": ""}, "host must be a non-empty string"),
            ({"host": "   "}, "host must be a non-empty string"),
            ({"client_id": 0}, "client_id must be a positive int"),
            ({"client_id": -1}, "client_id must be a positive int"),
            ({"client_id": True}, "client_id must be a positive int"),
            ({"readonly": True}, "readonly must be False"),
            ({"readonly": None}, "readonly must be False"),
            ({"account_id": ""}, "account_id must be a non-empty string"),
            ({"account_id": "   "}, "account_id must be a non-empty string"),
        ]
        for override, message in cases:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    validate_ibkr_paper_config(valid_config(**override))


class IbkrPaperOrderPlanMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = validate_ibkr_paper_config(valid_config())

    def plan(self, **overrides: object):
        return build_ibkr_paper_order_plan(
            valid_request(**overrides),
            config=self.config,
        )

    def test_buy_and_sell_side_mapping(self) -> None:
        self.assertEqual(self.plan(side="BUY").action, "BUY")
        self.assertEqual(self.plan(side="SELL").action, "SELL")

    def test_limit_and_market_order_type_mapping(self) -> None:
        limit_plan = self.plan(order_type="LIMIT", limit_price=1.25)
        market_plan = self.plan(order_type="MARKET", limit_price=None)

        self.assertEqual(limit_plan.order_type, "LMT")
        self.assertEqual(limit_plan.ibkr_order_hint["orderType"], "LMT")
        self.assertEqual(limit_plan.ibkr_order_hint["lmtPrice"], 1.25)
        self.assertEqual(market_plan.order_type, "MKT")
        self.assertEqual(market_plan.ibkr_order_hint["orderType"], "MKT")
        self.assertNotIn("lmtPrice", market_plan.ibkr_order_hint)

    def test_market_with_limit_price_fails_closed(self) -> None:
        plan = self.plan(order_type="MARKET", limit_price=1.25)

        self.assertFalse(plan.ready_for_submission)
        self.assertIn("MARKET orders must not include limit_price", plan.blockers)

    def test_limit_without_positive_limit_price_fails_closed(self) -> None:
        for limit_price in (None, 0, -1):
            with self.subTest(limit_price=limit_price):
                plan = self.plan(order_type="LIMIT", limit_price=limit_price)

                self.assertFalse(plan.ready_for_submission)
                self.assertIn("LIMIT orders require positive limit_price", plan.blockers)

    def test_time_in_force_defaults_and_maps_supported_values(self) -> None:
        default_plan = self.plan(time_in_force=None)
        day_plan = self.plan(time_in_force="DAY")
        gtc_plan = self.plan(time_in_force="GTC")

        self.assertIsInstance(default_plan.time_in_force, str)
        self.assertIsInstance(day_plan.time_in_force, str)
        self.assertIsInstance(gtc_plan.time_in_force, str)
        self.assertEqual(default_plan.time_in_force, "DAY")
        self.assertEqual(default_plan.ibkr_order_hint["tif"], "DAY")
        self.assertEqual(day_plan.ibkr_order_hint["tif"], "DAY")
        self.assertEqual(gtc_plan.ibkr_order_hint["tif"], "GTC")

    def test_unsupported_time_in_force_fails_closed_with_blocker(self) -> None:
        plan = self.plan(time_in_force="IOC")

        self.assertFalse(plan.ready_for_submission)
        self.assertIn("time_in_force must be one of ['DAY', 'GTC']", plan.blockers)
        self.assertNotIn("tif", plan.ibkr_order_hint)

    def test_option_metadata_extracts_json_contract_hints(self) -> None:
        plan = self.plan(
            asset_type="OPTION",
            metadata={
                "expiry": "20260619",
                "strike": 525,
                "right": "CALL",
                "unsafe": UnsafeThing(),
            },
        )

        self.assertTrue(plan.ready_for_submission)
        self.assertEqual(plan.ibkr_contract_hint["secType"], "OPT")
        self.assertEqual(plan.ibkr_contract_hint["expiry"], "20260619")
        self.assertEqual(plan.ibkr_contract_hint["strike"], 525.0)
        self.assertEqual(plan.ibkr_contract_hint["right"], "CALL")
        self.assertEqual(
            plan.ibkr_contract_hint["metadata"]["unsafe"],
            f"<{UnsafeThing.__module__}.UnsafeThing>",
        )
        self.assertIs(type(plan.ibkr_contract_hint), dict)
        json.dumps(plan.ibkr_contract_hint, sort_keys=True)

    def test_option_missing_critical_metadata_is_not_ready(self) -> None:
        plan = self.plan(asset_type="OPTION", metadata={})

        self.assertFalse(plan.ready_for_submission)
        self.assertIn("OPTION contract hint requires metadata.expiry", plan.blockers)
        self.assertIn("OPTION contract hint requires numeric metadata.strike", plan.blockers)
        self.assertIn("OPTION contract hint requires metadata.right", plan.blockers)

    def test_supported_non_option_asset_types_use_plain_contract_hints(self) -> None:
        cases = [
            ("STOCK", "STK"),
            ("FUTURE", "FUT"),
            ("CASH", "CASH"),
        ]
        for asset_type, sec_type in cases:
            with self.subTest(asset_type=asset_type):
                plan = self.plan(asset_type=asset_type, metadata=None)

                self.assertTrue(plan.ready_for_submission)
                self.assertEqual(plan.ibkr_contract_hint["secType"], sec_type)

    def test_unsupported_values_fail_closed_with_blockers(self) -> None:
        cases = [
            ({"side": "HOLD"}, "unsupported side: HOLD"),
            ({"order_type": "STOP", "limit_price": None}, "unsupported order_type: STOP"),
            ({"asset_type": "INDEX_OPTION"}, "unsupported asset_type: INDEX_OPTION"),
            ({"quantity": 0}, "quantity must be positive numeric"),
        ]
        for override, blocker in cases:
            with self.subTest(override=override):
                plan = self.plan(**override)

                self.assertFalse(plan.ready_for_submission)
                self.assertIn(blocker, plan.blockers)

    def test_blocked_order_hints_omit_invalid_ibkr_shaped_values(self) -> None:
        unsupported_tif = self.plan(time_in_force="IOC")
        unsupported_side = self.plan(side="HOLD")
        unsupported_order_type = self.plan(order_type="STOP", limit_price=None)

        self.assertFalse(unsupported_tif.ready_for_submission)
        self.assertNotEqual(unsupported_tif.ibkr_order_hint.get("tif"), "IOC")
        self.assertNotIn("tif", unsupported_tif.ibkr_order_hint)

        self.assertFalse(unsupported_side.ready_for_submission)
        self.assertNotEqual(unsupported_side.ibkr_order_hint.get("action"), "HOLD")
        self.assertNotIn("action", unsupported_side.ibkr_order_hint)

        self.assertFalse(unsupported_order_type.ready_for_submission)
        self.assertNotEqual(unsupported_order_type.ibkr_order_hint.get("orderType"), "STOP")
        self.assertNotIn("orderType", unsupported_order_type.ibkr_order_hint)

    def test_unsupported_asset_type_does_not_imply_valid_ibkr_contract(self) -> None:
        plan = self.plan(asset_type="INDEX_OPTION")

        self.assertFalse(plan.ready_for_submission)
        self.assertIn("unsupported asset_type: INDEX_OPTION", plan.blockers)
        self.assertNotIn("secType", plan.ibkr_contract_hint)

    def test_dry_run_and_paper_only_invariants_hold_for_ready_and_blocked_plans(self) -> None:
        ready_plan = self.plan()
        blocked_plan = self.plan(side="HOLD")

        self.assertTrue(ready_plan.ready_for_submission)
        self.assertTrue(ready_plan.dry_run)
        self.assertTrue(ready_plan.paper_only)
        self.assertFalse(blocked_plan.ready_for_submission)
        self.assertTrue(blocked_plan.dry_run)
        self.assertTrue(blocked_plan.paper_only)

    def test_client_order_id_is_preserved_and_request_is_not_mutated(self) -> None:
        request = valid_request(
            client_order_id="client-order-exact",
            metadata={"expiry": "20260619", "strike": 525.0, "right": "P"},
        )
        before = copy.deepcopy(asdict(request))

        plan = build_ibkr_paper_order_plan(request, config=self.config)

        self.assertEqual(plan.client_order_id, "client-order-exact")
        self.assertEqual(asdict(request), before)

    def test_plan_is_json_safe_and_uses_plain_dict_hints(self) -> None:
        plan = self.plan(metadata={"expiry": "20260619", "strike": 525.0, "right": "C"})

        self.assertTrue(is_dataclass(plan))
        self.assertIs(type(plan.ibkr_contract_hint), dict)
        self.assertIs(type(plan.ibkr_order_hint), dict)
        json.dumps(asdict(plan), sort_keys=True)

    def test_no_nondeterministic_values_are_generated_inside_mapper_module(self) -> None:
        source = inspect.getsource(ibkr_paper_order_mapper)
        forbidden_tokens = (
            "uuid." + "uuid4",
            "random.",
            "time.time",
            "datetime.now",
        )
        for token in forbidden_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, source)


class Stage4D5SafetyBoundaryTests(unittest.TestCase):
    def test_stage4d5_files_do_not_import_or_call_blocked_integrations(self) -> None:
        blocked_tokens = (
            "ib_" + "insync",
            "req" + "MktData",
            "qualify" + "Contracts",
            "place" + "Order",
            "y" + "finance",
            "requests",
            "url" + "lib",
            "system" + "ctl",
            "system" + "d",
            "uuid." + "uuid4",
            "rand" + "om",
        )
        for path in STAGE4D5_FILES:
            source = path.read_text(encoding="utf-8")
            for token in blocked_tokens:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_stage4d5_files_do_not_import_broker_client_symbols(self) -> None:
        forbidden_lines = (
            "from ib",
            "import ib",
            "IB(",
            " IB(",
        )
        for path in STAGE4D5_FILES:
            source = path.read_text(encoding="utf-8")
            for token in forbidden_lines:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_mapper_is_not_wired_into_adapter_daemon_scheduler_or_lifecycle(self) -> None:
        for path in UNWIRED_RUNTIME_FILES:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn("ibkr_paper_order_mapper", source)
                self.assertNotIn("build_ibkr_paper_order_plan", source)


if __name__ == "__main__":
    unittest.main()
