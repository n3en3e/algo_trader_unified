from __future__ import annotations

import importlib
import inspect
import unittest
from copy import deepcopy
from decimal import Decimal

from algo_trader_unified.core import close_intents


BASE_RECORD = {
    "close_intent_id": "close:1",
    "position_id": "position:1",
    "strategy_id": "S01_VOL_BASELINE",
    "sleeve_id": "VOL",
    "symbol": "XSP",
    "execution_mode": "paper_only",
    "dry_run": True,
    "close_intent_created_event_id": "evt_close_created",
    "position_opened_event_id": "evt_position_opened",
    "fill_confirmed_event_id": "evt_fill_confirmed",
    "quantity": 2,
    "entry_price": Decimal("0.75"),
    "close_reason": "manual",
    "requested_by": "operator",
}


def created_record(**overrides) -> dict:
    record = deepcopy(BASE_RECORD)
    record.update(overrides)
    return record


def submitted_record(**overrides) -> dict:
    record = created_record(
        close_order_submitted_event_id="evt_close_submitted",
        close_order_ref="S01_VOL_BASELINE:position:1:close",
        simulated_close_order_id="sim_close_1",
    )
    record.update(overrides)
    return record


def confirmed_record(**overrides) -> dict:
    record = submitted_record(close_order_confirmed_event_id="evt_close_confirmed")
    record.update(overrides)
    return record


class CloseIntentValidationCleanupTests(unittest.TestCase):
    def assert_missing_dry_run_references_phase(self, validator, record: dict, phase: str) -> None:
        record.pop("dry_run")
        with self.assertRaisesRegex(ValueError, phase):
            validator(record)

    def assert_false_dry_run_references_phase(self, validator, record: dict, phase: str) -> None:
        record["dry_run"] = False
        with self.assertRaisesRegex(ValueError, phase):
            validator(record)

    def assert_missing_field_is_clear(self, validator, record: dict, field: str) -> None:
        record.pop(field)
        with self.assertRaisesRegex(ValueError, field):
            validator(record)

    def assert_numeric_rejection(self, validator, record_factory) -> None:
        for field, value in (
            ("quantity", "2"),
            ("entry_price", "0.75"),
            ("quantity", True),
            ("entry_price", False),
        ):
            with self.subTest(field=field, value=value):
                record = record_factory()
                record[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validator(record)

    def test_created_wrapper_preserves_phase_specific_validation(self) -> None:
        validator = close_intents._validate_close_intent
        self.assert_missing_dry_run_references_phase(validator, created_record(), "Phase 3O-1")
        self.assert_false_dry_run_references_phase(validator, created_record(), "Phase 3O-1")
        self.assert_missing_field_is_clear(validator, created_record(), "close_reason")
        self.assert_numeric_rejection(validator, created_record)

        quantity, entry_price = validator(created_record(quantity=3.0, entry_price=Decimal("1.25")))
        self.assertEqual(quantity, 3.0)
        self.assertEqual(entry_price, 1.25)

    def test_submitted_wrapper_preserves_phase_specific_validation(self) -> None:
        validator = close_intents._validate_submitted_close_intent
        self.assert_missing_dry_run_references_phase(validator, submitted_record(), "Phase 3O-2")
        self.assert_false_dry_run_references_phase(validator, submitted_record(), "Phase 3O-2")
        self.assert_missing_field_is_clear(
            validator, submitted_record(), "close_order_submitted_event_id"
        )
        self.assert_numeric_rejection(validator, submitted_record)

        quantity, entry_price = validator(submitted_record(quantity=4, entry_price=1.5))
        self.assertEqual(quantity, 4)
        self.assertEqual(entry_price, 1.5)

    def test_confirmed_wrapper_preserves_phase_specific_validation(self) -> None:
        validator = close_intents._validate_confirmed_close_intent
        self.assert_missing_dry_run_references_phase(validator, confirmed_record(), "Phase 3O-3")
        self.assert_false_dry_run_references_phase(validator, confirmed_record(), "Phase 3O-3")
        self.assert_missing_field_is_clear(
            validator, confirmed_record(), "close_order_confirmed_event_id"
        )
        self.assert_numeric_rejection(validator, confirmed_record)

        quantity, entry_price = validator(confirmed_record(quantity=5, entry_price=2))
        self.assertEqual(quantity, 5)
        self.assertEqual(entry_price, 2)

    def test_validation_source_uses_single_base_helper(self) -> None:
        module = importlib.import_module("algo_trader_unified.core.close_intents")
        source = inspect.getsource(module)
        self.assertEqual(source.count("def _validate_close_intent_base("), 1)
        self.assertEqual(source.count("is missing close_intent.dry_run"), 1)
        self.assertEqual(source.count("has dry_run={close_intent['dry_run']!r}"), 1)

        for wrapper_name in (
            "_validate_close_intent",
            "_validate_submitted_close_intent",
            "_validate_confirmed_close_intent",
        ):
            wrapper_source = inspect.getsource(getattr(module, wrapper_name))
            self.assertIn("_validate_close_intent_base", wrapper_source)


if __name__ == "__main__":
    unittest.main()
