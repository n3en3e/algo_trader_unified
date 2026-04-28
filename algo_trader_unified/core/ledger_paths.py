"""Shared ledger path constants."""

from __future__ import annotations

from pathlib import Path


LEDGER_DIR_RELATIVE_PATH = Path("data") / "ledger"
ORDER_LEDGER_FILENAME = "order_ledger.jsonl"
EXECUTION_LEDGER_FILENAME = "execution_ledger.jsonl"
ORDER_LEDGER_RELATIVE_PATH = LEDGER_DIR_RELATIVE_PATH / ORDER_LEDGER_FILENAME
EXECUTION_LEDGER_RELATIVE_PATH = LEDGER_DIR_RELATIVE_PATH / EXECUTION_LEDGER_FILENAME
