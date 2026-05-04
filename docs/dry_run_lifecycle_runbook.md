# Dry-Run Lifecycle Runbook

## Purpose

Use this runbook to validate the complete dry-run lifecycle in a temporary or test root. The flow exercises scheduler job wrappers and existing lifecycle helpers only.

## Preconditions

- The repo compiles.
- The test suite passes.
- Use paper/dry-run state only.
- Do not use production state for smoke validation.
- No broker, market data, deployment service, or live scheduler is required.

## Verification Commands

```bash
python3 -m py_compile $(find algo_trader_unified -name '*.py' -not -path '*/__pycache__/*')
python3 -m unittest discover -s algo_trader_unified/tests
python3 -m unittest algo_trader_unified.tests.test_phase3w_e2e_dry_run_lifecycle
```

## One-Shot Chain Runner

The manual chain runner executes the configured dry-run scheduler jobs once and exits. It does not start a scheduler, install a service, connect to a broker, or fetch market data.

Default entry-scan behavior is safe: without an injected test signal provider, the chain skips entry scans and still runs downstream jobs once, which no-op on empty state.

Examples:

```bash
python3 -m algo_trader_unified.tools.run_dry_run_chain --root-dir <test-root>
python3 -m algo_trader_unified.tools.run_dry_run_chain --root-dir <test-root> --json
python3 -m algo_trader_unified.tools.run_dry_run_chain --root-dir <test-root> --strategy-id S01_VOL_BASELINE --skip-entry-scan --skip-management-scan
python3 -m algo_trader_unified.tools.run_dry_run_chain --root-dir <test-root> --json --skip-position-transitions
```

Use `--now 2026-05-04T16:00:00Z` to pin timestamps. The `--dry-run` flag is accepted for operator clarity; the runner is dry-run only.

## Lifecycle Sequence

Entry:

1. Run S01 vol scan with an injected test signal provider.
2. Run intent submission.
3. Run intent confirmation.
4. Run intent fill confirmation.
5. Run position transitions to open the position.

Exit:

1. Run management scan with an injected close signal provider.
2. Run intent submission.
3. Run intent confirmation.
4. Run intent fill confirmation.
5. Run position transitions to close the position.

Expected lifecycle events:

- Order ledger lifecycle: `ORDER_INTENT_CREATED`, `ORDER_SUBMITTED`, `ORDER_CONFIRMED`, `FILL_CONFIRMED`, `CLOSE_INTENT_CREATED`, `CLOSE_ORDER_SUBMITTED`, `CLOSE_ORDER_CONFIRMED`, `CLOSE_FILL_CONFIRMED`
- Execution ledger lifecycle: `POSITION_OPENED`, `POSITION_CLOSED`

## Inspecting State

Useful read-only tools:

```bash
python3 -m algo_trader_unified.tools.system_status --root-dir <test-root> --json
python3 -m algo_trader_unified.tools.list_order_intents --root-dir <test-root>
python3 -m algo_trader_unified.tools.list_positions --root-dir <test-root>
```

Ledger files to inspect under the selected test root:

- `data/ledger/order_ledger.jsonl`
- `data/ledger/execution_ledger.jsonl`

## Safety Notes

- Keep smoke validation on temp or test state.
- Do not point the chain runner at production state unless intentionally doing operator dry-run validation.
- Do not start a live scheduler.
- Do not add deployment service changes for this smoke.
- Do not use broker credentials or live market data.
