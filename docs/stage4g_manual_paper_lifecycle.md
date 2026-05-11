# Stage 4G Manual Paper Lifecycle

## Stage 4G-1 Purpose

Stage 4G-1 builds a read-only lifecycle intake and reconciliation report from the manual Stage 4F paper-order artifacts. It answers whether the one-order paper smoke test can be represented as a coherent internal lifecycle intake candidate.

This stage is lifecycle intake/reporting only. It does not mutate state, write ledgers, wire daemon jobs, wire scheduler jobs, automate lifecycle transitions, submit orders, cancel orders, poll broker status, request market data, qualify contracts, or enable live trading.

## Input Artifact Sequence

1. Stage 4F-6 acceptance report
2. Stage 4F-5 smoke test report
3. Stage 4F-3 submit report
4. Stage 4F-4 status/cancel reports
5. Stage 4G-1 lifecycle intake report
6. Stage 4G-2 lifecycle state preview

The Stage 4G-1 report consumes injected JSON artifacts. It does not read StateStore, read ledgers, write StateStore JSON, or write `.jsonl` ledger files.

## Suggested Lifecycle States

The report may produce a lifecycle intake candidate with a suggested internal lifecycle state. These states are report-only and must not be treated as actual lifecycle transitions.

- `submitted_unverified`: the submit report says the order was submitted, but no broker status is available.
- `broker_submitted`: broker status indicates `Submitted`, `PreSubmitted`, or `PendingSubmit`.
- `broker_filled`: broker status indicates `Filled` and `remaining_quantity` safely parses to zero.
- `broker_partially_filled`: broker status indicates `PartiallyFilled`, or broker status says `Filled` while `remaining_quantity` is greater than zero.
- `broker_cancelled`: broker status indicates `Cancelled` or `ApiCancelled`.
- `broker_rejected_or_inactive`: broker status indicates `Rejected` or `Inactive`.
- `cancel_requested_unverified`: a cancel was attempted but the final broker status is unknown.
- `needs_reconciliation`: broker or client IDs mismatch across artifacts.
- `unsafe_artifact`: safety flags indicate live orders, market data, contract qualification, scheduler changes, lifecycle wiring, state mutation, or ledger writes.
- `unknown_broker_status`: status is missing, malformed, or not recognized.

Partial fills require manual operator review in this phase. `PartiallyFilled` and `Filled` with `remaining_quantity > 0` both map to `broker_partially_filled`.

## Example

```bash
python3 -m algo_trader_unified.tools.stage4g1_lifecycle_intake_report \
  --dry-run-only \
  --json \
  --stage4f-acceptance-json '{"readiness_for_stage4g":{"ready_to_begin_manual_paper_lifecycle_validation":true}}' \
  --smoke-test-json '{"stage4f5_smoke_test_report":true,"smoke_test":{"accepted":true,"broker_order_id":"9001","client_order_id":"intent-001","submitted":true}}' \
  --submit-json '{"manual_real_paper_submit":true,"submission":{"submitted":true,"broker_order_id":"9001","client_order_id":"intent-001"}}' \
  --order-control-json '{"manual_real_paper_order_control":true,"action":"status","order":{"broker_order_id":"9001","client_order_id":"intent-001"},"status":{"status":"Submitted"}}'
```

## Next Stage

## Stage 4G-2 Purpose

Stage 4G-2 builds a read-only manual lifecycle state preview from the Stage 4G-1 lifecycle intake report. It answers what internal order, position, lifecycle, reconciliation, operator-action, and ledger-event records would be proposed later, and whether it is safe to proceed to a manual state-write proposal phase.

This stage is preview/reporting only. It does not mutate StateStore, write ledgers, wire daemon jobs, wire scheduler jobs, automate lifecycle transitions, submit orders, cancel orders, poll broker status, request market data, qualify contracts, or enable live trading.

The proposed records and proposed ledger events in the Stage 4G-2 output are preview-only dictionaries. They are not persisted. The `write_plan` always disables state writes, ledger writes, lifecycle transitions, daemon wiring, and scheduler wiring in this phase.

Stage 4G-2 consumes injected JSON artifacts. The intended artifact sequence for this step is:

1. Stage 4G-1 lifecycle intake report
2. Stage 4G-2 lifecycle state preview

## Stage 4G-2 Example

```bash
python3 -m algo_trader_unified.tools.stage4g2_lifecycle_state_preview \
  --dry-run-only \
  --json \
  --lifecycle-intake-json '{"stage4g1_lifecycle_intake_report":true,"lifecycle_intake_candidate":{"available":true,"broker_order_id":"9001","client_order_id":"intent-001","strategy_id":"S01_VOL_BASELINE","symbol":"XSP","action":"BUY","quantity":1,"filled_quantity":0,"remaining_quantity":1,"avg_fill_price":0,"suggested_internal_lifecycle_state":"broker_submitted","reconciliation_required":false,"reconciliation_reasons":[]},"readiness_for_stage4g2":{"ready_to_build_manual_lifecycle_state_preview":true}}' \
  --state-snapshot-json '{"unresolved_needs_reconciliation_count":0,"active_halt":false}'
```

## Next Stage

Stage 4G-3 is the manual state write proposal. It remains behind explicit operator gates. It still does not enable scheduler automation, lifecycle automation, automated paper trading, or live trading.
