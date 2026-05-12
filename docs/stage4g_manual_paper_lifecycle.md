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
7. Stage 4G-3 state write proposal

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

## Stage 4G-3 Purpose

Stage 4G-3 builds a read-only manual state write proposal from the Stage 4G-2 lifecycle state preview. It answers what StateStore records and ledger events would be proposed in a later explicitly approved phase, and whether the proposal is complete enough to move to a manually gated dry run.

This stage is proposal/reporting only. It does not mutate StateStore, write ledgers, wire daemon jobs, wire scheduler jobs, automate lifecycle transitions, submit orders, cancel orders, poll broker status, request market data, qualify contracts, or enable live trading.

The intended artifact sequence for this step is:

1. Stage 4G-2 lifecycle state preview
2. Stage 4G-3 state write proposal

The `proposed_state_store_operations` field uses structured dictionaries with an explicit operation name and payload, such as `{"operation": "upsert_order", "payload": {...}}` and `{"operation": "upsert_position", "payload": {...}}`. These operations are proposal-only and are not executed in Stage 4G-3.

The `proposed_ledger_events` field contains proposal-only ledger event payloads. They are not appended to any ledger in this phase. The Stage 4G-3 `write_plan` always disables StateStore writes, ledger writes, lifecycle transitions, daemon wiring, and scheduler wiring.

Stage 4G-3 lists the operator acknowledgements that would be required for a future write phase, including acknowledgement that paper lifecycle state and ledger events would be written later, that the flow remains PAPER only, that scheduler automation remains disabled, and that the proposed payloads were reviewed. These acknowledgements are listed for review but are not enforced yet.

## Stage 4G-3 Example

```bash
python3 -m algo_trader_unified.tools.stage4g3_state_write_proposal \
  --dry-run-only \
  --json \
  --lifecycle-state-preview-json '{"stage4g2_lifecycle_state_preview":true,"preview":{"available":true,"proposed_lifecycle_state":"paper_order_submitted","proposed_order_record":{"broker_order_id":"9001","client_order_id":"intent-001","strategy_id":"S01_VOL_BASELINE","symbol":"XSP","action":"BUY","quantity":1,"order_type":"LIMIT","status":"submitted"},"proposed_position_record":null,"proposed_ledger_events":[{"event_type":"paper_order_lifecycle_state_preview","timestamp":"2026-05-10T12:30:00+00:00","client_order_id":"intent-001","broker_order_id":"9001"}]},"write_plan":{"state_store_write_enabled":false,"ledger_write_enabled":false,"lifecycle_transition_enabled":false,"daemon_wiring_enabled":false,"scheduler_wiring_enabled":false},"readiness_for_stage4g3":{"ready_to_build_manual_state_write_proposal":true}}' \
  --state-snapshot-json '{"unresolved_needs_reconciliation_count":0,"active_halt":false}' \
  --operator-notes-json '{"manual_observation":"reviewed in paper account","cleanup_ticket":null,"operator_initials":"AB","follow_up_required":false}'
```

## Next Stage

Stage 4G-4 is the manual state write dry run. It remains manually gated, still does not enable scheduler automation, and still does not enable live trading.

## Stage 4G-4 Purpose

Stage 4G-4 builds a read-only manual state write dry-run report from the Stage 4G-3 state write proposal. It answers what StateStore writes, ledger events, validation checks, lifecycle transition record, and rollback plan would be attempted later if an operator explicitly approved a separate write executor phase.

This stage is dry-run/reporting only. It does not mutate StateStore, write ledgers, wire daemon jobs, wire scheduler jobs, automate lifecycle transitions, submit orders, cancel orders, poll broker status, request market data, qualify contracts, or enable live trading.

The intended artifact sequence for this step is:

1. Stage 4G-3 state write proposal
2. Stage 4G-4 state write dry run

The `dry_run_operations` field is preview-only. Each operation has `would_execute=false`, a target of `StateStore`, `Ledger`, or `Lifecycle`, and preserves the proposal payload that would be reviewed in the next phase. StateStore order operations appear before StateStore position operations, ledger events preserve proposal order, and the lifecycle transition appears last.

The `rollback_simulation` section is descriptive only. Rollback requires manual StateStore/ledger file reversion using standard system backups, and no automated rollback is supported in this phase. Stage 4G-4 does not generate rollback code, shell commands, SQL, or executable rollback operations.

The Stage 4G-4 `write_plan` always disables StateStore writes, ledger writes, lifecycle transitions, daemon wiring, and scheduler wiring. Missing or malformed proposal artifacts, enabled write flags, unsafe flags, duplicate IDs, unresolved `NEEDS_RECONCILIATION`, active halts, and missing acknowledgements block readiness for Stage 4G-5.

Exact acknowledgements required for future write readiness:

- `I understand this will write paper lifecycle state.`
- `I understand this will write ledger events.`
- `I understand this is still PAPER only.`
- `I understand this does not enable scheduler automation.`
- `I reviewed the proposed StateStore and ledger payloads.`

Extra acknowledgement text does not compensate for a missing required acknowledgement, and a single combined string does not satisfy the gate.

## Stage 4G-4 Example

```bash
python3 -m algo_trader_unified.tools.stage4g4_state_write_dry_run \
  --dry-run-only \
  --json \
  --state-write-proposal-json '{"stage4g3_state_write_proposal":true,"proposal":{"available":true,"proposed_state_store_operations":[{"operation":"upsert_order","payload":{"client_order_id":"intent-001","broker_order_id":"9001","symbol":"XSP","paper_only":true}}],"proposed_ledger_events":[{"event_type":"paper_order_lifecycle_state_preview","timestamp":"2026-05-10T12:30:00+00:00","client_order_id":"intent-001","broker_order_id":"9001"}],"proposed_lifecycle_transition":{"transition_to":"paper_order_submitted","proposal_only":true,"enabled":false}},"write_plan":{"state_store_write_enabled":false,"ledger_write_enabled":false,"lifecycle_transition_enabled":false,"daemon_wiring_enabled":false,"scheduler_wiring_enabled":false},"safety_checks":{"no_live_orders":true,"no_market_data":true,"no_contract_qualification":true,"no_scheduler_changes":true,"no_lifecycle_wiring":true,"no_state_mutation":true,"no_ledger_writes":true},"readiness_for_stage4g4":{"ready_to_build_manual_state_write_dry_run":true}}' \
  --state-snapshot-json '{"unresolved_needs_reconciliation_count":0,"active_halt":false}' \
  --ack "I understand this will write paper lifecycle state." \
  --ack "I understand this will write ledger events." \
  --ack "I understand this is still PAPER only." \
  --ack "I understand this does not enable scheduler automation." \
  --ack "I reviewed the proposed StateStore and ledger payloads."
```

## Next Stage

Stage 4G-5 is the manual state write executor behind explicit operator gates. It still does not enable scheduler automation and still does not enable live trading.

## Stage 4G-5 Purpose

Stage 4G-5 is the manual paper lifecycle state write executor. It answers whether an explicitly approved Stage 4G-4 dry-run packet can be applied through injected StateStore and ledger writer abstractions only, with deterministic reporting and no automation.

This is the first permitted write phase in Stage 4G, but only after all manual gates pass. The executor requires `allow_state_write=true`, `allow_ledger_write=true`, a valid and ready Stage 4G-4 dry-run packet, clean write-plan and safety flags, and exact operator acknowledgement list items. The acknowledgements use exact string equality after trimming each list item; a single combined acknowledgement string does not pass.

The core executor does not instantiate production StateStore, ledger, IB, execution, scheduler, or lifecycle objects. Writes are allowed only through injected writer abstractions such as `state_store_writer.upsert_order(...)`, `state_store_writer.upsert_position(...)`, and `ledger_writer.append_event(...)`. The optional CLI is conservative and does not instantiate real production writers in this phase.

Partial failure reporting is explicit:

- `applied_operations` lists successful writes in the order they occurred.
- `skipped_operations` lists abandoned operations after the first failure.
- `rollback_required=true` means manual rollback using standard backups is required.
- No automated rollback is supported, and no rollback commands are generated.

Idempotent or duplicate writer responses such as `already_exists` are accepted only when the returned record IDs match the requested `client_order_id` and `broker_order_id` values. Mismatched IDs fail closed and stop execution.

Stage 4G-5 does not submit orders, cancel orders, poll status, call broker APIs, request market data, qualify contracts, execute lifecycle transitions, wire daemon jobs, wire scheduler jobs, automate paper trading, or enable live trading. The lifecycle transition from the dry-run packet is recorded only in the report context and is not executed as a mutation.

## Stage 4G-5 Example

```bash
python3 -m algo_trader_unified.tools.stage4g5_state_write_executor \
  --dry-run-only \
  --json \
  --state-write-dry-run-json '{"stage4g4_state_write_dry_run":true,"operation_schema_checks":{"operations_structured":true,"recognized_operations":true,"deterministic_operation_order":true},"dry_run_packet":{"available":true,"dry_run_operations":[{"sequence_number":1,"target":"StateStore","operation":"upsert_order","would_execute":false,"payload":{"client_order_id":"intent-001","broker_order_id":"9001","symbol":"XSP","paper_only":true}},{"sequence_number":2,"target":"Ledger","operation":"append_event","would_execute":false,"payload":{"event_type":"paper_order_lifecycle_state_write","timestamp":"2026-05-10T12:30:00+00:00","client_order_id":"intent-001","broker_order_id":"9001"}},{"sequence_number":3,"target":"Lifecycle","operation":"record_lifecycle_transition","would_execute":false,"payload":{"transition_to":"paper_order_submitted","proposal_only":true,"enabled":false}}]},"write_plan":{"state_store_write_enabled":false,"ledger_write_enabled":false,"lifecycle_transition_enabled":false,"daemon_wiring_enabled":false,"scheduler_wiring_enabled":false},"safety_checks":{"no_live_orders":true,"no_market_data":true,"no_contract_qualification":true,"no_scheduler_changes":true,"no_lifecycle_wiring":true,"no_state_mutation":true,"no_ledger_writes":true},"readiness_for_stage4g5":{"ready_to_build_manual_state_write_executor":true}}' \
  --allow-state-write \
  --allow-ledger-write \
  --ack "I understand this will write paper lifecycle state." \
  --ack "I understand this will write ledger events." \
  --ack "I understand this is still PAPER only." \
  --ack "I understand this does not enable scheduler automation." \
  --ack "I reviewed the proposed StateStore and ledger payloads."
```

## Next Stage

Stage 4G-6 is the manual lifecycle write acceptance report. It verifies the Stage 4G-5 write report and any manually inspected StateStore and ledger records. It still does not enable scheduler automation and still does not enable live trading.
