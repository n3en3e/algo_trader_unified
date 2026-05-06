# Stage 4B Dry-Run Operations

## Purpose

Stage 4B dry-run mode validates the local daemon stack, scheduler job registration, readiness gates, local snapshots, digest output, halt state, and dry-run lifecycle wrappers.

It does not execute broker orders, fetch market data, manage systemd, submit paper/live orders, or deploy anything to production. Paper broker adapters and paper execution are later-stage work, not part of Stage 4B.

## AyoBot Checkpoint

```bash
cd /home/algobot/algo_trader_unified
git pull
python3 -m py_compile $(find algo_trader_unified -name '*.py' -not -path '*/__pycache__/*')
python3 -m unittest discover -s algo_trader_unified/tests
```

## Acceptance Report

The acceptance report is read-only. It requires `--dry-run-only`, does not start the scheduler, does not run jobs, and reports scheduler registration using a test-safe collector.

```bash
cd /home/algobot/algo_trader_unified
python3 -m algo_trader_unified.tools.daemon --dry-run-only --acceptance-report
```

Expected success criteria:

- `dry_run` and `acceptance_report` are `true`.
- `startup_gate.state_store_loadable` is `true`.
- `scheduler.jobs_registered_without_triggers` is empty.
- `scheduler.observation_jobs_registered` contains only Stage 4A observation jobs.
- `scheduler.lifecycle_jobs_registered` contains Stage 4A jobs plus the three Stage 4B intent-level lifecycle jobs: `JOB_DRY_RUN_SUBMIT_PENDING_INTENTS`, `JOB_DRY_RUN_EXPIRE_INTENTS`, and `JOB_DRY_RUN_EOD_INTENT_CLEANUP`.
- `safety.broker_calls_enabled`, `market_data_enabled`, `systemd_enabled`, and `paper_live_orders_enabled` are all `false`.

## Bounded Smoke

Runs the dry-run jobs once and exits. Use the lifecycle flag only when explicitly validating the Stage 4B lifecycle wrappers.

```bash
cd /home/algobot/algo_trader_unified
python3 -m algo_trader_unified.tools.daemon --dry-run-only --smoke-cycles 1
```

```bash
cd /home/algobot/algo_trader_unified
python3 -m algo_trader_unified.tools.daemon --dry-run-only --smoke-cycles 1 --enable-lifecycle-pipeline
```

## Bounded Foreground

Starts the foreground scheduler for a fixed local window, then shuts it down.

```bash
cd /home/algobot/algo_trader_unified
python3 -m algo_trader_unified.tools.daemon --dry-run-only --foreground-runtime-seconds 60 --enable-triggers
```

```bash
cd /home/algobot/algo_trader_unified
python3 -m algo_trader_unified.tools.daemon --dry-run-only --foreground-runtime-seconds 60 --enable-triggers --enable-lifecycle-pipeline
```

## Daily Digest

The daily digest is part of the Stage 4A observation set and is included in smoke runs. It writes a local digest text file under `data/snapshots`.

```bash
cd /home/algobot/algo_trader_unified
python3 -m algo_trader_unified.tools.daemon --dry-run-only --smoke-cycles 1
```

## Interpreting Results

Active halt: `startup_gate.halt_active=true` means operator halt state is active. Acceptance reporting only reports it; normal daemon startup remains blocked by halt state.

Missing readiness: readiness marked unavailable or false means strategy gates have not been evaluated or StateStore readiness is incomplete. Entry scans should skip rather than trade.

Stale snapshot: `snapshots.account_snapshot_fresh=false` means the latest local account snapshot is missing or older than the freshness window. Treat NLV/account health as not validated.

Digest output: `snapshots.latest_digest_path` points to the latest local digest if one exists. Missing digest output means the daily digest job has not recently written a local file.

Lifecycle disabled/enabled: lifecycle pipeline is disabled by default. Without `--enable-lifecycle-pipeline`, report and foreground modes should show only Stage 4A observation jobs. With the flag, only the three Stage 4B intent-level lifecycle jobs are added: `JOB_DRY_RUN_SUBMIT_PENDING_INTENTS`, `JOB_DRY_RUN_EXPIRE_INTENTS`, and `JOB_DRY_RUN_EOD_INTENT_CLEANUP`.

Manual-only lifecycle: `JOB_DRY_RUN_CONFIRM_SUBMITTED_ORDERS`, `JOB_DRY_RUN_CONFIRM_FILLS`, and `JOB_DRY_RUN_APPLY_POSITION_TRANSITIONS` are excluded from scheduled Stage 4B automation.
## Boundaries

Stage 4B operations are explicitly:

- no broker execution
- no market data
- no systemd changes
- no paper/live orders
- no scheduled order confirmation, fill simulation, position transition, or position close finalization

The next boundary is a later paper broker adapter / paper execution stage. Do not add that work while validating Stage 4B dry-run operations.
