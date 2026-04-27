"""Run Phase 1 reconciliation diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from algo_trader_unified.core.ledger import LedgerAppender
from algo_trader_unified.core.reconciliation import reconcile_check
from algo_trader_unified.core.state_store import StateStore


@dataclass(frozen=True)
class BrokerStub:
    aggregate_exposure: dict[str, float]


def _parse_exposure(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    payload = json.loads(raw)
    return {str(k): float(v) for k, v in payload.items()}


def _write_snapshot(root: Path, result: object) -> Path:
    snapshot_dir = root / "data" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"reconcile_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": asdict(result),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--state-path", default="data/state/portfolio_state.json")
    parser.add_argument("--broker-exposure-json", default=None)
    parser.add_argument("--write-snapshot", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    state = StateStore(root / args.state_path)
    ledger = LedgerAppender(root)
    broker = BrokerStub(_parse_exposure(args.broker_exposure_json))
    result = reconcile_check(broker, state, ledger)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.write_snapshot:
        print(f"snapshot={_write_snapshot(root, result)}")
    return 0 if result.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())

