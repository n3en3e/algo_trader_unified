"""Print configured scheduler job specs without starting the scheduler."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from algo_trader_unified.config.scheduler import JOB_SPECS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    payload = [asdict(spec) for spec in JOB_SPECS.values()]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
