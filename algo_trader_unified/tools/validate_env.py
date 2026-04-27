"""Validate required runtime environment variables."""

from __future__ import annotations

import argparse
import sys

from algo_trader_unified.config.env import get_required_env


DEFAULT_REQUIRED_VARS = ("IBKR_ACCOUNT",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required", nargs="*", default=list(DEFAULT_REQUIRED_VARS))
    args = parser.parse_args(argv)

    missing = []
    for name in args.required:
        try:
            value = get_required_env(name)
        except RuntimeError:
            print(f"FAIL {name}: missing")
            missing.append(name)
        else:
            redacted = "<set>" if value else "<empty>"
            print(f"PASS {name}: {redacted}")
    if missing:
        print("Missing required env vars: " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

