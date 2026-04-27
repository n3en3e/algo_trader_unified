"""Environment loading helpers.

Secrets are supplied by either a local gitignored .env file or by the process
environment, such as systemd on the VPS.  Config modules should not contain
runtime secrets.
"""

from __future__ import annotations

import os
from pathlib import Path


def _fallback_load_dotenv(path: Path) -> None:
    """Small fallback for local tests when python-dotenv is not installed."""
    try:
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except FileNotFoundError:
        return


_ENV_PATH = Path.cwd() / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        _fallback_load_dotenv(_ENV_PATH)
    else:
        load_dotenv(_ENV_PATH)


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_optional_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be boolean-like, got {value!r}")


def get_int_env(name: str, default: int | None = None) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        if default is None:
            raise RuntimeError(f"Missing required integer environment variable: {name}")
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc

