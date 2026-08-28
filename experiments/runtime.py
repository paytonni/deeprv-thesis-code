"""Runtime measurement and JSON serialization helpers."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter


@contextmanager
def elapsed_seconds(result: dict, key: str = "elapsed_seconds"):
    started = perf_counter()
    try:
        yield
    finally:
        result[key] = perf_counter() - started


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
