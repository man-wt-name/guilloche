"""Структурированный JSONL для metadata.jsonl.

P3-2: удалены `dump_spec`/`load_spec` (dead code; spec.py имеет свои
`save_spec_file`/`load_spec_file`).
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

METADATA_VERSION = "2.0.0"


def _clean(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _clean(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"cannot serialize metadata value: {type(obj).__name__}")


class MetadataWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._f = open(self.path, "w", encoding="utf-8")

    def write(self, entry: dict[str, Any]) -> None:
        full = {"version": METADATA_VERSION, **_clean(entry)}
        self._f.write(json.dumps(full, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "MetadataWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()