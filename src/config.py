"""Typed access to config.yaml. Every tunable lives in the YAML file —
this module only loads it and gives dotted, defaulted access."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class Config:
    """Thin wrapper around a nested dict loaded from YAML.

    Usage:
        cfg = Config.load()
        cfg.get("training.rounds")          # -> 5
        cfg.get("training.missing", 10)     # -> 10 (default)
    """

    def __init__(self, data: dict, path: Path | None = None):
        self._data = data
        self.path = path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else _DEFAULT_PATH
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data, path=path)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"
