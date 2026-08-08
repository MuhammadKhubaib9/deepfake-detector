"""Configuration loader for the DeepFake Detection System.

Loads config.yaml (paths resolved relative to the project root).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Thin dict-like wrapper around the YAML configuration."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, item):
        try:
            value = self._data[item]
        except KeyError as exc:
            raise AttributeError(f"config key '{item}' not found") from exc
        return Config(value) if isinstance(value, dict) else value

    # dict protocol - allows leaf dicts (e.g. ensemble weights) to be iterated
    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    @property
    def raw(self) -> dict:
        return self._data

    def as_dict(self) -> dict:
        return self._data


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the YAML config file (defaults to <project_root>/config.yaml)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data)


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
