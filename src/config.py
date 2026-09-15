"""Config loading. Dotted access so stages read `cfg.screening.window`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def _namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path = "config.yaml") -> SimpleNamespace:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path.resolve()}")
    with path.open() as fh:
        return _namespace(yaml.safe_load(fh))


def as_dict(ns) -> dict:
    if isinstance(ns, SimpleNamespace):
        return {k: as_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [as_dict(v) for v in ns]
    return ns
