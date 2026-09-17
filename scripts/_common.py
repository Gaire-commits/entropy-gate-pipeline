"""Shared script setup: repo root on the path, config and output folder from --config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = "configs/etf_intraday.yaml"


def output_dir(cfg) -> Path:
    out = Path("outputs") / cfg.experiment
    out.mkdir(parents=True, exist_ok=True)
    return out
