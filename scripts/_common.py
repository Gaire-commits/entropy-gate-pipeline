"""Shared script setup: repo root on the path, config and output folder from --config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = "configs/etf_intraday.yaml"


def unreachable(path) -> str | None:
    """The first part of `path` that is a link to nothing, or None. In Colab `data/` and `outputs/` link
    into Drive, so a broken link means Drive is not mounted; without this check the script fails deep inside
    with "File exists: 'outputs'"."""
    p = Path(path)
    for part in [*reversed(p.parents), p]:
        if part.is_symlink() and not part.exists():
            return str(part)
    return None


def reachable(*paths) -> bool:
    """False, with a message, when any path runs through a broken link (Drive not mounted)."""
    broken = sorted({b for b in (unreachable(p) for p in paths) if b})
    if broken:
        print(f"STOPPED: {', '.join(broken)} point at Google Drive, but Drive is not mounted. Run section 1 to "
              "mount it (or restart the session and run sections 1-3), then run this cell again.")
    return not broken


def enough_symbols(bars: dict, universe: list, cache_dir, minimum: float = 0.9) -> bool:
    """False, with a message, when the cache holds too little of the universe to run on.

    A fetch that stopped partway leaves a cache that loads without error; a result computed on a
    fraction of the stocks looks like a result and is not one."""
    if len(bars) >= minimum * len(universe):
        return True
    print(f"STOPPED: only {len(bars)} of {len(universe)} symbols are cached in {cache_dir} "
          f"({len(bars) / max(len(universe), 1):.0%}; at least {minimum:.0%} is required). "
          "The fetch has not finished: re-run scripts/fetch_data.py with this config, then run this again.")
    return False


def output_dir(cfg) -> Path:
    out = Path("outputs") / cfg.experiment
    out.mkdir(parents=True, exist_ok=True)
    return out
