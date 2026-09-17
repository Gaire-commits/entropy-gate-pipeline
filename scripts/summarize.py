#!/usr/bin/env python3
"""Turn saved predictions into the results tables.

    python scripts/summarize.py --config configs/spy_gao.yaml

Writes outputs/<experiment>/summary.md, overall.csv and gate.csv.
"""

from __future__ import annotations

import argparse

from _common import DEFAULT_CONFIG, output_dir

from src.config import load_config
from src.experiment import load_predictions, markdown, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = output_dir(cfg)
    pred = load_predictions(out)
    if pred.empty:
        print(f"no predictions under {out / 'predictions'} — run scripts/sweep.py first")
        return 1

    overall, gate = report(pred, cfg)
    overall.to_csv(out / "overall.csv", index=False)
    gate.to_csv(out / "gate.csv", index=False)
    text = markdown(overall, gate, cfg)
    (out / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
