"""The exploration runs end to end on a synthetic panel and counts what it tried."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

import explore_pe_long as script
from src import volforecast as vf
from test_volforecast import CFG, DAYS, panel


def test_the_exploration_runs_and_reports_every_variant(capsys):
    p = panel(n=40, seed=2)
    rows = vf.build_rows(p, CFG)
    script.analyse(rows, p.pivot(index="date", columns="symbol", values="ret"), CFG, DAYS[300], DAYS[-1])
    out = capsys.readouterr().out
    for part in ("1. How PE", "2. Out of sample", "3. Is it PE", "4. Where it lives", "VARIANTS TRIED: 6"):
        assert part in out
