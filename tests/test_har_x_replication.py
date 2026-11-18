"""The frozen replication matches the large-cap model, and its verdict follows the rule."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

import explore_har_x
import har_x_replication as script
from src import volforecast as vf
from test_volforecast import CFG, DAYS, panel


def test_the_protocol_is_the_large_cap_model():
    doc = script.load()
    assert doc["universe"] == "universe/sp400.csv" and len(doc["extra"]) == 7


def test_the_verdict_needs_both_intervals_below_zero():
    below, above = {"hi": -0.001}, {"hi": 0.001}
    assert script.verdict(below, below) == "REPLICATES"
    assert script.verdict(below, above) == script.verdict(above, below) == "PARTLY REPLICATES"
    assert script.verdict(above, above) == "DOES NOT REPLICATE"


def test_it_runs_end_to_end(capsys):
    p = panel(n=40, seed=6)
    doc = {**script.load(), "first_target": str(DAYS[300].date()), "last_target": str(DAYS[-1].date())}
    rows = explore_har_x.add_features(vf.build_rows(p, CFG), p)
    doc["extra"] = [c for c in doc["extra"] if c in rows]
    doc["large_cap_signs"] = {k: v for k, v in doc["large_cap_signs"].items() if k in rows}
    r = script.run(rows, doc, CFG)
    script.report(r)
    assert "VERDICT:" in capsys.readouterr().out and r["verdict"] in {"REPLICATES", "PARTLY REPLICATES", "DOES NOT REPLICATE"}
