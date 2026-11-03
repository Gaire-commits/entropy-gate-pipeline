#!/usr/bin/env python3
"""Finalize the portfolio's rules on development and validation (protocols/portfolio_rule_search.yaml). Prints every
candidate; loads nothing after the lock.

    python scripts/portfolio_rules.py --reference data/reference
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import yaml

from _common import ROOT

from src import audit, portfolio as pf, volpanel


def information_ratio(active: pd.Series) -> float:
    return float(active.mean() / active.std() * np.sqrt(252)) if active.std() > 0 else float("nan")


def load_market(reference, cfg: pf.Config, score_holdout: bool = False) -> pf.Market:
    daily = {t: pd.read_parquet(f"{reference}/daily/daily_sip_{t}.parquet") for t in ("raw", "split", "all")}
    actions = audit.normalize_actions(pd.read_parquet(f"{reference}/corporate_actions.parquet"))
    universe = pd.read_csv(ROOT / "universe" / "sp500.csv")
    registry = volpanel.load_registry(ROOT / "registry" / "vol_registry.csv")
    joined = pf.load_joined(ROOT / "registry" / "sp500_join_dates.csv")
    return pf.build_market(daily["raw"], daily["split"], daily["all"], actions, registry, universe, until=pf.lock_date(cfg, score_holdout), joined=joined)


def search(m: pf.Market, doc: dict) -> pd.DataFrame:
    """Every candidate on both windows, with its score under the protocol's rule."""
    grid, spans = doc["grid"], {k: tuple(pd.Timestamp(x) for x in v) for k, v in doc["windows"].items()}
    base = replace(pf.Config(sector_cap=doc["fixed"]["sector_cap"], execution=doc["fixed"]["execution"]), **doc["costs"])
    rows = []
    for cadence in grid["cadence_sessions"]:
        cfg = replace(base, rebalance_every=cadence, horizon=cadence, tranches=grid["tranches_by_cadence"][cadence])
        feats, eligible = pf.build_features(m, cfg)
        target = pf.forward_returns(m, cfg)
        dates = pf.decision_dates(m.close.index, cfg)
        end = pf.scored_end(m.close.index, cfg)
        forecasts, _ = pf.walk_forward(feats, eligible, target, pf.MODELS[doc["fixed"]["model"]], dates, cfg)
        ic = pf.information_coefficient(forecasts, target)
        ew = pf.simulate(pf.equal_weight_targets(eligible, dates), m, cfg, end)
        for top, mult in itertools.product(grid["top_frac"], grid["hold_multiple"]):
            rule = replace(cfg, top_frac=top, hold_frac=top * mult)
            book = pf.simulate(pf.portfolio_targets(forecasts, m, rule), m, rule, end)
            row = {"cadence": cadence, "tranches": cfg.tranches, "top_frac": top, "hold_frac": top * mult}
            for name, (a, b) in spans.items():
                d = book.loc[a:b]
                active = d["net"] - ew["net"].reindex(d.index).fillna(0.0)
                row[f"ir_{name[:3]}"] = information_ratio(active)
                row[f"net_{name[:3]}"] = pf.performance(d["net"])["return_ann"]
                row[f"turn_{name[:3]}"] = d["turnover"].sum() * 252 / len(d)
                row[f"ic_{name[:3]}"] = ic.loc[a:b].mean()
            rows.append(row)
    out = pd.DataFrame(rows)
    out["score"] = out[["ir_dev", "ir_val"]].min(axis=1)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def choose(table: pd.DataFrame, tie: float = 0.05) -> pd.Series:
    """The protocol's rule: the highest score; within `tie` of it, the lowest validation turnover."""
    near = table[table["score"] >= table["score"].max() - tie]
    return near.sort_values("turn_val", kind="stable").iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference")
    ap.add_argument("--protocol", default=str(ROOT / "protocols" / "portfolio_rule_search_v2.yaml"))
    args = ap.parse_args()
    with open(args.protocol) as f:
        doc = yaml.safe_load(f)
    m = load_market(args.reference, pf.Config())
    print(f"data through {m.close.index.max().date()}; {len(doc['grid']['cadence_sessions']) * len(doc['grid']['top_frac']) * len(doc['grid']['hold_multiple'])} candidates, model {doc['fixed']['model']}")
    table = search(m, doc)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(table.round(4).to_string(index=False))
    pick = choose(table)
    print(f"\nchosen by the rule: hold {int(pick['cadence'])} sessions in {int(pick['tranches'])} staggered tranche(s), best {pick['top_frac']:.0%}, keep while in the best {pick['hold_frac']:.0%}")
    print(f"score {pick['score']:.3f} (information ratio against equal-weight: development {pick['ir_dev']:.3f}, validation {pick['ir_val']:.3f}); this is the best of {len(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
