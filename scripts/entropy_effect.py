#!/usr/bin/env python3
"""What does permutation entropy add to the trees? `gbm` (entropy-free run) against `gbm_ent`, same rows.

    python scripts/entropy_effect.py --free configs/sip_pe_free.yaml --with configs/sip_sp500_multihour.yaml

Reads the saved predictions of both runs (no training), keeps the rows both scored, and compares the two
models' gross bps per trade with a paired day-block bootstrap: each resample draws whole days and scores both
models on the same days. Writes <free output>/entropy_effect.md. The rule is in configs/sip_pe_free.yaml.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import output_dir

from src.config import load_config
from src.experiment import KEY, ensemble, load_predictions
from src.stats import trades


def day_table(pred: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
    t = trades(pred)
    g = t.groupby("date").agg(pnl=("gross_bps", "sum"), n=("gross_bps", "size"), hits=("correct", "sum"))
    return g.reindex(days, fill_value=0)


def paired(a: pd.DataFrame, b: pd.DataFrame, days: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    ta, tb = day_table(a, days), day_table(b, days)
    idx = np.random.default_rng(seed).integers(0, len(days), (n_boot, len(days)))

    def stat(t, col):
        s, n = t[col].to_numpy(float), t["n"].to_numpy(float)
        return s.sum() / n.sum(), s[idx].sum(axis=1) / n[idx].sum(axis=1)

    out = {}
    for name, col in (("gross_bps", "pnl"), ("accuracy", "hits")):
        (pa, ba), (pb, bb) = stat(ta, col), stat(tb, col)
        d = pb - pa
        out[name] = {"free": pa, "with": pb, "diff": d, "lo": float(np.percentile(bb - ba, 2.5)),
                     "hi": float(np.percentile(bb - ba, 97.5))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free", default="configs/sip_pe_free.yaml")
    parser.add_argument("--with", dest="with_", default="configs/sip_sp500_multihour.yaml")
    parser.add_argument("--free-arch", default="gbm")
    parser.add_argument("--with-arch", default="gbm_ent")
    args = parser.parse_args()
    cf, cw = load_config(args.free), load_config(args.with_)
    pf, pw = load_predictions(output_dir(cf)), load_predictions(output_dir(cw))
    if pf.empty or pw.empty:
        print("missing predictions: run notebook sections 32 and 33 first")
        return 1
    a, b = pf[pf["arch"] == args.free_arch], pw[pw["arch"] == args.with_arch]
    if a.empty or b.empty:
        print(f"need {args.free_arch} in the entropy-free run and {args.with_arch} in the other")
        return 1
    a, b = ensemble(a), ensemble(b)
    both = a[KEY].merge(b[KEY], on=KEY)
    a, b = a.merge(both, on=KEY), b.merge(both, on=KEY)
    if len(both) < 0.99 * min(len(ensemble(pf[pf["arch"] == args.free_arch])), len(ensemble(pw[pw["arch"] == args.with_arch]))):
        print("the two runs scored different rows")
        return 1
    # same rows, same labels: the only difference between the two is the entropy inputs
    a, b = a.sort_values(KEY).reset_index(drop=True), b.sort_values(KEY).reset_index(drop=True)
    same = bool(np.allclose(a["ret"], b["ret"]))
    days = np.unique(a["date"])
    r = paired(a, b, days)
    g, acc = r["gross_bps"], r["accuracy"]
    verdict = ("entropy adds nothing" if g["lo"] <= 0 <= g["hi"]
               else "entropy helps" if g["lo"] > 0 else "entropy hurts")
    f = lambda d, s="+.2f": f"{d['diff']:{s}} [{d['lo']:{s}}, {d['hi']:{s}}]"
    lines = [f"# What permutation entropy adds to the trees: {args.free_arch} against {args.with_arch}", "",
             f"**Verdict: {verdict.upper()}** on gross bps per trade (paired difference, 95% day-block interval).", "",
             f"{len(a):,} rows both runs scored over {len(days):,} days; labels identical: {same}.", "",
             "| | entropy-free | with entropy | difference [95%] |", "|---|---:|---:|---|",
             f"| gross bps per trade | {g['free']:+.2f} | {g['with']:+.2f} | {f(g)} |",
             f"| accuracy | {acc['free']:.4f} | {acc['with']:.4f} | {f(acc, '+.4f')} |", "",
             f"Net at 2 bps: {g['free'] - 2:+.2f} without entropy, {g['with'] - 2:+.2f} with."]
    text = "\n".join(lines) + "\n"
    (output_dir(cf) / "entropy_effect.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
