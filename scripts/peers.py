#!/usr/bin/env python3
"""E12: does a stock follow the stocks it actually moves with?

    python scripts/peers.py --config configs/sip_e12_peers.yaml

Reads the cached SIP bars (no training). Writes outputs/<experiment>/peers.md, peers.csv and
refreshes.csv (one row per monthly peer refresh). The method is src/peers.py; the rule is in the config.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import enough_symbols, output_dir

from src.capacity import paired_ic_difference
from src.config import load_config
from src.crosssection import cross_sectional_ic, quantile_book, score_book
from src.data import load_universe, resolve_universe
from src.peers import collect, residualize, verdict


def fmt(x, spec="+.4f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def scored(f: pd.DataFrame, values) -> pd.DataFrame:
    return f[["date", "bar_index", "symbol", "ret"]].assign(prob=np.asarray(values, dtype=np.float32))


def compare(f: pd.DataFrame, a: str, b: str, e, n_boot: int) -> dict:
    """Residualized `a` against residualized `b` on the same moments: the full span at the rule's level,
    and each half at the halves' level."""
    fa, fb = scored(f, residualize(f, a)), scored(f, residualize(f, b))
    kw = dict(min_peers=e.min_peers, n_boot=n_boot)
    split = pd.Timestamp(e.split)
    early, late = f["date"] < split, f["date"] >= split
    return {"all": paired_ic_difference(fa, fb, level=e.level, **kw),
            "early": paired_ic_difference(fa[early], fb[early], level=e.half_level, **kw),
            "late": paired_ic_difference(fa[late], fb[late], level=e.half_level, **kw),
            "ic_a": cross_sectional_ic(fa, min_peers=e.min_peers, n_boot=n_boot),
            "ic_b": cross_sectional_ic(fb, min_peers=e.min_peers, n_boot=n_boot), "frame": fa}


def diff(d: dict) -> str:
    return f"{fmt(d.get('diff'))} [{fmt(d.get('lo'))}, {fmt(d.get('hi'))}]"


def ic(s: dict) -> str:
    return f"{fmt(s.get('ic'))} [{fmt(s.get('ic_lo'))}, {fmt(s.get('ic_hi'))}], t {fmt(s.get('t'), '+.1f')}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sip_e12_peers.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, e = output_dir(cfg), cfg.peers
    n_boot = int(cfg.evaluation.bootstrap)
    universe = resolve_universe(cfg.data.universe)
    t0 = time.time()
    bars = load_universe(cfg.data.cache_dir, universe, cfg.data.timeframe)
    if not bars:
        print(f"no cached bars under {cfg.data.cache_dir} — fetch the SIP bars first (Colab section 21)")
        return 1
    if not enough_symbols(bars, universe, cfg.data.cache_dir):
        return 1
    table = pd.read_csv(cfg.data.universe)
    sectors = dict(zip(table["symbol"], table["sector"]))
    print(f"{len(bars)} symbols loaded ({time.time() - t0:.0f}s)")
    frame, refreshes = collect(bars, sectors, tuple(e.ks), tuple(e.holds), e.lookback, e.embargo, e.stride,
                               e.peer_days, e.min_share, e.min_valid, e.min_sector, e.seed)
    del bars
    print(f"{len(frame):,} stock-moments, {len(refreshes)} peer refreshes ({time.time() - t0:.0f}s)")

    k, other_k = e.k, [x for x in e.ks if x != e.k][0]
    other_h = [h for h in e.holds if h != e.hold][0]
    prim = frame[np.isfinite(frame[f"ret_{e.hold}"])].assign(ret=lambda x: x[f"ret_{e.hold}"]).reset_index(drop=True)
    main_cmp = compare(prim, f"peers_{k}", f"random_{k}", e, n_boot)
    v = verdict(main_cmp["all"], main_cmp["early"], main_cmp["late"])
    print(f"primary done ({time.time() - t0:.0f}s)")
    book = score_book(quantile_book(main_cmp.pop("frame"), q=e.q, min_peers=e.min_peers), e.cost_bps, n_boot)
    more_k = compare(prim, f"peers_{other_k}", f"random_{other_k}", e, n_boot)
    sec = prim[np.isfinite(prim["sector"])].reset_index(drop=True)
    sector_vs_random = compare(sec, "sector", f"random_{k}", e, n_boot)
    peers_vs_sector = compare(sec, f"peers_{k}", "sector", e, n_boot)
    del sec
    own = cross_sectional_ic(scored(prim, prim["own"]), min_peers=e.min_peers, n_boot=n_boot)
    raw = cross_sectional_ic(scored(prim, prim[f"peers_{k}"]), min_peers=e.min_peers, n_boot=n_boot)
    rows = len(prim)
    del prim
    longer = frame[np.isfinite(frame[f"ret_{other_h}"])].assign(ret=lambda x: x[f"ret_{other_h}"]).reset_index(drop=True)
    del frame
    hour = compare(longer, f"peers_{k}", f"random_{k}", e, n_boot)
    del longer
    print(f"reported done ({time.time() - t0:.0f}s)")

    for c in (more_k, sector_vs_random, peers_vs_sector, hour):
        c.pop("frame", None)
    r = refreshes
    lines = [f"# {cfg.experiment}: does a stock follow the stocks it actually moves with?", "",
             f"**Verdict: {v['level'].upper()}** — {v['reason']}. "
             f"{'The correlated-peers arm goes into the forward test.' if v['live'] else 'The correlated-peers arm does not go into the forward test.'}",
             "",
             f"{rows:,} stock-moments. Peers: each stock's {k} most correlated stocks (5-minute returns relative to the "
             f"market, previous {e.peer_days} trading days), refreshed monthly ({len(r)} refreshes). The peers' move over "
             f"the last {e.lookback} bars, relative to the market and **residualized on the stock's own move** at each "
             f"moment, scored against the next {e.hold} bars (entry {e.embargo} bar later). Control: {k} random peers per "
             "stock, redrawn monthly, on the same moments.", "",
             "## The rule", "",
             "| | rank IC of the residual move [95%] |", "|---|---|",
             f"| real peers | {ic(main_cmp['ic_a'])} |", f"| random peers | {ic(main_cmp['ic_b'])} |", "",
             f"- **Real minus random** ({e.level:.0%}): {diff(main_cmp['all'])} over {main_cmp['all'].get('days', 0)} days.",
             f"- **Halves** ({e.half_level:.0%}): before {e.split} {diff(main_cmp['early'])}; from it {diff(main_cmp['late'])}.",
             "", "## Reported", "",
             "| comparison (residual IC, same moments) | difference [99%] |", "|---|---|",
             f"| {other_k} real peers minus {other_k} random | {diff(more_k['all'])} |",
             f"| {k} real minus {k} random, held {other_h} bars | {diff(hour['all'])} |",
             f"| sector average (E5's peers) minus {k} random | {diff(sector_vs_random['all'])} |",
             f"| {k} correlated peers minus the sector average | {diff(peers_vs_sector['all'])} |", "",
             f"- **The own reversal** (E5's signal, same moments): {ic(own)}.",
             f"- **The raw peers' move, not residualized** (it carries the stock's own move): {ic(raw)}.",
             f"- **Long-short book on the real residual** (top and bottom {e.q:.0%}, {e.cost_bps:g} bps): gross "
             f"{fmt(book.get('gross'), '+.2f')} bps [{fmt(book.get('gross_lo'), '+.2f')}, {fmt(book.get('gross_hi'), '+.2f')}], "
             f"net {fmt(book.get('net'), '+.2f')} bps per position.",
             f"- **The peers themselves:** {fmt(r['eligible'].mean(), '.0f')} stocks eligible per refresh; the chosen peers' "
             f"average correlation {fmt(r['peer_corr'].mean(), '.2f')}; {fmt(100 * r['same_sector'].mean(), '.0f')}% of them in "
             "the stock's own sector."]
    text = "\n".join(lines) + "\n"
    (out / "peers.md").write_text(text)
    refreshes.to_csv(out / "refreshes.csv", index=False)
    pd.DataFrame([{"comparison": name, **{f"{p}_{kk}": x for p in ("all", "early", "late") for kk, x in c[p].items()},
                   **{f"a_{kk}": x for kk, x in c["ic_a"].items()}, **{f"b_{kk}": x for kk, x in c["ic_b"].items()}}
                  for name, c in (("primary", main_cmp), (f"k{other_k}", more_k), (f"hold{other_h}", hour),
                                  ("sector_vs_random", sector_vs_random), ("peers_vs_sector", peers_vs_sector))]
                 ).to_csv(out / "peers.csv", index=False)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
