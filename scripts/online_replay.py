#!/usr/bin/env python3
"""Run the online agent over the cached bars as if they were arriving live, against placebos.

    python scripts/online_replay.py --config configs/online_sp500.yaml
    python scripts/online_replay.py --config configs/online_sp500.yaml --only universe/nasdaq100_in_sp500.csv
    python scripts/online_replay.py --config configs/online_sp500.yaml --placebos 3     # a quick look

1. Replay check: a few symbols are streamed and compared with the batch pipeline sample for sample
   (features and returns must be identical), so the stream is known to see nothing early.
2. The whole universe is streamed once (cached in outputs/<experiment>/online_events.pkl).
3. Three agents run over it: full feedback with entropy control (the main run), bandit feedback with
   it, and bandit feedback without it (the ablation). Each also runs on placebo worlds where the
   returns are shuffled across stocks within each moment.

Writes outputs/<experiment>/online.md (online_<list>.md with --only, which scores only those symbols;
the agent still learns from every stock) and online_trace_<run>.csv (per day: policy entropy, bonus
weight, drift boosts, trade rate, greedy P&L). Paper positions only: nothing here places an order.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import output_dir

from src.config import load_config
from src.data import load_universe, resolve_universe
from src.dataset import build_dataset
from src.engine import score_positions
from src.online import EntropyControl, OnlineConfig, book_pnl, compare_to_batch, run_agent, stream_events

RUNS = [("full", True), ("bandit", True), ("bandit", False)]


def agent_config(o, feedback: str, entropy: bool, seed: int = 0) -> OnlineConfig:
    return OnlineConfig(cost_bps=o.cost_bps, feedback=feedback, neutral=o.neutral, hidden=o.hidden, lr=o.lr,
                        buffer=o.buffer, min_buffer=o.min_buffer, steps=o.steps, batch=o.batch,
                        warmup_moments=o.warmup_moments, drift=o.drift, seed=seed,
                        entropy=EntropyControl(target=o.target_entropy, enabled=entropy))


def run_name(feedback: str, entropy: bool) -> str:
    return f"{feedback}{'' if entropy else '_no_entropy'}"


def fmt(x, spec="+.2f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/online_sp500.yaml")
    parser.add_argument("--only", default=None, help="universe CSV (or symbols): score only these symbols")
    parser.add_argument("--placebos", type=int, default=None, help="placebo runs for the main run")
    parser.add_argument("--rebuild", action="store_true", help="stream the bars again instead of using the cache")
    args = parser.parse_args()

    cfg = load_config(args.config)
    o, out = cfg.online, output_dir(cfg)
    n_boot = int(getattr(cfg.evaluation, "bootstrap", 2000))
    symbols = resolve_universe(cfg.data.universe)
    t0 = time.time()
    bars = load_universe(cfg.data.cache_dir, symbols, cfg.data.timeframe)
    if not bars:
        print(f"no cached bars under {cfg.data.cache_dir} — run scripts/fetch_data.py first")
        return 1
    print(f"{len(bars)} symbols loaded ({time.time() - t0:.0f}s)")

    # 1. the stream against the batch pipeline, on a few symbols
    few = {s: bars[s] for s in list(bars)[: int(o.check_symbols)]}
    moments, _ = stream_events(few, cfg.features, normalize="running", keep_windows=True)
    check = compare_to_batch(moments, build_dataset(few, None, cfg))
    ok = (check["missing_in_stream"] == 0 and check["extra_with_finite_return"] == 0
          and check["max_window_diff"] < 1e-5 and check["max_return_diff"] < 1e-9)
    print(f"replay check on {len(few)} symbols: {'identical' if ok else 'MISMATCH'} {check}")
    if not ok:
        print("the stream does not reproduce the batch pipeline; not running the agent")
        return 1

    # 2. stream everything once
    cache = out / "online_events.pkl"
    if cache.exists() and not args.rebuild:
        moments, events, d = pickle.loads(cache.read_bytes())
        print(f"stream from cache: {len(moments)} moments")
    else:
        t0 = time.time()
        moments, events = stream_events(bars, cfg.features, normalize=o.normalize, min_peers=int(o.min_peers))
        d = next(iter(moments.values())).F.shape[1]
        cache.write_bytes(pickle.dumps((moments, events, d), protocol=pickle.HIGHEST_PROTOCOL))
        print(f"streamed {len(moments)} moments ({time.time() - t0:.0f}s)")
    del bars

    keep, tag = None, ""
    if args.only:
        keep, tag = set(resolve_universe(args.only)), "_" + Path(args.only).stem

    def restrict(dec):
        return dec if keep is None else dec[dec["symbol"].isin(keep)]

    # 3. the agents, real and placebo
    results = []
    for feedback, entropy in RUNS:
        name = run_name(feedback, entropy)
        main_run = (feedback, entropy) == RUNS[0]
        k = (args.placebos if args.placebos is not None else int(o.placebos)) if main_run else int(o.placebos_other)
        t0 = time.time()
        dec, trace = run_agent(moments, events, d, agent_config(o, feedback, entropy))
        dec = restrict(dec)
        books = {b: score_positions(book_pnl(dec, b, o.cost_bps), n_boot=n_boot) for b in ("explore", "greedy", "frozen")}
        placebo = []
        for seed in range(k):
            fake, _ = run_agent(moments, events, d, agent_config(o, feedback, entropy), placebo_seed=1000 + seed)
            placebo.append(float(score_positions(book_pnl(restrict(fake), "greedy", o.cost_bps), n_boot=1)["net_bps"]))
        real = books["greedy"]["net_bps"]
        p = (1 + sum(x >= real for x in placebo)) / (k + 1) if k else np.nan
        after = trace[trace["mid"] >= o.warmup_moments]
        results.append({"run": name, "main": main_run, "books": books, "placebo": placebo, "p": p,
                        "entropy": float(after["entropy"].mean()), "beta": float(after["beta"].median()),
                        "alarms": int((after["boost"].astype(int).diff() > 0).sum()),
                        "boost_share": float(after["boost"].mean())})

        dec = dec[dec["scored"]]
        pnl = (dec["pos_greedy"] * dec["ret_bps"] - o.cost_bps * dec["pos_greedy"].abs()).groupby(dec["date"]).mean()
        daily = trace.groupby("date").agg(entropy=("entropy", "mean"), beta=("beta", "mean"), boost=("boost", "max"),
                                          trades_explore=("trades_explore", "mean"))
        daily["greedy_net_bps"] = pnl.reindex(pd.to_datetime(daily.index)).to_numpy()
        daily.to_csv(out / f"online_trace_{name}{tag}.csv")
        print(f"  {name}: greedy {fmt(real)} bps [{fmt(books['greedy']['net_lo'])}, {fmt(books['greedy']['net_hi'])}], "
              f"placebo mean {fmt(np.mean(placebo) if placebo else np.nan)}, p {fmt(p, '.2f')} ({time.time() - t0:.0f}s)")

    write_report(cfg, out, tag, results, moments, check, n_boot)
    return 0


def write_report(cfg, out, tag, results, moments, check, n_boot):
    o = cfg.online
    first = min(m.date for m in moments.values())
    last = max(m.date for m in moments.values())
    lines = [f"# {cfg.experiment}: an online agent on streaming bars" + (f" ({tag[1:]})" if tag else ""), "",
             f"{len(moments)} signal moments, {first} to {last}. The agent sees one bar at a time and learns from a "
             f"trade only when its {cfg.features.horizon}-bar return has matured; scoring starts after "
             f"{o.warmup_moments} moments of warm-up, and every scored return was scored before the agent learned from it. "
             f"Rewards are market-neutral, {o.cost_bps:g} bps per position. Net is bps per stock and moment (flat counts "
             "as zero), with 95% day-block intervals."
             + (" Only the listed symbols are scored; the agent learns from every stock." if tag else ""), "",
             f"Replay check ({check['batch']} batch samples): the stream reproduces every one "
             f"(largest window difference {check['max_window_diff']:.1e}, return difference {check['max_return_diff']:.1e}).",
             "", "## Results", "",
             "**greedy**: the policy's most likely action (what would be deployed). **explore**: actions sampled from "
             "the policy (the agent that learns; exploring costs). **frozen**: the greedy policy at the end of warm-up, "
             "never updated (does continuing to learn help?). **placebo**: the same agent where each moment's returns "
             "are shuffled across its stocks (same features, volatility and timing, nothing to learn); p is the share of "
             "placebo runs that did at least as well, counting the real run.", "",
             "| run | greedy net [95% CI] · trades | explore net · trades | frozen net · trades | placebo greedy mean (runs) | p |",
             "|---|---|---|---|---|---:|"]
    for r in results:
        b = r["books"]
        cell = lambda s: f"{fmt(s['net_bps'])} · {s['trade_rate']:.0%}"
        lines.append(f"| {'**' + r['run'] + '**' if r['main'] else r['run']} | {fmt(b['greedy']['net_bps'])} "
                     f"[{fmt(b['greedy']['net_lo'])}, {fmt(b['greedy']['net_hi'])}] · {b['greedy']['trade_rate']:.0%} | "
                     f"{cell(b['explore'])} | {cell(b['frozen'])} | "
                     f"{fmt(np.mean(r['placebo']) if r['placebo'] else np.nan)} ({len(r['placebo'])}) | {fmt(r['p'], '.2f')} |")
    lines += ["", "## Exploration", "",
              f"Policy entropy in nats after warm-up (target {o.target_entropy}; ln 3 = 1.10 is uniform). **beta** is the "
              "entropy bonus weight the controller settled on; **drift boosts** counts the times a drift detection raised the target "
              "for a while.", "",
              "| run | mean entropy | median beta | drift boosts started | share of moments boosted |", "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['run']} | {r['entropy']:.2f} | {r['beta']:.3f} | {r['alarms']} | {r['boost_share']:.0%} |")
    main = next(r for r in results if r["main"])
    g = main["books"]["greedy"]
    found = g["net_lo"] > 0 and len(main["placebo"]) >= 19 and main["p"] <= 0.05
    lines += ["", "## Verdict (rule in the config, fixed before running)", "",
              f"Main run `{main['run']}`: greedy net {fmt(g['net_bps'])} bps [{fmt(g['net_lo'])}, {fmt(g['net_hi'])}], "
              f"placebo p = {fmt(main['p'], '.2f')} over {len(main['placebo'])} placebos. "
              + ("**Found something**: the interval is above zero and it beats the placebos." if found else
                 "**A null**: " + ("the interval is not above zero" if g["net_lo"] <= 0 else "the interval is above zero")
                 + (" and " if g["net_lo"] <= 0 else " but ")
                 + ("it does not beat the placebos." if not (main["p"] <= 0.05 and len(main["placebo"]) >= 19)
                    else "it beats the placebos.")),
              ""]
    text = "\n".join(lines) + "\n"
    (out / f"online{tag}.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    raise SystemExit(main())
