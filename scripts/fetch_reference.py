#!/usr/bin/env python3
"""Reference data for the data audit, straight from Alpaca. The 5-minute caches are not touched.

    python scripts/fetch_reference.py --universe universe/sp500.csv --start 2016-01-04 --end 2026-10-13

Writes under data/reference/ (on Drive in Colab):
  daily/daily_sip_<setting>.parquet   SIP daily bars for every symbol, one file per setting: adjustment
      raw, split, split+dividend and all (symbol mapping as of the retrieval date, Alpaca's default), and
      split with symbol mapping off (asof "-"), which shows what each ticker traded as on its own
  corporate_actions.parquet           every corporate action Alpaca returns for these symbols from 2015,
      plus the older symbols its name changes point to
  sample/ext_5min.parquet             5-minute bars including extended hours for a fixed sample of symbols
      and days (every half-day plus seeded full days), to align session definitions
  sample/auctions.parquet             opening and closing auction prints for the sample symbols, every day
  sample/trades_open_close.parquet    trades around the open and the close, for a smaller subsample
  meta/conditions_{A,B,C}.json, meta/exchanges.json   code tables for the trades and auctions
  auctions.parquet                    opening and closing auction prints for every symbol, every day (only with
      --parts auctions; scripts/official_prices.py turns them into the official open and close)
  manifest.json                       for every file: endpoint, request settings, retrieval time, request
      and row counts, symbols missing, and a SHA-256 of the file

A file is written only when its download has finished (written aside, then renamed), and existing files
are kept unless --force, so a dropped session resumes where it stopped. The sample is drawn with a fixed
seed before any of it is looked at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import reachable

from src import nyse
from src.alpaca_rest import BASE_URL, DOCS, AlpacaREST, daily_frame, rfc3339
from src.data import resolve_universe

DAILY = [("raw", None), ("split", None), ("split,dividend", None), ("all", None), ("split", "-")]
BATCH = 50


def make_client() -> AlpacaREST:
    return AlpacaREST()


def tag(adjustment: str, asof: str | None) -> str:
    return adjustment.replace(",", "+") + ("_nomap" if asof == "-" else "")


def now() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def batches(items: list, n: int = BATCH):
    return [items[i:i + n] for i in range(0, len(items), n)]


def day_window(day) -> tuple[str, str]:
    """A New York calendar day as an inclusive RFC-3339 interval (a daily bar is stamped at its 00:00)."""
    day = pd.Timestamp(day).normalize()
    return rfc3339(day), rfc3339(day + pd.Timedelta(hours=23, minutes=59, seconds=59))


def record(manifest: dict, out: Path, path: Path, **info) -> None:
    info.update(sha256=sha256(path), bytes=path.stat().st_size)
    manifest["files"][str(path.relative_to(out))] = info
    save_json(manifest, out / "manifest.json")


def failed(manifest: dict, out: Path, path: Path, exc: Exception, **info) -> None:
    """A part that could not be fetched: recorded with the server's message instead of stopping the run."""
    info.update(error=f"{type(exc).__name__}: {str(exc)[:500]}", failed_at=now())
    manifest["files"][str(path.relative_to(out))] = info
    save_json(manifest, out / "manifest.json")
    print(f"could not fetch {path.name}: {info['error']} (recorded in the manifest; the rest goes on)")


def fetch_daily(client, symbols, start, end, out, manifest, force) -> None:
    lo, _ = day_window(start)
    _, hi = day_window(end)
    for adjustment, asof in DAILY:
        path = out / "daily" / f"daily_sip_{tag(adjustment, asof)}.parquet"
        if path.exists() and not force:
            print(f"kept {path}")
            continue
        began, n0, frames = now(), len(client.log), []
        for i, group in enumerate(batches(symbols)):
            frames.append(client.bars(group, "1Day", lo, hi, adjustment, feed="sip", asof=asof))
            print(f"  {tag(adjustment, asof)}: {min((i + 1) * BATCH, len(symbols))}/{len(symbols)} symbols", end="\r")
        df = daily_frame(pd.concat(frames, ignore_index=True))
        save(df, path)
        got = set(df["symbol"])
        record(manifest, out, path, endpoint="GET /v2/stocks/bars", docs=DOCS["bars"], rules=DOCS["bar_rules"],
               settings={"timeframe": "1Day", "adjustment": adjustment, "feed": "sip",
                         "asof": "- (symbol mapping off)" if asof == "-" else "not sent: Alpaca's default, symbol "
                                 "mapping as of the retrieval date", "start": lo, "end": hi, "limit": 10000,
                         "sort": "asc", "symbols_per_request": BATCH},
               retrieved_at={"start": began, "end": now()}, requests=len(client.log) - n0, rows=len(df),
               symbols_requested=len(symbols), symbols_returned=len(got),
               missing_symbols=sorted(set(symbols) - got),
               first_date=str(df["date"].min().date()) if len(df) else None,
               last_date=str(df["date"].max().date()) if len(df) else None)
        print(f"wrote {path}: {len(df):,} bars, {len(got)} of {len(symbols)} symbols")


def fetch_corporate_actions(client, symbols, out, manifest, force) -> None:
    path = out / "corporate_actions.parquet"
    if path.exists() and not force:
        print(f"kept {path}")
        return
    start, end = "2015-01-01", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    began, n0, notes = now(), len(client.log), []

    def query(group):
        try:
            return client.corporate_actions(group, start, end, data_quality="all")
        except RuntimeError as exc:  # an older API version may not know data_quality
            notes.append(f"data_quality=all refused ({str(exc)[:120]}); repeated without it")
            return client.corporate_actions(group, start, end, data_quality=None)

    frames = [query(g) for g in batches(symbols)]
    ca = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    universe = set(symbols)
    older = []
    if len(ca) and {"old_symbol", "new_symbol"} <= set(ca.columns):
        renamed = ca[ca["new_symbol"].isin(universe)]
        older = sorted(set(renamed["old_symbol"].dropna()) - universe)
    if older:
        ca = pd.concat([ca] + [query(g) for g in batches(older)], ignore_index=True)
    if "id" in ca.columns:
        ca = ca.drop_duplicates("id")
    for col in ca.columns:
        if ca[col].dtype == object:
            ca[col] = ca[col].map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    save(ca, path)
    record(manifest, out, path, endpoint="GET /v1/corporate-actions", docs=DOCS["corporate_actions"],
           settings={"start": start, "end": end, "types": "all (not sent)", "data_quality": "all",
                     "limit": 1000, "symbols_per_request": BATCH, "second_pass_old_symbols": older},
           retrieved_at={"start": began, "end": now()}, requests=len(client.log) - n0, rows=len(ca),
           kinds=ca["kind"].value_counts().to_dict() if "kind" in ca.columns else {}, notes=notes)
    print(f"wrote {path}: {len(ca):,} corporate actions ({len(older)} older symbols added)")


def fetch_meta(client, out, manifest, force) -> None:
    for name, call, doc in [(f"conditions_{t}", (lambda t=t: client.conditions(t)), DOCS["conditions"]) for t in "ABC"] \
            + [("exchanges", client.exchanges, DOCS["exchanges"])]:
        path = out / "meta" / f"{name}.json"
        if path.exists() and not force:
            continue
        began = now()
        save_json(call(), path)
        record(manifest, out, path, endpoint="GET /v2/stocks/meta/" + ("exchanges" if name == "exchanges" else
                                                                     "conditions/trade"), docs=doc,
               retrieved_at={"start": began, "end": now()})


def sample_plan(symbols: list[str], start, end, seed: int, n_symbols: int, n_full: int, n_trades: int) -> dict:
    """Which symbols and days the session study uses, drawn once from the seed."""
    rng = np.random.default_rng(seed)
    pool = sorted(symbols)
    chosen = sorted(rng.choice(pool, size=min(n_symbols, len(pool)), replace=False).tolist())
    days = nyse.sessions(start, end)
    half = days[nyse.is_half_day(days)]
    full = days[~nyse.is_half_day(days)]
    picked = pd.DatetimeIndex(sorted(rng.choice(full.values, size=min(n_full, len(full)), replace=False)))
    trade_days = list(half[:3]) + list(picked[:3])
    return {"seed": seed, "symbols": chosen, "half_days": [str(d.date()) for d in half],
            "full_days": [str(d.date()) for d in picked], "trade_symbols": chosen[:n_trades],
            "trade_days": [str(pd.Timestamp(d).date()) for d in trade_days]}


def fetch_sample(client, symbols, start, end, out, manifest, force, seed, n_symbols, n_full, n_trades) -> None:
    plan = sample_plan(symbols, start, end, seed, n_symbols, n_full, n_trades)
    save_json(plan, out / "sample" / "plan.json")
    days = sorted(plan["half_days"] + plan["full_days"])

    path = out / "sample" / "ext_5min.parquet"
    if force or not path.exists():
        began, n0 = now(), len(client.log)
        frames = [client.bars(plan["symbols"], "5Min", *day_window(d), "split", feed="sip") for d in days]
        df = pd.concat(frames, ignore_index=True)
        save(df, path)
        record(manifest, out, path, endpoint="GET /v2/stocks/bars", docs=DOCS["bars"],
               settings={"timeframe": "5Min", "adjustment": "split", "feed": "sip", "asof": "not sent (default)",
                         "hours": "every bar Alpaca returns for the New York day, extended hours included",
                         "symbols": plan["symbols"], "days": days, "seed": seed},
               retrieved_at={"start": began, "end": now()}, requests=len(client.log) - n0, rows=len(df))
        print(f"wrote {path}: {len(df):,} bars")

    path = out / "sample" / "auctions.parquet"
    if force or not path.exists():
        began, n0 = now(), len(client.log)
        lo, _ = day_window(start)
        _, hi = day_window(end)
        settings = {"feed": "sip", "asof": "not sent (default)", "start": lo, "end": hi, "symbols": plan["symbols"]}
        try:
            df = client.auctions(plan["symbols"], lo, hi)
        except Exception as exc:  # noqa: BLE001  (the audit runs without this part)
            failed(manifest, out, path, exc, endpoint="GET /v2/stocks/auctions", docs=DOCS["auctions"],
                   settings=settings, retrieved_at={"start": began}, requests=len(client.log) - n0)
        else:
            save(df, path)
            record(manifest, out, path, endpoint="GET /v2/stocks/auctions", docs=DOCS["auctions"], settings=settings,
                   retrieved_at={"start": began, "end": now()}, requests=len(client.log) - n0, rows=len(df))
            print(f"wrote {path}: {len(df):,} auction prints")

    path = out / "sample" / "trades_open_close.parquet"
    if force or not path.exists():
        began, n0, frames, windows = now(), len(client.log), [], []
        try:
            for d in plan["trade_days"]:
                day = pd.Timestamp(d)
                close = day + nyse.close_time(day)
                for name, lo, hi in (("open", day + pd.Timedelta(hours=9, minutes=29, seconds=30),
                                      day + pd.Timedelta(hours=9, minutes=31)),
                                     ("close", close - pd.Timedelta(seconds=30), close + pd.Timedelta(minutes=15))):
                    windows.append({"day": d, "window": name, "start": rfc3339(lo), "end": rfc3339(hi)})
                    frames.append(client.trades(plan["trade_symbols"], rfc3339(lo), rfc3339(hi)).assign(window=name))
        except Exception as exc:  # noqa: BLE001
            failed(manifest, out, path, exc, endpoint="GET /v2/stocks/trades", docs=DOCS["trades"],
                   settings={"feed": "sip", "symbols": plan["trade_symbols"], "windows": windows},
                   retrieved_at={"start": began}, requests=len(client.log) - n0)
            return
        df = pd.concat(frames, ignore_index=True)
        save(df, path)
        record(manifest, out, path, endpoint="GET /v2/stocks/trades", docs=DOCS["trades"],
               settings={"feed": "sip", "asof": "not sent (default)", "symbols": plan["trade_symbols"],
                         "windows": windows}, retrieved_at={"start": began, "end": now()},
               requests=len(client.log) - n0, rows=len(df))
        print(f"wrote {path}: {len(df):,} trades")


def fetch_auctions(client, symbols, start, end, out, manifest, force) -> None:
    """Opening and closing auction prints for every symbol, every day: the official open and close. Fetched 50 symbols
    at a time; each finished batch is kept aside, so a dropped session resumes, and the batches are combined at the end."""
    path = out / "auctions.parquet"
    if path.exists() and not force:
        print(f"kept {path}")
        return
    parts = out / "auctions_parts"
    lo, _ = day_window(start)
    _, hi = day_window(end)
    settings = {"feed": "sip", "asof": "not sent (default)", "start": lo, "end": hi, "limit": 10000,
                "symbols_per_request": BATCH}
    began, n0 = now(), len(client.log)
    groups = batches(symbols)
    for i, group in enumerate(groups):
        part = parts / f"batch_{i:03d}.parquet"
        if part.exists() and not force:
            continue
        try:
            save(client.auctions(group, lo, hi), part)
        except Exception as exc:  # noqa: BLE001  (finished batches stay; the next run retries from here)
            failed(manifest, out, path, exc, endpoint="GET /v2/stocks/auctions", docs=DOCS["auctions"],
                   settings=settings, retrieved_at={"start": began}, requests=len(client.log) - n0,
                   batches_done=i, batches=len(groups))
            return
        print(f"  auctions: {min((i + 1) * BATCH, len(symbols))}/{len(symbols)} symbols", end="\r")
    df = pd.concat([pd.read_parquet(parts / f"batch_{i:03d}.parquet") for i in range(len(groups))], ignore_index=True)
    save(df, path)
    got = set(df["symbol"])
    record(manifest, out, path, endpoint="GET /v2/stocks/auctions", docs=DOCS["auctions"], settings=settings,
           retrieved_at={"start": began, "end": now()}, requests=len(client.log) - n0, rows=len(df),
           symbols_requested=len(symbols), symbols_returned=len(got), missing_symbols=sorted(set(symbols) - got))
    for f in parts.glob("batch_*.parquet"):
        f.unlink()
    parts.rmdir()
    print(f"wrote {path}: {len(df):,} auction prints, {len(got)} of {len(symbols)} symbols")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="universe/sp500.csv")
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2026-10-13", help="last day, inclusive")
    parser.add_argument("--out", default="data/reference")
    parser.add_argument("--parts", default="daily,corporate_actions,meta,sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-symbols", type=int, default=20)
    parser.add_argument("--sample-full-days", type=int, default=25)
    parser.add_argument("--trade-symbols", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="download again even if a file exists")
    args = parser.parse_args()

    out = Path(args.out)
    if not reachable(args.out):
        return 1
    out.mkdir(parents=True, exist_ok=True)
    symbols = resolve_universe(args.universe)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    import importlib.metadata as md
    versions = {}
    for pkg in ("pandas", "numpy", "requests", "pyarrow"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    manifest.update(
        source=f"Alpaca Market Data API, {BASE_URL}; feed sip = the consolidated tape of all US exchanges "
               "(the free plan serves SIP history older than 15 minutes).",
        universe=args.universe, universe_symbols=len(symbols), start=args.start, end=args.end,
        created_by="scripts/fetch_reference.py",
        environment={"python": platform.python_version(), **versions})
    manifest.setdefault("runs", []).append({"started": now(), "parts": args.parts, "force": args.force})
    save_json(manifest, manifest_path)

    client = make_client()
    t0 = time.time()
    parts = set(args.parts.split(","))
    if "daily" in parts:
        fetch_daily(client, symbols, args.start, args.end, out, manifest, args.force)
    if "corporate_actions" in parts:
        fetch_corporate_actions(client, symbols, out, manifest, args.force)
    if "meta" in parts:
        fetch_meta(client, out, manifest, args.force)
    if "sample" in parts:
        fetch_sample(client, symbols, args.start, args.end, out, manifest, args.force, args.seed,
                     args.sample_symbols, args.sample_full_days, args.trade_symbols)
    if "auctions" in parts:
        fetch_auctions(client, symbols, args.start, args.end, out, manifest, args.force)
    manifest["runs"][-1].update(finished=now(), requests=len(client.log))
    save_json(manifest, manifest_path)
    print(f"done in {time.time() - t0:.0f}s, {len(client.log)} requests; manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
