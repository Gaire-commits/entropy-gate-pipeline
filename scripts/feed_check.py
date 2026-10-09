#!/usr/bin/env python3
"""Does this Alpaca account serve consolidated (SIP) bars, and how do they differ from IEX's?

    python scripts/feed_check.py
    python scripts/feed_check.py --symbols SPY AAPL JPM

Downloads about two weeks of 5-minute bars for a few stocks at five dates, from both feeds, and
compares them (src/feed_compare.py). Nothing is cached or written except outputs/feed_check.md.
IEX has no data before July 2020 (those periods show SIP only); recent SIP data newer than 15 minutes
is not available on the free plan, so the recent period ends yesterday.

Credentials come from the environment or .env and are never printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import fetch_symbol, get_client
from src.feed_compare import compare_feeds, gridded, summarize

SYMBOLS = ["SPY", "AAPL", "JPM", "XOM", "ALK", "HAS"]     # from the most to the least heavily traded
PERIODS = [("2016 (earliest SIP)", "2016-01-04", "2016-01-15"), ("2018", "2018-06-04", "2018-06-15"),
           ("2020 crash", "2020-03-09", "2020-03-20"), ("2021", "2021-06-01", "2021-06-15"), ("recent", None, None)]


def fmt(x, spec=".2f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def download(client, symbol, start, end, feed):
    """(bars or None, error text or None). An API refusal is reported, not raised."""
    try:
        raw = fetch_symbol(client, symbol, start, str(pd.Timestamp(end) + pd.Timedelta(days=1)), "5Min", feed, "split")
        return gridded(raw), None
    except Exception as e:                                                  # noqa: BLE001 - report any API error
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    args = parser.parse_args()
    client = get_client()
    yesterday = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize() - pd.Timedelta(days=1)
    periods = [(n, s or str((yesterday - pd.Timedelta(days=13)).date()), e or str(yesterday.date())) for n, s, e in PERIODS]

    lines = ["# Feed check: IEX against consolidated (SIP) 5-minute bars", "",
             "Same symbols and days from both feeds. **grid filled**: share of 5-minute slots with a bar. "
             "**volume ratio**: SIP volume over IEX volume on bars both have (median). **close difference**: "
             "|IEX close / SIP close − 1| in bps on shared bars. **return correlation**: 5-minute log returns "
             "on shared consecutive bars.", "",
             "| period | days | symbols (IEX / SIP with data) | IEX grid filled | SIP grid filled | volume ratio | "
             "close diff median / 95% (bps) | return corr |", "|---|---|---|---:|---:|---:|---|---:|"]
    detail, errors, summaries = [], [], []
    for name, start, end in periods:
        rows = []
        for symbol in args.symbols:
            iex, e1 = download(client, symbol, start, end, "iex")
            sip, e2 = download(client, symbol, start, end, "sip")
            errors += [f"{name} {symbol} {feed}: {e}" for feed, e in (("iex", e1), ("sip", e2)) if e]
            rows.append({"symbol": symbol, **compare_feeds(iex, sip)})
        s = summarize(rows)
        summaries.append(s)
        lines.append(f"| {name} | {start} to {end} | {s['symbols_iex']} / {s['symbols_sip']} of {s['symbols']} | "
                     f"{fmt(s['iex_completeness'], '.0%')} | {fmt(s['sip_completeness'], '.0%')} | "
                     f"{fmt(s['volume_ratio'], '.0f')}x | {fmt(s['close_diff_bps_median'])} / "
                     f"{fmt(s['close_diff_bps_p95'])} | {fmt(s['return_corr'], '.3f')} |")
        detail += [f"| {name} | {r['symbol']} | {r['iex_sessions']} | {r['sip_sessions']} | {fmt(r['iex_completeness'], '.0%')} | "
                   f"{fmt(r['sip_completeness'], '.0%')} | {fmt(r['volume_ratio'], '.0f')}x |" for r in rows]
    lines += ["", "## Per symbol", "", "| period | symbol | IEX sessions | SIP sessions | IEX grid filled | SIP grid filled | "
              "volume ratio |", "|---|---|---:|---:|---:|---:|---:|"] + detail
    if errors:
        lines += ["", "## Errors (as returned by the API)", ""] + [f"- {e}" for e in sorted(set(errors))]
    has_2016 = summaries[0]["symbols_sip"] > 0
    lines += ["", "**SIP history from 2016 available on this account:** "
              + (f"yes ({summaries[0]['symbols_sip']} of {summaries[0]['symbols']} symbols have 2016 bars)." if has_2016 else
                 "NOT CONFIRMED. See the errors above before fetching.")]
    text = "\n".join(lines) + "\n"
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "feed_check.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
