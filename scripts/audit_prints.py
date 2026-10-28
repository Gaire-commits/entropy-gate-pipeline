#!/usr/bin/env python3
"""The auction-print and trade checks of the extended audit, on their own (about a minute).

    python scripts/audit_prints.py --audit outputs/data_audit_v2 --reference data/reference

Does the vendor's daily open (close) equal the opening (closing) auction print, and the open of the first trade
after 09:30? The same question for the open and close rebuilt from the 5-minute cache. Reads the audit's
sessions.parquet (so the 30-minute cache read is not repeated) and the reference sample; writes
auction_summary.csv, auction_prints.csv, trades_open_close_check.csv and prints.md next to them.
Prints arrive unadjusted, so the vendor is compared on its raw bars; the cache is split-adjusted and rounded to the cent,
so it is compared only on days nothing adjusts it (audit.split_factor). Read-only: nothing is fitted or excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import reachable

from src import audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="outputs/data_audit_v2")
    ap.add_argument("--reference", default="data/reference")
    args = ap.parse_args()
    out, ref = Path(args.audit), Path(args.reference)
    if not reachable(out, ref):
        return 1
    missing = [p for p in (out / "sessions.parquet", ref / "daily" / "daily_sip_raw.parquet",
                           ref / "daily" / "daily_sip_split.parquet") if not p.exists()]
    if missing:
        print("STOPPED: missing", ", ".join(map(str, missing)), "- run sections 35 and 36 first.")
        return 1
    sip = pd.read_parquet(out / "sessions.parquet")
    traded = audit.real(pd.read_parquet(ref / "daily" / "daily_sip_split.parquet"))
    unadjusted = audit.real(pd.read_parquet(ref / "daily" / "daily_sip_raw.parquet"))
    factor = audit.split_factor(unadjusted, traded)
    lines = ["# Auction prints and trades at the open and close (read-only)", "",
             "Prints are unadjusted, so the vendor is compared on its raw bars. The cache (split-adjusted, rounded to the "
             "cent) is compared only on the days nothing adjusts it. Equal means within 0.5 bp.", ""]
    for name, path in (("auctions", ref / "sample" / "auctions.parquet"),
                       ("trades", ref / "sample" / "trades_open_close.parquet")):
        if not path.exists():
            lines += [f"`{path.name}` is missing: run section 35 again (it retries what failed).", ""]
            continue
        frame = pd.read_parquet(path)
        if name == "auctions":
            res = audit.auction_check(frame, unadjusted, sip, factor=factor)
            res["summary"].to_csv(out / "auction_summary.csv", index=False)
            res["by_print"].to_csv(out / "auction_prints.csv", index=False)
            lines += [f"**Auction prints** ({len(frame):,} prints). Share of symbol-days where the vendor's (rebuilt) "
                      "open or close equals some auction print, and which prints:", "",
                      audit.md(res["summary"], ".4f"), "", audit.md(res["by_print"], ".4f", 30), ""]
        else:
            res = audit.trades_check(frame, unadjusted, sip, factor=factor)
            res.to_csv(out / "trades_open_close_check.csv", index=False)
            if len(res):
                lines += [f"**Trades at the open and close** ({len(frame):,} trades). For each candidate print, how "
                          "often the vendor's and the rebuilt value equal it:", "",
                          audit.md(res.groupby(["window", "candidate"])[["vendor_equal", "rebuilt_equal"]].mean()
                                   .reset_index(), ".3f"), ""]
    (out / "prints.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
