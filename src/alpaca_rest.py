"""Alpaca's market-data REST API, called directly, for the audit's reference data.

alpaca-py's request models cannot express everything the audit needs: combined adjustments
("split,dividend"), symbol mapping switched off (asof "-"), the auctions endpoint, and three of the
documented corporate-action types. So these calls follow the documented endpoints
(https://docs.alpaca.markets/reference/stockbars, .../corporateactions-1, .../stockauctions-1,
.../stocktrades-1, .../stockmetaconditions-1, .../stockmetaexchanges-1). Credentials come from the
environment as in src/data.py and are sent only as request headers: the request log, which the
reference manifest summarises, holds paths and parameters, never the keys.

Every page is retried on HTTP 429 and 5xx, and calls are spaced to stay under the plan's rate limit.
"""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np
import pandas as pd

BASE_URL = "https://data.alpaca.markets"
EASTERN = "America/New_York"
DOCS = {
    "bars": "https://docs.alpaca.markets/reference/stockbars",
    "auctions": "https://docs.alpaca.markets/us/reference/stockauctions-1.md",
    "trades": "https://docs.alpaca.markets/us/reference/stocktrades-1.md",
    "corporate_actions": "https://docs.alpaca.markets/reference/corporateactions-1",
    "conditions": "https://docs.alpaca.markets/us/reference/stockmetaconditions-1.md",
    "exchanges": "https://docs.alpaca.markets/us/reference/stockmetaexchanges-1.md",
    "bar_rules": "https://docs.alpaca.markets/us/docs/market-data-faq.md#how-are-bars-aggregated",
}


def rfc3339(ts) -> str:
    """A timestamp as UTC RFC-3339; a naive timestamp is read as New York time."""
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize(EASTERN) if ts.tz is None else ts
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


class AlpacaREST:
    def __init__(self, key: str | None = None, secret: str | None = None, session=None, base_url: str = BASE_URL,
                 per_minute: int = 180, retries: int = 6, sleep=time.sleep, clock=time.monotonic):
        if key is None or secret is None:
            from .data import _load_dotenv
            _load_dotenv()
            key = key or os.environ.get("APCA_API_KEY_ID")
            secret = secret or os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Alpaca credentials not found: set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
                               "(Colab section 3 writes them to .env from the notebook secrets).")
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        if session is None:
            import requests
            session = requests.Session()
        self.session, self.base_url = session, base_url.rstrip("/")
        self.per_minute, self.retries = per_minute, retries
        self._sleep, self._clock, self._calls = sleep, clock, deque()
        self.log: list[dict] = []

    def _throttle(self) -> None:
        while True:
            now = self._clock()
            while self._calls and now - self._calls[0] >= 60:
                self._calls.popleft()
            if len(self._calls) < self.per_minute:
                self._calls.append(now)
                return
            self._sleep(60 - (now - self._calls[0]) + 0.05)

    def request(self, path: str, params: dict) -> dict:
        """One GET, retried on 429 and 5xx. Raises on any other error, with the server's message."""
        params = {k: v for k, v in params.items() if v is not None}
        for attempt in range(self.retries):
            self._throttle()
            r = self.session.get(self.base_url + path, params=params, headers=self._headers, timeout=60)
            if r.status_code == 200:
                self.log.append({"path": path, "params": {k: v for k, v in params.items() if k != "page_token"},
                                 "page_token": "page_token" in params})
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                wait = r.headers.get("Retry-After") if hasattr(r, "headers") else None
                self._sleep(float(wait) if wait else min(60.0, 2.0 ** attempt))
                continue
            raise RuntimeError(f"GET {path} -> HTTP {r.status_code}: {str(getattr(r, 'text', ''))[:300]}")
        raise RuntimeError(f"GET {path}: gave up after {self.retries} attempts")

    def pages(self, path: str, params: dict):
        """Every page of a paginated endpoint, following next_page_token."""
        token = None
        while True:
            body = self.request(path, {**params, "page_token": token})
            yield body
            token = body.get("next_page_token")
            if not token:
                return

    def bars(self, symbols: list[str], timeframe: str, start, end, adjustment: str, feed: str = "sip",
             asof: str | None = None, limit: int = 10000) -> pd.DataFrame:
        """Bars for these symbols; `asof=None` leaves Alpaca's default symbol mapping on, "-" turns it off."""
        rows = []
        params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start, "end": end,
                  "adjustment": adjustment, "feed": feed, "asof": asof, "limit": limit, "sort": "asc"}
        for body in self.pages("/v2/stocks/bars", params):
            for sym, items in (body.get("bars") or {}).items():
                rows += [(sym, b.get("t"), b.get("o"), b.get("h"), b.get("l"), b.get("c"), b.get("v"), b.get("n"),
                          b.get("vw")) for b in items]
        df = pd.DataFrame(rows, columns=["symbol", "t", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
        df["t"] = pd.to_datetime(df["t"], utc=True, format="ISO8601")
        return df

    def auctions(self, symbols: list[str], start, end, asof: str | None = None, limit: int = 10000) -> pd.DataFrame:
        """Opening and closing auction prints, one row per print (the docs: only feed sip is valid)."""
        rows = []
        params = {"symbols": ",".join(symbols), "start": start, "end": end, "feed": "sip", "asof": asof,
                  "limit": limit, "sort": "asc"}
        for body in self.pages("/v2/stocks/auctions", params):
            for sym, days in (body.get("auctions") or {}).items():
                for day in days:
                    for side, key in (("open", "o"), ("close", "c")):
                        rows += [(sym, day.get("d"), side, a.get("c"), a.get("p"), a.get("s"), a.get("t"), a.get("x"))
                                 for a in day.get(key) or []]
        df = pd.DataFrame(rows, columns=["symbol", "date", "side", "condition", "price", "size", "t", "exchange"])
        df["t"] = pd.to_datetime(df["t"], utc=True, format="ISO8601")
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
        return df

    def trades(self, symbols: list[str], start, end, feed: str = "sip", asof: str | None = None,
               limit: int = 10000) -> pd.DataFrame:
        rows = []
        params = {"symbols": ",".join(symbols), "start": start, "end": end, "feed": feed, "asof": asof,
                  "limit": limit, "sort": "asc"}
        for body in self.pages("/v2/stocks/trades", params):
            for sym, items in (body.get("trades") or {}).items():
                rows += [(sym, t.get("t"), t.get("p"), t.get("s"), t.get("x"), " ".join(t.get("c") or []), t.get("z"),
                          t.get("i"), t.get("u")) for t in items]
        df = pd.DataFrame(rows, columns=["symbol", "t", "price", "size", "exchange", "conditions", "tape", "id", "update"])
        df["t"] = pd.to_datetime(df["t"], utc=True, format="ISO8601")
        return df

    def corporate_actions(self, symbols: list[str], start, end, types: list[str] | None = None,
                          data_quality: str | None = "all", limit: int = 1000) -> pd.DataFrame:
        """Every corporate action returned, one row each; `kind` is the response's group (e.g. cash_dividends)."""
        rows = []
        params = {"symbols": ",".join(symbols), "start": start, "end": end,
                  "types": ",".join(types) if types else None, "data_quality": data_quality, "limit": limit,
                  "sort": "asc"}
        for body in self.pages("/v1/corporate-actions", params):
            for kind, items in (body.get("corporate_actions") or {}).items():
                rows += [{"kind": kind, **item} for item in items or []]
        return pd.DataFrame(rows)

    def conditions(self, tape: str, ticktype: str = "trade") -> dict:
        return self.request(f"/v2/stocks/meta/conditions/{ticktype}", {"tape": tape})

    def exchanges(self) -> dict:
        return self.request("/v2/stocks/meta/exchanges", {})


def daily_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily bars with their New York session date (the bar's timestamp is the start of that day)."""
    out = bars.copy()
    out["date"] = out["t"].dt.tz_convert(EASTERN).dt.tz_localize(None).dt.normalize()
    for col in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float64)
    return out.drop(columns="t")
