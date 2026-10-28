"""The Alpaca REST client: pagination, exact parameters, retries, the rate limit, and no keys in the log."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alpaca_rest import AlpacaREST, daily_frame, rfc3339


class Response:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body or {}, headers or {}, "error text"

    def json(self):
        return self._body


class FakeSession:
    """Replays responses in order and records every call."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers)})
        return self.responses.pop(0)


def client(responses, **kw):
    return AlpacaREST("KEY-ID", "SECRET", session=FakeSession(responses), sleep=lambda s: None, **kw)


def test_bars_follow_pages_and_send_the_exact_settings():
    page1 = {"bars": {"AAA": [{"t": "2024-01-02T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10, "n": 3,
                               "vw": 1.2}]}, "next_page_token": "abc"}
    page2 = {"bars": {"AAA": [{"t": "2024-01-03T05:00:00Z", "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 5, "n": 2,
                               "vw": 1.6}], "BBB": [{"t": "2024-01-02T05:00:00Z", "o": 9, "h": 9, "l": 9, "c": 9,
                                                     "v": 1, "n": 1, "vw": 9}]}, "next_page_token": None}
    c = client([Response(200, page1), Response(200, page2)])
    df = c.bars(["AAA", "BBB"], "1Day", "2024-01-02T05:00:00Z", "2024-01-03T23:59:59Z", "split,dividend", asof="-")
    assert len(df) == 3 and set(df["symbol"]) == {"AAA", "BBB"}
    calls = c.session.calls
    assert calls[0]["params"]["adjustment"] == "split,dividend" and calls[0]["params"]["asof"] == "-"
    assert "page_token" not in calls[0]["params"] and calls[1]["params"]["page_token"] == "abc"
    assert calls[0]["url"].endswith("/v2/stocks/bars")
    daily = daily_frame(df)
    assert daily.loc[daily["symbol"] == "AAA", "date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_default_symbol_mapping_sends_no_asof():
    c = client([Response(200, {"bars": {}, "next_page_token": None})])
    c.bars(["AAA"], "1Day", "a", "b", "split")
    assert "asof" not in c.session.calls[0]["params"]


def test_retries_on_429_and_server_errors_then_fails_loudly_on_others():
    c = client([Response(429, headers={"Retry-After": "1"}), Response(503), Response(200, {"x": 1})])
    assert c.request("/v2/stocks/meta/exchanges", {}) == {"x": 1}
    c = client([Response(403)])
    with pytest.raises(RuntimeError, match="HTTP 403"):
        c.request("/v2/stocks/meta/exchanges", {})


def test_the_log_never_holds_the_keys():
    c = client([Response(200, {"bars": {}, "next_page_token": None})])
    c.bars(["AAA"], "1Day", "a", "b", "raw")
    assert c.session.calls[0]["headers"]["APCA-API-KEY-ID"] == "KEY-ID"
    assert "KEY-ID" not in str(c.log) and "SECRET" not in str(c.log)


def test_rate_limit_waits_instead_of_exceeding_it():
    now, slept = [0.0], []
    c = AlpacaREST("k", "s", session=FakeSession([Response(200, {})] * 4), per_minute=2,
                   sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)), clock=lambda: now[0])
    for _ in range(4):
        c.request("/x", {})
    assert len(slept) == 1 and slept[0] > 59


def test_corporate_actions_and_auctions_flatten():
    ca = {"corporate_actions": {"cash_dividends": [{"id": "1", "symbol": "AAA", "rate": 2.0, "special": True,
                                                    "ex_date": "2024-01-03"}],
                                "name_changes": [{"id": "2", "old_symbol": "OLD", "new_symbol": "AAA"}]},
          "next_page_token": None}
    c = client([Response(200, ca)])
    df = c.corporate_actions(["AAA"], "2015-01-01", "2026-10-28")
    assert set(df["kind"]) == {"cash_dividends", "name_changes"} and c.session.calls[0]["params"]["data_quality"] == "all"
    auctions = {"auctions": {"AAA": [{"d": "2024-01-03", "o": [{"c": "Q", "p": 10.0, "t": "2024-01-03T14:30:00Z",
                                                                "x": "N"}],
                                      "c": [{"c": "M", "p": 11.0, "t": "2024-01-03T21:00:00Z", "x": "N"}]}]},
                "next_page_token": None}
    c = client([Response(200, auctions)])
    a = c.auctions(["AAA"], "a", "b")
    assert sorted(a["side"]) == ["close", "open"] and c.session.calls[0]["params"]["feed"] == "sip"


def test_timestamps_with_and_without_fractional_seconds_parse_together():
    """Alpaca writes a whole-second time without a fraction and others with one; the first section 35 run failed on this."""
    auctions = {"auctions": {"AAA": [{"d": "2024-01-03", "o": [{"c": "Q", "p": 10.0, "t": "2024-01-03T14:30:00.123456Z",
                                                                "x": "N"}],
                                      "c": [{"c": "M", "p": 11.0, "t": "2024-01-03T21:00:00Z", "x": "N"}]}]},
                "next_page_token": None}
    a = client([Response(200, auctions)]).auctions(["AAA"], "a", "b")
    assert a["t"].tolist() == [pd.Timestamp("2024-01-03T14:30:00.123456Z"), pd.Timestamp("2024-01-03T21:00:00Z")]
    trades = {"trades": {"AAA": [{"t": "2024-01-03T14:30:00.5Z", "p": 10, "s": 1, "x": "N", "c": ["O"], "z": "A", "i": 1},
                                 {"t": "2024-01-03T14:30:01Z", "p": 10, "s": 1, "x": "N", "c": [], "z": "A", "i": 2}]},
              "next_page_token": None}
    t = client([Response(200, trades)]).trades(["AAA"], "a", "b")
    assert t["t"].dt.second.tolist() == [0, 1]
    bars = {"bars": {"AAA": [{"t": "2024-01-03T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "n": 1, "vw": 1},
                             {"t": "2024-01-03T14:35:00.25Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "n": 1, "vw": 1}]},
            "next_page_token": None}
    assert len(client([Response(200, bars)]).bars(["AAA"], "5Min", "a", "b", "split")) == 2


def test_rfc3339_reads_naive_times_as_new_york():
    assert rfc3339("2024-01-03 09:30") == "2024-01-03T14:30:00Z"
    assert rfc3339("2024-07-03 09:30") == "2024-07-03T13:30:00Z"
