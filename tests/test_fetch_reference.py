"""The reference fetch: every file, the manifest's settings and hashes, and resuming without refetching."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.alpaca_rest import AlpacaREST


class Response:
    status_code, headers, text = 200, {}, ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class Refused:
    status_code, headers, text = 403, {}, '{"message":"subscription does not permit querying auctions"}'


class Router:
    """Answers each endpoint with a small valid body and remembers the parameters it was sent."""

    def __init__(self, refuse=()):
        self.calls, self.refuse = [], refuse

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params)
        self.calls.append((url, params))
        if any(url.endswith(r) for r in self.refuse):
            return Refused()
        syms = params.get("symbols", "").split(",")
        if url.endswith("/v2/stocks/bars"):
            days = pd.bdate_range(params["start"][:10], params["end"][:10])
            step = "1Day" == params["timeframe"]
            bars = {s: [{"t": f"{d.date()}T05:00:00Z" if step else f"{d.date()}T14:30:00Z", "o": 10, "h": 11,
                         "l": 9, "c": 10.5, "v": 100, "n": 5, "vw": 10.2} for d in days] for s in syms}
            return Response({"bars": bars, "next_page_token": None})
        if url.endswith("/v2/stocks/auctions"):
            return Response({"auctions": {s: [{"d": "2024-11-29", "o": [{"c": "Q", "p": 10, "t": "2024-11-29T14:30:00Z",
                                                                         "x": "N"}], "c": []}] for s in syms},
                             "next_page_token": None})
        if url.endswith("/v2/stocks/trades"):
            return Response({"trades": {s: [{"t": "2024-11-29T14:30:00.1Z", "p": 10, "s": 100, "x": "N", "c": ["O"],
                                             "z": "A", "i": 1}] for s in syms}, "next_page_token": None})
        if url.endswith("/v1/corporate-actions"):
            body = {"cash_dividends": [{"id": f"d-{s}", "symbol": s, "rate": 0.5, "special": False,
                                        "ex_date": "2024-11-27", "process_date": "2024-11-27"} for s in syms]}
            if "AAA" in syms:
                body["name_changes"] = [{"id": "n1", "old_symbol": "OLDA", "new_symbol": "AAA",
                                         "process_date": "2020-01-02"}]
            return Response({"corporate_actions": body, "next_page_token": None})
        if "/meta/" in url:
            return Response({"@": "Regular Sale"})
        raise AssertionError(url)


def run(tmp_path, monkeypatch, *extra, refuse=()):
    import fetch_reference as script
    router = Router(refuse)
    monkeypatch.setattr(script, "make_client", lambda: AlpacaREST("k", "s", session=router, sleep=lambda s: None))
    pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "name": "x", "sector": "x"}).to_csv(tmp_path / "u.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["fetch_reference.py", "--universe", str(tmp_path / "u.csv"), "--start",
                                      "2024-11-25", "--end", "2024-12-03", "--out", str(tmp_path / "ref"),
                                      "--sample-symbols", "2", "--sample-full-days", "2", "--trade-symbols", "1", *extra])
    assert script.main() == 0
    return router


def test_every_file_and_its_settings_are_recorded(tmp_path, monkeypatch):
    router = run(tmp_path, monkeypatch)
    ref = tmp_path / "ref"
    manifest = json.loads((ref / "manifest.json").read_text())
    for tag in ("raw", "split", "split+dividend", "all", "split_nomap"):
        path = ref / "daily" / f"daily_sip_{tag}.parquet"
        entry = manifest["files"][f"daily/daily_sip_{tag}.parquet"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["symbols_returned"] == 3 and entry["retrieved_at"]["end"]
    assert manifest["files"]["daily/daily_sip_split+dividend.parquet"]["settings"]["adjustment"] == "split,dividend"
    assert manifest["files"]["daily/daily_sip_split_nomap.parquet"]["settings"]["asof"].startswith("-")
    sent = [p for url, p in router.calls if url.endswith("/v2/stocks/bars") and p["timeframe"] == "1Day"]
    assert {p["adjustment"] for p in sent} == {"raw", "split", "split,dividend", "all"}
    assert {p.get("asof") for p in sent} == {None, "-"}
    ca = pd.read_parquet(ref / "corporate_actions.parquet")
    assert "OLDA" in manifest["files"]["corporate_actions.parquet"]["settings"]["second_pass_old_symbols"]
    assert ca["id"].is_unique
    plan = json.loads((ref / "sample" / "plan.json").read_text())
    assert plan["half_days"] == ["2024-11-29"] and len(plan["full_days"]) == 2
    for name in ("ext_5min", "auctions", "trades_open_close"):
        assert (ref / "sample" / f"{name}.parquet").exists()
    assert (ref / "meta" / "conditions_C.json").exists()


def test_a_refused_part_is_recorded_and_the_rest_goes_on(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch, refuse=("/v2/stocks/auctions",))
    ref = tmp_path / "ref"
    manifest = json.loads((ref / "manifest.json").read_text())
    entry = manifest["files"]["sample/auctions.parquet"]
    assert "HTTP 403" in entry["error"] and "sha256" not in entry
    assert not (ref / "sample" / "auctions.parquet").exists()
    assert (ref / "sample" / "trades_open_close.parquet").exists()            # the parts after it still ran
    assert manifest["runs"][-1]["finished"]
    again = run(tmp_path, monkeypatch)                                        # the next run retries only that part
    assert {url.rsplit("/", 1)[-1] for url, _ in again.calls} == {"auctions"}


def test_a_broken_drive_link_stops_with_a_message(tmp_path, monkeypatch, capsys):
    import fetch_reference as script
    (tmp_path / "data").symlink_to(tmp_path / "drive_not_mounted" / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fetch_reference.py", "--out", "data/reference"])
    assert script.main() == 1
    assert "Drive is not mounted" in capsys.readouterr().out


def test_a_second_run_keeps_finished_files_and_force_refetches(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch)
    again = run(tmp_path, monkeypatch)
    assert not again.calls
    forced = run(tmp_path, monkeypatch, "--force", "--parts", "daily")
    assert len(forced.calls) == 5
