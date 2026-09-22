"""Record Interactive Brokers quotes, trades and order-book depth to CSV.

Market data only: nothing in here places or modifies an order.

This is the recorder from the first capture notebook with four fixes, each from a
problem found in that capture:

- Depth is requested as SMART depth. Asking for SMART with `isSmartDepth=False`
  is what produced IB error 10092 and an empty depth file.
- The depth file records the `operation` (insert / update / delete). Without it a
  deleted level looks the same as an updated one and the book cannot be rebuilt.
- The live book is a ranked list, not a dict, because IB positions shift on every
  insert and delete.
- Rows are flushed to disk every few seconds by a background thread, so a crashed
  kernel loses seconds rather than a session, and timestamps are UTC with
  milliseconds (the first version mixed local time with exchange time).

Run it between 09:30 and 16:00 ET on a weekday. Outside those hours spreads are
several times wider and depth is usually empty.
"""

from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

from .ibkr import apply_depth_op

EASTERN = "America/New_York"
TICK_COLUMNS = ["recv_ts", "exch_ts", "symbol", "tick_type", "price", "size", "exchange"]
DEPTH_COLUMNS = ["recv_ts", "symbol", "side", "position", "operation", "price", "size", "market_maker"]

QUIET_CODES = {2104, 2106, 2107, 2108, 2158}
HINTS = {
    10092: "This depth request is not valid as sent. Use SMART depth (isSmartDepth=True), or give the depth "
           "request a real exchange such as ISLAND for Nasdaq stocks or ARCA for NYSE Arca ETFs.",
    354: "No market data subscription for this instrument.",
    10089: "This data needs an extra subscription that must also be enabled for API use.",
    309: "IBKR allows only a few depth streams at once (3 by default). Cancel one or request fewer symbols.",
    10190: "IBKR's limit on tick-by-tick streams was reached. Request fewer symbols.",
}


def utc_iso(epoch: float | None = None) -> str:
    stamp = datetime.fromtimestamp(time.time() if epoch is None else epoch, tz=timezone.utc)
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def market_open_now(now: datetime | None = None) -> bool:
    """Weekday 09:30-16:00 Eastern. Holidays are not checked."""
    now = (now or datetime.now(ZoneInfo(EASTERN))).astimezone(ZoneInfo(EASTERN))
    return now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 0)


def stock_contract(symbol: str, exchange: str = "SMART") -> Contract:
    contract = Contract()
    contract.symbol, contract.secType, contract.exchange, contract.currency = symbol, "STK", exchange, "USD"
    return contract


class CaptureApp(EWrapper, EClient):
    def __init__(self, output_dir: str | Path = "data/ibkr", max_rows: int = 10):
        EClient.__init__(self, self)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self.connected = False
        self.depth_req_map: dict[int, str] = {}
        self.tick_req_map: dict[int, str] = {}
        self.books = defaultdict(lambda: {"bid": [], "ask": []})
        self.counts = defaultdict(int)
        self.last_quote: dict[str, tuple[float, float, float]] = {}
        self.errors: list[tuple[int, int, str]] = []
        self._depth_buffer: list[list] = []
        self._tick_buffer: list[list] = []
        self._lock = threading.Lock()
        self._next_req = 1
        self._stop_flush = threading.Event()
        self._flusher: threading.Thread | None = None

    # ---- connection -------------------------------------------------------------------

    def nextValidId(self, orderId):
        self.connected = True

    def connectionClosed(self):
        self.connected = False
        print("Connection to TWS closed.")

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in QUIET_CODES:
            return
        self.errors.append((reqId, errorCode, errorString))
        symbol = self.depth_req_map.get(reqId) or self.tick_req_map.get(reqId) or ""
        print(f"[error] reqId={reqId} {symbol} code={errorCode} {errorString}")
        if errorCode in HINTS:
            print(f"        -> {HINTS[errorCode]}")

    # ---- market data callbacks ---------------------------------------------------------

    def tickByTickAllLast(self, reqId, tickType, time_, price, size, tickAttribLast, exchange, specialConditions):
        symbol = self.tick_req_map.get(reqId, f"reqId_{reqId}")
        with self._lock:
            self._tick_buffer.append([utc_iso(), utc_iso(time_), symbol, "trade", price, float(size), exchange])
            self.counts[(symbol, "trade")] += 1

    def tickByTickBidAsk(self, reqId, time_, bidPrice, askPrice, bidSize, askSize, tickAttribBidAsk):
        symbol = self.tick_req_map.get(reqId, f"reqId_{reqId}")
        recv, exch = utc_iso(), utc_iso(time_)
        with self._lock:
            self._tick_buffer.append([recv, exch, symbol, "bid", bidPrice, float(bidSize), ""])
            self._tick_buffer.append([recv, exch, symbol, "ask", askPrice, float(askSize), ""])
            self.counts[(symbol, "quote")] += 1
            self.last_quote[symbol] = (bidPrice, askPrice, time.time())

    def _depth(self, reqId, position, operation, side, price, size, maker):
        symbol = self.depth_req_map.get(reqId, f"reqId_{reqId}")
        side_name = "bid" if side == 1 else "ask"
        with self._lock:
            apply_depth_op(self.books[symbol][side_name], position, operation, (price, float(size), maker), self.max_rows)
            self._depth_buffer.append([utc_iso(), symbol, side_name, position, operation, price, float(size), maker])
            self.counts[(symbol, "depth")] += 1

    def updateMktDepthL2(self, reqId, position, marketMaker, operation, side, price, size, isSmartDepth=False):
        self._depth(reqId, position, operation, side, price, size, marketMaker)

    def updateMktDepth(self, reqId, position, operation, side, price, size):
        self._depth(reqId, position, operation, side, price, size, "")

    # ---- subscriptions -----------------------------------------------------------------

    def _req_id(self) -> int:
        req_id, self._next_req = self._next_req, self._next_req + 1
        return req_id

    def subscribe(self, symbol: str, depth: bool = True, depth_exchange: str | None = None) -> None:
        """Start trades and best bid/ask for `symbol`, and optionally depth.

        Depth is SMART depth unless `depth_exchange` names a real exchange, which
        then goes out as a direct-exchange request.
        """
        if not market_open_now():
            print(f"note: US markets are closed right now. Quotes will be thin and wide and depth may be "
                  f"empty; the useful capture is 09:30-16:00 ET on a weekday.")
        contract = stock_contract(symbol)
        if depth:
            req = self._req_id()
            self.depth_req_map[req] = symbol
            if depth_exchange:
                self.reqMktDepth(req, stock_contract(symbol, depth_exchange), self.max_rows, False, [])
            else:
                self.reqMktDepth(req, contract, self.max_rows, True, [])
        for kind in ("AllLast", "BidAsk"):   # AllLast also returns off-exchange prints, so volume is the whole tape
            req = self._req_id()
            self.tick_req_map[req] = symbol
            self.reqTickByTickData(req, contract, kind, 0, False)

    def unsubscribe_all(self) -> None:
        for req in list(self.depth_req_map):
            self.cancelMktDepth(req, True)
        for req in list(self.tick_req_map):
            self.cancelTickByTickData(req)
        self.depth_req_map.clear()
        self.tick_req_map.clear()

    # ---- files -------------------------------------------------------------------------

    def _path(self, kind: str) -> Path:
        day = datetime.now(ZoneInfo(EASTERN)).strftime("%Y%m%d")
        return self.output_dir / f"{kind}_{day}.csv"

    def _append(self, path: Path, header: list[str], rows: list[list]) -> None:
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(header)
            writer.writerows(rows)

    def flush_to_csv(self) -> tuple[int, int]:
        """Write buffered rows to today's files. Returns (depth rows, tick rows) written."""
        with self._lock:
            depth, ticks = self._depth_buffer, self._tick_buffer
            self._depth_buffer, self._tick_buffer = [], []
        if depth:
            self._append(self._path("depth"), DEPTH_COLUMNS, depth)
        if ticks:
            self._append(self._path("ticks"), TICK_COLUMNS, ticks)
        return len(depth), len(ticks)

    def start_autoflush(self, interval: float = 5.0) -> None:
        if self._flusher and self._flusher.is_alive():
            return
        self._stop_flush.clear()

        def loop():
            while not self._stop_flush.wait(interval):
                self.flush_to_csv()

        self._flusher = threading.Thread(target=loop, daemon=True)
        self._flusher.start()

    def stop_autoflush(self) -> None:
        self._stop_flush.set()
        if self._flusher:
            self._flusher.join(timeout=5)
        self.flush_to_csv()

    # ---- inspection --------------------------------------------------------------------

    def get_current_book(self, symbol: str) -> dict[str, list]:
        with self._lock:
            return {side: list(levels) for side, levels in self.books[symbol].items()}

    def status(self) -> list[dict]:
        """Rows received per symbol since start, plus the latest quote and how old it is."""
        with self._lock:
            symbols = sorted({s for s, _ in self.counts} | set(self.last_quote))
            rows = []
            for symbol in symbols:
                bid, ask, seen = self.last_quote.get(symbol, (None, None, None))
                rows.append({
                    "symbol": symbol,
                    "quotes": self.counts[(symbol, "quote")],
                    "trades": self.counts[(symbol, "trade")],
                    "depth_msgs": self.counts[(symbol, "depth")],
                    "bid": bid, "ask": ask,
                    "quote_age_s": None if seen is None else round(time.time() - seen, 1),
                })
        return rows
