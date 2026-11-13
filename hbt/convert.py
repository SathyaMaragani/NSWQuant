"""Bybit historical order book (ob200 snapshot + deltas) and trades -> hftbacktest event data, for one time window.

    hbt/.venv/Scripts/python.exe hbt/convert.py 2026-11-12 14:00 15:00      # -> hbt/data/ETHUSDT_2026-11-12_1400_1500.npz
    hbt/.venv/Scripts/python.exe hbt/convert.py --selftest

Based on hftbacktest.data.utils.bybithistmktdata.convert (hftbacktest 2.4.4), with two changes:
  * snapshot rows carry their side flag (BUY_EVENT / SELL_EVENT); the bundled converter writes DEPTH_SNAPSHOT_EVENT
    without one, so the engine cannot tell bids from asks in a snapshot;
  * only one window is emitted: the book is rebuilt silently up to the window start, written there as a clear +
    full snapshot, then every delta and trade inside the window follows (a whole day does not fit the bundled
    converter's 100 M-row buffer on this machine).
Timestamps are nanoseconds; Bybit's archive has no receive time, so local time = exchange time + FEED_LATENCY_NS.
"""
import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from hftbacktest import BUY_EVENT, DEPTH_CLEAR_EVENT, DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT, SELL_EVENT, TRADE_EVENT
from hftbacktest.data import correct_event_order, validate_event_order
from hftbacktest.data.validation import correct_local_timestamp
from hftbacktest.types import event_dtype

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hbt" / "data"
FEED_LATENCY_NS = 1_000_000                       # 1 ms from Bybit's matching engine to us (an assumption)
SIDE = {"b": BUY_EVENT, "a": SELL_EVENT}


def snapshot_rows(ts, books, lat):
    """Clear each side down to its deepest level, then the full book, every row flagged with its side."""
    rows = []
    for k, book in books.items():
        if not book:
            continue
        deepest = min(book) if k == "b" else max(book)
        rows.append((DEPTH_CLEAR_EVENT | SIDE[k], ts, ts + lat, deepest, 0.0, 0, 0, 0.0))
        rows += [(DEPTH_SNAPSHOT_EVENT | SIDE[k], ts, ts + lat, p, q, 0, 0, 0.0) for p, q in book.items()]
    return rows


def depth_rows(lines, t0_ns, t1_ns, lat=FEED_LATENCY_NS):
    """JSON lines of the ob200 archive -> event rows inside [t0, t1)."""
    books, rows, started = {"b": {}, "a": {}}, [], False
    for line in lines:
        m = json.loads(line)
        ts = int(float(m["ts"]) * 1_000_000)
        if ts >= t1_ns:
            break
        d, snap = m["data"], m["type"] == "snapshot"
        for k in ("b", "a"):
            if snap:
                books[k] = {}
            for p, q in d.get(k, []):
                p, q = float(p), float(q)
                if q == 0:
                    books[k].pop(p, None)
                else:
                    books[k][p] = q
        if ts < t0_ns:
            continue
        if not started or snap:                   # first message in the window, or a fresh snapshot: full book
            rows += snapshot_rows(ts, books, lat)
            started = True
            continue
        for k in ("b", "a"):
            rows += [(DEPTH_EVENT | SIDE[k], ts, ts + lat, float(p), float(q), 0, 0, 0.0) for p, q in d.get(k, [])]
    return rows


def trade_rows(reader, t0_ns, t1_ns, lat=FEED_LATENCY_NS):
    """Bybit trades CSV rows (header: timestamp, symbol, side, size, price, ...) -> event rows inside [t0, t1)."""
    header = next(reader)
    it, iside, ipx, iq = (header.index(c) for c in ("timestamp", "side", "price", "size"))
    rows = []
    for r in reader:
        ts = int(round(float(r[it]) * 1_000_000_000))
        if t0_ns <= ts < t1_ns:                   # side = the aggressor (trade initiator), as in hftbacktest
            rows.append((TRADE_EVENT | (SELL_EVENT if r[iside] == "Sell" else BUY_EVENT), ts, ts + lat, float(r[ipx]), float(r[iq]), 0, 0, 0.0))
    return rows


def finish(rows):
    tmp = np.array(rows, dtype=event_dtype)
    tmp = correct_local_timestamp(tmp, 0)
    data = correct_event_order(tmp, np.argsort(tmp["exch_ts"], kind="mergesort"), np.argsort(tmp["local_ts"], kind="mergesort"))
    validate_event_order(data)
    return data


def convert(day, start, end, symbol="ETHUSDT"):
    t0 = int(datetime.fromisoformat(f"{day}T{start}:00").replace(tzinfo=timezone.utc).timestamp() * 1e9)
    t1 = int(datetime.fromisoformat(f"{day}T{end}:00").replace(tzinfo=timezone.utc).timestamp() * 1e9)
    zf = ROOT / "raw" / "bybit" / f"{day}_{symbol}_linear.zip"
    tf = ROOT / "raw" / "trades" / f"{symbol}_linear_{day}.csv.gz"
    print("depth", zf.name)
    with ZipFile(zf) as z, z.open(z.namelist()[0]) as f:
        rows = depth_rows((ln.decode() for ln in f), t0, t1)
    print("trades", tf.name)
    with gzip.open(tf, "rt") as f:
        rows += trade_rows(csv.reader(f), t0, t1)
    data = finish(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{symbol}_{day}_{start.replace(':', '')}_{end.replace(':', '')}.npz"
    np.savez_compressed(out, data=data)
    print(f"{len(data):,} events -> {out}")
    return out


def selftest():
    snap = json.dumps({"type": "snapshot", "ts": 1000, "data": {"b": [["100", "2"], ["99", "1"]], "a": [["101", "3"]]}})
    pre = json.dumps({"type": "delta", "ts": 1500, "data": {"b": [["99", "0"]], "a": [["102", "4"]]}})
    dlt = json.dumps({"type": "delta", "ts": 2500, "data": {"b": [["100", "5"]], "a": []}})
    after = json.dumps({"type": "delta", "ts": 9000, "data": {"b": [["98", "1"]], "a": []}})
    rows = depth_rows([snap, pre, dlt, after], t0_ns=2_000_000_000, t1_ns=3_000_000_000)   # archive ts in ms
    ev = [(r[0], r[3], r[4]) for r in rows]
    # window starts at the 2.5 s delta: the book as of then (99 removed, 102 added, 100 -> 5) is written as one snapshot
    assert ev == [(DEPTH_CLEAR_EVENT | BUY_EVENT, 100.0, 0.0), (DEPTH_SNAPSHOT_EVENT | BUY_EVENT, 100.0, 5.0),
                  (DEPTH_CLEAR_EVENT | SELL_EVENT, 102.0, 0.0), (DEPTH_SNAPSHOT_EVENT | SELL_EVENT, 101.0, 3.0),
                  (DEPTH_SNAPSHOT_EVENT | SELL_EVENT, 102.0, 4.0)], ev
    tr = trade_rows(iter([["timestamp", "symbol", "side", "size", "price"], ["0.0025", "ETHUSDT", "Sell", "1.5", "100"],
                          ["0.0040", "ETHUSDT", "Buy", "1", "101"]]), 2_000_000, 3_000_000)
    assert [(r[0], r[1], r[3], r[4]) for r in tr] == [(TRADE_EVENT | SELL_EVENT, 2_500_000, 100.0, 1.5)], tr
    data = finish(rows + tr)
    assert len(data) > 0
    print("selftest ok: window snapshot with side flags, deltas, trades, hftbacktest event order")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        convert(*sys.argv[1:4])
