"""Download Bybit historical order books and rebuild them into top-5 book events.

    python fetch_bybit.py spot BTCUSDT 2026-11-09 2026-11-10
    python fetch_bybit.py linear ETHUSDT 2026-11-09 2026-11-10
    python fetch_bybit.py --selftest

Source: Bybit's public order-book archive (one zip per day: a snapshot, then 100 ms deltas;
size "0" deletes a level). Output: data/markets/bybit_<category>_<symbol>_<date>.npz with
  R    (n, 23) rows in the notebook's BOOK_COLS order: bp1..5, bq1..5, ap1..5, aq1..5, volume, tbq, tsq
  ts   (n,) exchange timestamps in ms
  tick price grid inferred from the snapshot
A row is kept only when the top 5 levels change (one "event", as in FI-2010). volume is 0 (no trades
in this archive); tbq / tsq are the total size resting across all published levels.
"""
import argparse
import io
import json
import sys
import urllib.request
import zipfile
from heapq import nlargest, nsmallest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "bybit"
OUT = ROOT / "data" / "markets"
DEPTH = {"spot": 200, "linear": 200}   # archive depth per category (linear was ob500 before ~mid-2025)
URL = "https://quote-saver.bycsi.com/orderbook/{cat}/{sym}/{date}_{sym}_ob{depth}.data.zip"
L = 5


def rebuild(lines):
    """Iterate JSON lines -> (R rows, ts list, tick). Pure function so it can be self-tested."""
    bids, asks = {}, {}
    sb = sa = 0.0
    rows, ts, prev, tick = [], [], None, None
    for line in lines:
        m = json.loads(line)
        d = m["data"]
        if m["type"] == "snapshot":
            bids = {float(p): float(q) for p, q in d["b"]}
            asks = {float(p): float(q) for p, q in d["a"]}
            sb, sa = sum(bids.values()), sum(asks.values())
            if tick is None:
                px = np.unique(np.round(np.r_[list(bids), list(asks)], 8))
                tick = float(np.round(np.diff(px).min(), 8))
        else:
            for side, book in (("b", bids), ("a", asks)):
                for p, q in d[side]:
                    p, q = float(p), float(q)
                    old = book.pop(p, 0.0)
                    if q > 0:
                        book[p] = q
                    if side == "b":
                        sb += q - old
                    else:
                        sa += q - old
        if len(bids) < L or len(asks) < L:
            continue
        bp = nlargest(L, bids); ap = nsmallest(L, asks)
        if bp[0] >= ap[0]:
            continue                      # crossed while a delta is half-applied: skip
        top = (bp, [bids[p] for p in bp], ap, [asks[p] for p in ap])
        if top == prev:
            continue                      # top 5 unchanged: not an event
        prev = top
        rows.append(bp + top[1] + ap + top[3] + [0.0, sb, sa])
        ts.append(m["ts"])
    return rows, ts, tick


def download(url, dest, tries=8):
    """Resumable download to dest.part, renamed to dest only once complete (the archive drops long transfers)."""
    part = dest.with_name(dest.name + ".part")
    print("downloading", url)
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                total = have + int(r.headers["Content-Length"])
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if part.stat().st_size >= total:
                part.rename(dest)
                return
        except OSError as e:
            print(f"  interrupted at {part.stat().st_size if part.exists() else 0:,} bytes ({e}); resuming")
    sys.exit(f"download failed after {tries} attempts: {url}")


def fetch(cat, sym, date):
    out = OUT / f"bybit_{cat}_{sym}_{date}.npz"
    if out.exists():
        print("exists:", out.name)
        return
    RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    zpath = RAW / f"{date}_{sym}_{cat}.zip"
    if not zpath.exists():
        download(URL.format(cat=cat, sym=sym, date=date, depth=DEPTH[cat]), zpath)
    with zipfile.ZipFile(zpath) as z, z.open(z.infolist()[0]) as f:
        rows, ts, tick = rebuild(io.TextIOWrapper(f, encoding="utf-8"))
    R = np.asarray(rows, dtype=np.float64)
    np.savez_compressed(out, R=R, ts=np.asarray(ts, dtype=np.int64), tick=tick)
    print(f"{out.name}: {len(R):,} events, tick={tick}")


def selftest():
    snap = {"type": "snapshot", "ts": 1, "data": {"b": [[str(100 - i), "1"] for i in range(6)],
                                                   "a": [[str(101 + i), "2"] for i in range(6)]}}
    same = {"type": "delta", "ts": 2, "data": {"b": [["90", "5"]], "a": []}}           # below top 5: no event
    hit = {"type": "delta", "ts": 3, "data": {"b": [["100", "0"]], "a": [["101", "7"]]}}  # best bid removed
    rows, ts, tick = rebuild(json.dumps(x) for x in (snap, same, hit))
    assert tick == 1.0 and ts == [1, 3] and len(rows) == 2
    r0, r1 = rows
    assert r0[:5] == [100, 99, 98, 97, 96] and r0[10:15] == [101, 102, 103, 104, 105]
    assert r0[21] == 6 and r0[22] == 12
    assert r1[0] == 99 and r1[4] == 95 and r1[15] == 7 and r1[21] == 10 and r1[22] == 17
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("category", nargs="?", choices=list(DEPTH))
    ap.add_argument("symbol", nargs="?")
    ap.add_argument("dates", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit()
    if not (a.category and a.symbol and a.dates):
        ap.error("need category, symbol and at least one date")
    for d in a.dates:
        fetch(a.category, a.symbol.upper(), d)
