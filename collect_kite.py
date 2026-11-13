"""Record the NIFTY near-month futures 5-level order book from Zerodha Kite Connect.

READ-ONLY: uses login, the instrument list and the market-data websocket. It never
calls any order / position / funds endpoint (--selftest enforces this).

One-time setup, in your own PowerShell (values from your Kite Connect app):
    [Environment]::SetEnvironmentVariable("KITE_API_KEY", "...", "User")
    [Environment]::SetEnvironmentVariable("KITE_API_SECRET", "...", "User")
Then:
    python collect_kite.py --login          # once per day; paste the redirected URL
    python collect_kite.py --resolve-only   # show the contract being recorded
    python collect_kite.py                  # record until the NSE futures close (Ctrl+C to stop)
    python collect_kite.py --selftest       # offline check

The redirect page (e.g. http://127.0.0.1/) does not need to load: copy the URL from the
address bar after login (it contains request_token=...) and paste it at the prompt.
Websocket ticks need the paid "Connect" app type (a free "Personal" app has no market data).
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

from nse_session import IST, session_hours

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TOKEN_FILE = ROOT / ".kite_token"
LEVELS = 5
STALL_S = 30   # seconds without a tick before the watchdog forces a reconnect (NIFTY futures tick ~every second)
COLS = (["recv_ns", "exch_ts", "last_trade_time", "ltp", "ltq", "volume", "tbq", "tsq", "oi"]
        + [f"{side}{f}{i}" for side, f in [("b", "p"), ("b", "q"), ("b", "o"), ("a", "p"), ("a", "q"), ("a", "o")]
           for i in range(1, LEVELS + 1)])


def env(name):
    """Process env, else the Windows user env (shells started before it was saved don't inherit it)."""
    v = os.environ.get(name)
    if v or sys.platform != "win32":
        return v
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except OSError:
        return None


def today():
    return dt.datetime.now(IST).date().isoformat()


def _ts(x):
    return x.isoformat() if x else ""


def flatten(tick, recv_ns):
    """One KiteTicker MODE_FULL tick -> one CSV row in COLS order."""
    depth = tick.get("depth") or {}
    buy, sell = depth.get("buy", []), depth.get("sell", [])

    def lvl(side, key):
        return [side[i][key] if i < len(side) else 0 for i in range(LEVELS)]

    return ([recv_ns, _ts(tick.get("exchange_timestamp")), _ts(tick.get("last_trade_time")),
             tick.get("last_price"), tick.get("last_traded_quantity"), tick.get("volume_traded"),
             tick.get("total_buy_quantity"), tick.get("total_sell_quantity"), tick.get("oi")]
            + lvl(buy, "price") + lvl(buy, "quantity") + lvl(buy, "orders")
            + lvl(sell, "price") + lvl(sell, "quantity") + lvl(sell, "orders"))


def login(kite, secret):
    print("Open this URL, log in, then copy the URL you land on:\n ", kite.login_url())
    s = input("Paste redirected URL (or the request_token): ").strip()
    m = re.search(r"request_token=([^&\s]+)", s)
    sess = kite.generate_session(m.group(1) if m else s, api_secret=secret)
    TOKEN_FILE.write_text(json.dumps({"date": today(), "access_token": sess["access_token"]}))
    print(f"Logged in. Access token saved to {TOKEN_FILE.name} (valid until ~6 AM tomorrow).")


def load_token():
    try:
        t = json.loads(TOKEN_FILE.read_text())
        return t["access_token"] if t.get("date") == today() else None
    except (OSError, ValueError, KeyError):
        return None


def resolve(kite):
    """Nearest-expiry NIFTY futures contract on NFO."""
    d = dt.datetime.now(IST).date()
    futs = [i for i in kite.instruments("NFO")
            if i["name"] == "NIFTY" and i["instrument_type"] == "FUT" and i["expiry"] >= d]
    if not futs:
        sys.exit("No NIFTY futures found in the NFO instrument list.")
    inst = min(futs, key=lambda i: i["expiry"])
    meta = {k: inst[k] for k in ("instrument_token", "tradingsymbol", "tick_size", "lot_size")}
    meta["expiry"] = str(inst["expiry"])
    DATA.mkdir(exist_ok=True)
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def record(api_key, access_token, meta):
    from kiteconnect import KiteTicker

    now = dt.datetime.now(IST)
    hours = session_hours(now.date(), meta.get("expiry"))
    if hours is None or now.time() >= hours[1]:
        sys.exit("No remaining NSE futures session today.")
    close = dt.datetime.combine(now.date(), hours[1], IST)
    path = DATA / f"nifty_fut_{now:%Y%m%d}.csv"
    new = not path.exists()
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(COLS)
    tok, n, last = meta["instrument_token"], 0, time.time()
    kws = KiteTicker(api_key, access_token)

    def on_connect(ws, _resp):
        ws.subscribe([tok])
        ws.set_mode(ws.MODE_FULL, [tok])
        print("connected; streaming", meta["tradingsymbol"])

    def on_ticks(ws, ticks):
        nonlocal n, last
        last = time.time()
        now = time.time_ns()
        for t in ticks:
            if t.get("instrument_token") == tok:
                w.writerow(flatten(t, now))
                n += 1
        f.flush()  # a few ticks/sec: flushing every callback is cheap and loses nothing on crash
        if n and n % 1000 == 0:
            print(n, "rows")
        if dt.datetime.now(IST) >= close:
            ws.stop()

    def on_close(_ws, code, reason):
        print("closed:", code, reason)

    def on_error(_ws, code, reason):
        print("error:", code, reason)

    def on_reconnect(_ws, attempt):
        print("reconnecting, attempt", attempt)

    # Watchdog: Kite's own check only watches pings, and pings can keep flowing while ticks stop.
    # No tick for STALL_S -> drop the socket; KiteTicker reconnects and on_connect resubscribes.
    def watchdog():
        nonlocal last
        if dt.datetime.now(IST) >= close:
            kws.stop()                     # also ends a quiet feed after the close
        elif time.time() - last > STALL_S and kws.ws is not None:
            print(f"{dt.datetime.now(IST):%H:%M:%S} no ticks for {STALL_S}s: forcing reconnect")
            last = time.time()
            kws.ws.dropConnection(abort=True)

    from twisted.internet import task
    task.LoopingCall(watchdog).start(5, now=False)

    kws.on_connect, kws.on_ticks, kws.on_close, kws.on_error = on_connect, on_ticks, on_close, on_error
    kws.on_reconnect = on_reconnect
    print(f"Recording -> {path}")
    try:
        kws.connect()  # blocks; KiteTicker reconnects on its own
    finally:
        f.close()
        print(f"saved {n} rows to {path}")


def selftest():
    depth = {"buy": [{"price": 24000.0 - i * 0.1, "quantity": 75 * (i + 1), "orders": i + 1} for i in range(5)],
             "sell": [{"price": 24000.5 + i * 0.1, "quantity": 150 * (i + 1), "orders": i + 2} for i in range(5)]}
    tick = {"instrument_token": 1, "last_price": 24000.5, "last_traded_quantity": 75, "volume_traded": 1000,
            "total_buy_quantity": 5000, "total_sell_quantity": 6000, "oi": 123,
            "exchange_timestamp": dt.datetime(2026, 11, 13, 10, 0, 0), "last_trade_time": None, "depth": depth}
    row = flatten(tick, 7)
    assert len(row) == len(COLS) == 9 + 6 * LEVELS
    r = dict(zip(COLS, row))
    assert r["recv_ns"] == 7 and r["exch_ts"] == "2026-11-13T10:00:00" and r["last_trade_time"] == ""
    assert r["bp1"] == 24000.0 and r["ap1"] == 24000.5 and r["bq3"] == 225 and r["aq2"] == 300
    assert r["bo5"] == 5 and r["ao1"] == 2 and r["tbq"] == 5000 and r["volume"] == 1000
    empty = dict(zip(COLS, flatten({"depth": {"buy": [], "sell": []}}, 0)))
    assert empty["bp1"] == 0 and empty["aq5"] == 0
    # read-only guarantee: no trading call may appear in this file
    banned = "|".join(a + "_" + b for a, b in [("place", "order"), ("modify", "order"),
                                               ("cancel", "order"), ("place", "gtt"), ("exit", "order")])
    assert not re.search(banned, Path(__file__).read_text()), "trading call found in collector"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--login", action="store_true", help="interactive daily login")
    g.add_argument("--resolve-only", action="store_true", help="print the contract and exit")
    g.add_argument("--selftest", action="store_true", help="offline check")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    from kiteconnect import KiteConnect
    key = env("KITE_API_KEY")
    if not key:
        sys.exit("Set KITE_API_KEY (and KITE_API_SECRET for --login) first (see: python collect_kite.py --help).")
    kite = KiteConnect(api_key=key)
    if a.login:
        secret = env("KITE_API_SECRET") or sys.exit("Set KITE_API_SECRET first.")
        return login(kite, secret)
    token = load_token() or sys.exit("No access token for today. Run: python collect_kite.py --login")
    kite.set_access_token(token)
    meta = resolve(kite)
    print(json.dumps(meta, indent=2))
    if not a.resolve_only:
        record(key, token, meta)


if __name__ == "__main__":
    main()
