"""Record the NIFTY near-month futures 5-level order book from the Upstox V3 market feed.

READ-ONLY: uses login, the public instrument list and the market-data websocket. It never
calls any order endpoint (--selftest enforces this).

One-time setup, in your own PowerShell (values from your Upstox developer app):
    [Environment]::SetEnvironmentVariable("UPSTOX_API_KEY", "...", "User")
    [Environment]::SetEnvironmentVariable("UPSTOX_API_SECRET", "...", "User")
    [Environment]::SetEnvironmentVariable("UPSTOX_REDIRECT_URI", "...", "User")   # exactly as in the app
Then:
    python collect_upstox.py --login          # once per day; paste the redirected URL
    python collect_upstox.py --resolve-only   # show the contract being recorded
    python collect_upstox.py                  # record until the NSE futures close (Ctrl+C to stop)
    python collect_upstox.py --selftest       # offline check

The redirect page does not need to load: copy the URL from the address bar after login
(it contains code=...) and paste it at the prompt.
Needs: pip install upstox-python-sdk websocket-client  (the SDK is used only to decode protobuf;
the websocket is opened here with TLS certificate verification on, which the SDK's own client disables).
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from nse_session import IST, session_hours

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TOKEN_FILE = ROOT / ".upstox_token"
LEVELS = 5
API = "https://api.upstox.com"
FEED_URL = "wss://api.upstox.com/v3/feed/market-data-feed"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
COLS = (["recv_ns", "exch_ts", "last_trade_time", "ltp", "ltq", "volume", "tbq", "tsq", "oi"]
        + [f"{c}{i}" for c in ("bp", "bq", "ap", "aq") for i in range(1, LEVELS + 1)])


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


def need(name):
    return env(name) or sys.exit(f"Set {name} first (see: python collect_upstox.py --help).")


def today():
    return dt.datetime.now(IST).date().isoformat()


def decode(buf):
    """Binary websocket frame -> dict (int64 fields come back as strings, zero fields are omitted)."""
    from google.protobuf import json_format
    from upstox_client.feeder.proto import MarketDataFeedV3_pb2 as pb
    return json_format.MessageToDict(pb.FeedResponse.FromString(buf))


def flatten(ff, recv_ns, feed_ts):
    """One marketFF dict (full mode) -> one CSV row in COLS order."""
    lt = ff.get("ltpc", {})
    q = ff.get("marketLevel", {}).get("bidAskQuote", [])[:LEVELS]
    q = q + [{}] * (LEVELS - len(q))
    return ([recv_ns, int(feed_ts or 0), int(lt.get("ltt", 0)), lt.get("ltp", 0.0), int(lt.get("ltq", 0)),
             int(ff.get("vtt", 0)), ff.get("tbq", 0.0), ff.get("tsq", 0.0), ff.get("oi", 0.0)]
            + [x.get("bidP", 0.0) for x in q] + [int(x.get("bidQ", 0)) for x in q]
            + [x.get("askP", 0.0) for x in q] + [int(x.get("askQ", 0)) for x in q])


def http_json(url, data=None, token=None):
    req = urllib.request.Request(url, data=data, headers={"Accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {url.split('?')[0]}: {e.read().decode(errors='replace')}")


def login():
    key, secret, redirect = need("UPSTOX_API_KEY"), need("UPSTOX_API_SECRET"), need("UPSTOX_REDIRECT_URI")
    url = f"{API}/v2/login/authorization/dialog?" + urllib.parse.urlencode(
        {"response_type": "code", "client_id": key, "redirect_uri": redirect})
    print("Open this URL, log in, then copy the URL you land on:\n ", url)
    s = input("Paste redirected URL (or the code): ").strip()
    m = re.search(r"[?&]code=([^&\s]+)", s)
    body = urllib.parse.urlencode({"code": m.group(1) if m else s, "client_id": key, "client_secret": secret,
                                   "redirect_uri": redirect, "grant_type": "authorization_code"}).encode()
    tok = http_json(f"{API}/v2/login/authorization/token", data=body)["access_token"]
    TOKEN_FILE.write_text(json.dumps({"date": today(), "access_token": tok}))
    print(f"Logged in. Access token saved to {TOKEN_FILE.name} (valid until ~3:30 AM tomorrow).")


def load_token():
    try:
        t = json.loads(TOKEN_FILE.read_text())
        return t["access_token"] if t.get("date") == today() else None
    except (OSError, ValueError, KeyError):
        return None


def resolve():
    """Nearest-expiry NIFTY futures contract from Upstox's public instrument list (no login needed)."""
    with urllib.request.urlopen(INSTRUMENTS_URL, timeout=60) as r:
        rows = json.loads(gzip.decompress(r.read()))
    now_ms = time.time() * 1000
    futs = [i for i in rows if i.get("segment") == "NSE_FO" and i.get("instrument_type") == "FUT"
            and i.get("underlying_symbol") == "NIFTY" and i.get("expiry", 0) >= now_ms]
    if not futs:
        sys.exit("No NIFTY futures found in the Upstox instrument list.")
    i = min(futs, key=lambda i: i["expiry"])
    meta = {"instrument_key": i["instrument_key"], "tradingsymbol": i["trading_symbol"],
            "expiry": dt.datetime.fromtimestamp(i["expiry"] / 1000, IST).date().isoformat(),
            "tick_size": i["tick_size"] / 100.0,          # Upstox lists tick size in paise
            "lot_size": i["lot_size"]}
    DATA.mkdir(exist_ok=True)
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def record(token, meta):
    import websocket

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
    key, n, stop = meta["instrument_key"], 0, False

    def on_open(ws):
        sub = {"guid": "nsequant", "method": "sub", "data": {"mode": "full", "instrumentKeys": [key]}}
        ws.send(json.dumps(sub).encode(), opcode=websocket.ABNF.OPCODE_BINARY)
        print("connected; streaming", meta["tradingsymbol"])

    def on_message(ws, msg):
        nonlocal n, stop
        now = time.time_ns()
        d = decode(msg)
        ff = d.get("feeds", {}).get(key, {}).get("fullFeed", {}).get("marketFF")
        if ff:
            w.writerow(flatten(ff, now, d.get("currentTs")))
            f.flush()  # a few msgs/sec: cheap, and nothing is lost on a crash
            n += 1
            if n % 1000 == 0:
                print(n, "rows")
        if dt.datetime.now(IST) >= close:
            stop = True
            ws.close()

    def on_error(ws, err):
        nonlocal stop
        print("error:", err)
        if getattr(err, "status_code", None) in (401, 403):
            stop = True
            print("Token rejected. Run: python collect_upstox.py --login")

    print(f"Recording -> {path}")
    try:
        while not stop and dt.datetime.now(IST) < close:
            ws = websocket.WebSocketApp(FEED_URL, header={"Authorization": f"Bearer {token}"},
                                        on_open=on_open, on_message=on_message, on_error=on_error)
            ws.run_forever(ping_interval=20, ping_timeout=10)   # TLS certificates verified (default)
            if not stop:
                print("disconnected; reconnecting in 5 s")
                time.sleep(5)
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        f.close()
        print(f"saved {n} rows to {path}")


def selftest():
    from upstox_client.feeder.proto import MarketDataFeedV3_pb2 as pb
    r = pb.FeedResponse(currentTs=1763013600000)
    mff = r.feeds["NSE_FO|1"].fullFeed.marketFF
    mff.ltpc.ltp, mff.ltpc.ltt, mff.ltpc.ltq = 26000.1, 1763013599000, 65
    mff.vtt, mff.tbq, mff.tsq, mff.oi = 123456, 5000.0, 6000.0, 1.5e7
    for i in range(4):   # only 4 levels, and level 4 bid qty 0 -> exercises padding and omitted zero fields
        mff.marketLevel.bidAskQuote.add(bidP=26000.0 - i * 0.1, bidQ=65 * (i + 1) if i < 3 else 0,
                                        askP=26000.1 + i * 0.1, askQ=130 * (i + 1))
    d = decode(r.SerializeToString())
    row = flatten(d["feeds"]["NSE_FO|1"]["fullFeed"]["marketFF"], 7, d["currentTs"])
    assert len(row) == len(COLS) == 9 + 4 * LEVELS
    x = dict(zip(COLS, row))
    assert x["recv_ns"] == 7 and x["exch_ts"] == 1763013600000 and x["last_trade_time"] == 1763013599000
    assert x["ltp"] == 26000.1 and x["ltq"] == 65 and x["volume"] == 123456 and x["tbq"] == 5000.0
    assert x["bp1"] == 26000.0 and x["ap1"] == 26000.1 and x["bq3"] == 195 and x["aq2"] == 260
    assert x["bq4"] == 0 and x["bp5"] == 0.0 and x["aq5"] == 0
    # read-only guarantee: no order endpoint may appear in this file
    verbs = ("place", "modify", "cancel", "exit")
    banned = "|".join([v + "_order" for v in verbs] + ["order/" + v for v in verbs])
    assert not re.search(banned, Path(__file__).read_text()), "order call found in collector"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--login", action="store_true", help="interactive daily login")
    g.add_argument("--resolve-only", action="store_true", help="print the contract and exit (no login needed)")
    g.add_argument("--selftest", action="store_true", help="offline check")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.login:
        return login()
    meta = resolve()
    print(json.dumps(meta, indent=2))
    if a.resolve_only:
        return
    token = load_token() or sys.exit("No access token for today. Run: python collect_upstox.py --login")
    record(token, meta)


if __name__ == "__main__":
    main()
