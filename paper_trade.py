"""Live PAPER trading of NSQ-MM on Bybit's public ETHUSDT perpetual feed, with a dashboard.

    python strategy.py                  # once: backtest + save the model and tuned settings (models/ethusdt_mm.npz)
    python paper_trade.py               # live: http://127.0.0.1:8770/
    python paper_trade.py --selftest

No account, no keys, no orders: the feed is Bybit's public market-data websocket, and every order is simulated
against the live book and live trades by the same MarketMaker class the backtest uses (queue position, latency).
Three strategies run side by side on identical data:
  A  market maker without the model    C  NSQ-MM (model-skewed quotes + quote pulling)    T  TinyML signal taker
"""
import argparse
import http.server
import json
import os
import shutil
import threading
import time
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np

import tinyml as tm
from strategy import MarketMaker

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "paper"
WS_URL = "wss://stream.bybit.com/v5/public/linear"
P_KEYS = ["W1", "b1", "m1", "r1", "W2", "b2", "m2", "r2", "W3", "b3"]
LAT_MS = 50


class LiveBook:
    """Bybit orderbook.200 snapshot + delta stream -> full book (size "0" deletes a level)."""

    def __init__(self):
        self.bids, self.asks = {}, {}

    def apply(self, msg):
        d = msg["data"]
        if msg["type"] == "snapshot":
            self.bids = {float(p): float(q) for p, q in d["b"]}
            self.asks = {float(p): float(q) for p, q in d["a"]}
            return
        for side, book in (("b", self.bids), ("a", self.asks)):
            for p, q in d[side]:
                p, q = float(p), float(q)
                if q == 0:
                    book.pop(p, None)
                else:
                    book[p] = q

    def top5(self):
        if len(self.bids) < 5 or len(self.asks) < 5:
            return None
        bp = sorted(self.bids, reverse=True)[:5]
        ap = sorted(self.asks)[:5]
        if bp[0] >= ap[0]:
            return None
        return bp, [self.bids[p] for p in bp], ap, [self.asks[p] for p in ap]


class LiveTaker:
    """Incremental version of strategy.run_taker: enter at the touch after latency, hold `hold` book events."""

    def __init__(self, size, tau, hold, lat_ms):
        self.size, self.tau, self.hold, self.lat = size, tau, hold, lat_ms
        self.state, self.pos, self.cash_ticks, self.fills = "flat", 0.0, 0.0, []
        self.side = 0; self.t_due = 0; self.events = 0

    def on_book(self, ts, bp1, ap1, prob):
        if self.state == "flat":
            c = int(np.argmax(prob))
            if c != 1 and prob[c] >= self.tau:
                self.state, self.side, self.t_due = "entering", (1 if c == 2 else -1), ts + self.lat
        elif self.state == "entering" and ts >= self.t_due:
            px = ap1 if self.side > 0 else bp1
            self._fill(ts, self.side, px)
            self.state, self.events = "holding", 0
        elif self.state == "holding":
            self.events += 1
            if self.events >= self.hold:
                self.state, self.t_due = "exiting", ts + self.lat
        elif self.state == "exiting" and ts >= self.t_due:
            px = bp1 if self.side > 0 else ap1
            self._fill(ts, -self.side, px)
            self.state = "flat"

    def _fill(self, ts, side, px):
        self.fills.append((ts, side, px, self.size))
        self.pos += side * self.size
        self.cash_ticks -= side * px * self.size

    def equity_ticks(self, mid):
        return self.cash_ticks + self.pos * mid


class Engine:
    def __init__(self, model_path, symbol):
        m = np.load(model_path)
        self.symbol, self.tick, self.unit = symbol, float(m["tick"]), float(m["unit"])
        self.P = tuple(m[k] for k in P_KEYS)
        self.mu, self.a, self.out_scale, self.class_move = m["mu"], m["a"], float(m["out_scale"]), m["class_move"]
        self.tau_pull, self.maker_bps, self.taker_bps = float(m["tau_pull"]), float(m["maker_bps"]), float(m["taker_bps"])
        self.strat = {
            "A": MarketMaker(size=self.unit, half=float(m["A_half"]), inv_skew=float(m["A_inv"]), use_signal=False, pull=False, latency_ms=LAT_MS),
            "C": MarketMaker(size=self.unit, half=float(m["C_half"]), inv_skew=float(m["C_inv"]), kappa=float(m["C_kappa"]),
                             use_signal=True, pull=True, latency_ms=LAT_MS),
        }
        self.taker = LiveTaker(self.unit, float(m["tau_taker"]), int(m["T_hold"]), LAT_MS)
        self.params = {"C": {"half": float(m["C_half"]), "inv_skew": float(m["C_inv"]), "kappa": float(m["C_kappa"]), "tau_pull": self.tau_pull},
                       "A": {"half": float(m["A_half"]), "inv_skew": float(m["A_inv"])},
                       "T": {"tau": float(m["tau_taker"]), "hold_events": int(m["T_hold"])}}
        self.st, self.xb, self.xq, self.h1, self.h2, self.lg = tm.buffers({"mu": self.mu, "P": self.P})
        self.book = LiveBook()
        self.series = deque(maxlen=900)            # ~90 s of book events
        self.eq = {k: deque(maxlen=3600) for k in ("A", "C", "T")}
        self.mids = deque(maxlen=20000)            # (ts, mid ticks) for markouts
        self.decide_us, self.net_ms = deque(maxlen=2000), deque(maxlen=2000)
        self.last_eq_ts, self.started = 0, time.time()
        self.n_book = self.n_trades = 0
        self.last = {}
        self.last_msg = time.time()
        OUT.mkdir(parents=True, exist_ok=True)
        tm.tick_predict(np.zeros(tm.ST_SIZE), np.ones(23), self.tick, self.mu, self.a, self.xb.copy(), self.xq.copy(),
                        *self.P, self.h1.copy(), self.h2.copy(), self.lg.copy())   # compile before the feed starts

    def ticks(self, x):
        return int(round(x / self.tick))

    def on_book_msg(self, msg):
        self.book.apply(msg)
        top = self.book.top5()
        if top is None:
            return
        bp, bq, ap, aq = top
        ts = int(msg["ts"])
        self.net_ms.append(time.time() * 1000 - ts)
        r = np.array(bp + bq + ap + aq + [0.0, sum(self.book.bids.values()), sum(self.book.asks.values())])
        t0 = time.perf_counter_ns()
        tm.tick_predict(self.st, r, self.tick, self.mu, self.a, self.xb, self.xq, *self.P, self.h1, self.h2, self.lg)
        z = self.lg * self.out_scale
        prob = np.exp(z - z.max()); prob /= prob.sum()
        move = float(prob @ self.class_move)
        c = int(prob.argmax())
        pull_bid, pull_ask = c == 0 and prob[c] >= self.tau_pull, c == 2 and prob[c] >= self.tau_pull
        self.decide_us.append((time.perf_counter_ns() - t0) / 1000)
        bpt, apt = tuple(self.ticks(p) for p in bp), tuple(self.ticks(p) for p in ap)
        self.strat["A"].on_book(ts, bpt, tuple(bq), apt, tuple(aq))
        self.strat["C"].on_book(ts, bpt, tuple(bq), apt, tuple(aq), move, pull_bid, pull_ask)
        self.taker.on_book(ts, bpt[0], apt[0], prob)
        mid = 0.5 * (bpt[0] + apt[0])
        self.mids.append((ts, mid))
        C = self.strat["C"]
        fair = mid - C.inv_skew * C.pos / C.size + C.kappa * move
        self.last = {"ts": ts, "mid": mid * self.tick, "bid": bp[0], "ask": ap[0], "probs": prob.round(4).tolist(),
                     "move_ticks": move, "pull_bid": bool(pull_bid), "pull_ask": bool(pull_ask),
                     "fair": fair * self.tick, "inv_term": -C.inv_skew * C.pos / C.size * self.tick,
                     "alpha_term": C.kappa * move * self.tick,
                     "quote_bid": None if C.target[+1] is None else C.target[+1] * self.tick,
                     "quote_ask": None if C.target[-1] is None else C.target[-1] * self.tick,
                     "book": {"bp": bp, "bq": bq, "ap": ap, "aq": aq}}
        self.series.append([ts, round(mid * self.tick, 4), bp[0], ap[0], self.last["quote_bid"], self.last["quote_ask"]])
        if ts - self.last_eq_ts >= 1000:
            for k, s in (("A", self.strat["A"]), ("C", self.strat["C"]), ("T", self.taker)):
                self.eq[k].append([ts, round(s.equity_ticks(mid) * self.tick, 4)])
            self.last_eq_ts = ts
        self.n_book += 1

    def on_trade_msg(self, msg):
        for t in msg["data"]:
            ts, px, q, side = int(t["T"]), self.ticks(float(t["p"])), float(t["v"]), (1 if t["S"] == "Buy" else -1)
            self.strat["A"].on_trade(ts, px, q, side)
            self.strat["C"].on_trade(ts, px, q, side)
            self.n_trades += 1

    def _markout(self, fills, horizon_ms=10_000):
        if not fills or not self.mids:
            return None, 0
        mts = np.array([m[0] for m in self.mids]); mm = np.array([m[1] for m in self.mids])
        vals = []
        for ts, side, px, _ in fills:
            if ts + horizon_ms > mts[-1] or ts < mts[0]:
                continue
            k = np.searchsorted(mts, ts + horizon_ms)
            vals.append(side * (mm[min(k, len(mm) - 1)] - px))
        return (float(np.mean(vals)) if vals else None), len(vals)

    def summary(self, key, s, bps):
        mid_t = self.mids[-1][1] if self.mids else 0.0
        gross = s.equity_ticks(mid_t) * self.tick
        f = s.fills
        notional = sum(px * q for _, _, px, q in f) * self.tick
        mk, n_mk = self._markout(f)
        return {"gross": gross, "fees_vip0": notional * bps * 1e-4, "net_vip0": gross - notional * bps * 1e-4,
                "fills": len(f), "position": s.pos, "markout10s_ticks": mk, "markout_n": n_mk,
                "volume": notional}

    def state(self):
        S = self.strat
        fills = []
        for key, s in (("A", S["A"]), ("C", S["C"]), ("T", self.taker)):
            for ts, side, px, q in list(s.fills)[-40:]:
                fills.append({"t": ts, "strategy": key, "side": side, "price": px * self.tick, "qty": q})
        fills.sort(key=lambda x: x["t"])
        lat = np.array(self.decide_us) if self.decide_us else np.array([0.0])
        net = np.array(self.net_ms) if self.net_ms else np.array([0.0])
        return {
            "symbol": self.symbol, "mode": "paper", "venue": "Bybit public feed", "updated": time.strftime("%H:%M:%S"),
            "uptime_s": int(time.time() - self.started), "book_events": self.n_book, "trades": self.n_trades,
            "unit": self.unit, "tick": self.tick, "latency_ms_sim": LAT_MS,
            "fees_bps": {"maker": self.maker_bps, "taker": self.taker_bps}, "params": self.params,
            "last": self.last, "series": list(self.series), "equity": {k: list(v) for k, v in self.eq.items()},
            "strategies": {"A": self.summary("A", S["A"], self.maker_bps), "C": self.summary("C", S["C"], self.maker_bps),
                           "T": self.summary("T", self.taker, self.taker_bps)},
            "fills": fills[-60:],
            "latency": {"decide_p50_us": float(np.percentile(lat, 50)), "decide_p99_us": float(np.percentile(lat, 99)),
                        "feed_p50_ms": float(np.percentile(net, 50)), "feed_p99_ms": float(np.percentile(net, 99))},
            "backtest": BACKTEST,
        }

    def publish(self):
        tmp = OUT / "state.json.tmp"
        tmp.write_text(json.dumps(self.state(), default=float))
        for _ in range(20):
            try:
                os.replace(tmp, OUT / "state.json"); break
            except PermissionError:
                time.sleep(0.005)


def load_backtest():
    p = ROOT / "results" / "trading" / "backtest.csv"
    if not p.exists():
        return None
    import pandas as pd
    d = pd.read_csv(p)
    cols = ["market", "strategy", "fills", "gross_pnl", "net_vip0", "net_rebate_0.5bp", "latency_ms"]
    return d[[c for c in cols if c in d]].round(3).to_dict(orient="records")


BACKTEST = load_backtest()


def serve(port):
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "web" / "trading.html", OUT / "index.html")
    shutil.copytree(ROOT / "web" / "vendor", OUT / "vendor", dirs_exist_ok=True)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Quiet, directory=str(OUT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"paper-trading dashboard: http://127.0.0.1:{port}/")


def stream(eng, symbol):
    import websocket
    topics = [f"orderbook.200.{symbol}", f"publicTrade.{symbol}"]
    last_pub = [0.0]

    def on_open(ws):
        ws.send(json.dumps({"op": "subscribe", "args": topics}))
        print("connected:", ", ".join(topics))

        def ping():
            while ws.keep_running:
                time.sleep(20)
                try:
                    ws.send(json.dumps({"op": "ping"}))
                except Exception:
                    return
        threading.Thread(target=ping, daemon=True).start()

    def on_message(ws, raw):
        msg = json.loads(raw)
        topic = msg.get("topic", "")
        eng.last_msg = time.time()
        if topic.startswith("orderbook"):
            eng.on_book_msg(msg)
        elif topic.startswith("publicTrade"):
            eng.on_trade_msg(msg)
        if time.time() - last_pub[0] > 0.5:
            eng.publish(); last_pub[0] = time.time()

    def on_error(ws, err):
        print("error:", err)

    while True:                                    # reconnect forever; the book is rebuilt from the next snapshot
        ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_message, on_error=on_error)
        ws.run_forever(ping_interval=30, ping_timeout=10)
        print("disconnected; reconnecting in 3 s")
        time.sleep(3)


def selftest():
    b = LiveBook()
    b.apply({"type": "snapshot", "data": {"b": [[str(100 - i), "1"] for i in range(6)], "a": [[str(101 + i), "2"] for i in range(6)]}})
    b.apply({"type": "delta", "data": {"b": [["100", "0"]], "a": [["101", "5"]]}})
    bp, bq, ap, aq = b.top5()
    assert bp[0] == 99 and aq[0] == 5 and len(bp) == 5
    t = LiveTaker(size=1.0, tau=0.5, hold=2, lat_ms=0)
    up = np.array([0.1, 0.2, 0.7])
    t.on_book(0, 100, 101, up)      # decide buy
    t.on_book(1, 100, 101, up)      # enter at ask 101
    t.on_book(2, 102, 103, up); t.on_book(3, 102, 103, up)   # hold 2 events -> exit decided
    t.on_book(4, 104, 105, up)      # exit at bid 104
    assert t.fills[0][2] == 101 and t.fills[1][2] == 104 and t.pos == 0 and t.equity_ticks(0) == 3
    print("selftest ok: live book rebuild and taker state machine")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        model = ROOT / "models" / f"{a.symbol.lower()}_mm.npz"
        if not model.exists():
            raise SystemExit(f"{model.name} missing: run python strategy.py first")
        eng = Engine(model, a.symbol)
        serve(a.port)
        stream(eng, a.symbol)
