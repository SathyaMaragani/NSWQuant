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
import math
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
HEAT_W, HEAT_MS, HEAT_N = 150, 300, 400   # depth heatmap: +-150 ticks around the mid, a column every 300 ms, 2 min kept


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
        self.heat, self.last_heat, self.last_hist = deque(maxlen=HEAT_N), 0, 0.0   # depth heatmap columns
        self.tbucket, self.trade_heat = {}, deque(maxlen=8000)   # traded volume per price per column: [t0, t1, tick, buy, sell]
        self.eq = {k: deque(maxlen=3600) for k in ("A", "C", "T")}
        self.mids = deque(maxlen=20000)            # (ts, mid ticks) for markouts
        self.decide_us, self.net_ms = deque(maxlen=2000), deque(maxlen=2000)
        self.last_eq_ts, self.started = 0, time.time()
        self.n_book = self.n_trades = 0
        self.last = {}
        self.last_msg = time.time()
        self.algo_log, self.log_seq, self.last_sig = deque(maxlen=600), 0, (None, -10 ** 15)   # terminal view on the page
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
        self.log_signal(ts, prob, move, c)
        before, tk = self.snap(self.strat["C"]), (self.taker.state, len(self.taker.fills))
        self.strat["C"].on_book(ts, bpt, tuple(bq), apt, tuple(aq), move, pull_bid, pull_ask)
        self.taker.on_book(ts, bpt[0], apt[0], prob)
        self.diff_mm(ts, before, prob, pull_bid, pull_ask, move, 0.5 * (bpt[0] + apt[0]))
        self.diff_taker(ts, tk, prob)
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
        if ts - self.last_heat >= HEAT_MS:
            t0 = self.last_heat or ts
            for tk, (b, sv) in self.tbucket.items():
                self.trade_heat.append([t0, ts, tk, round(b, 4), round(sv, 4)])
            self.tbucket = {}
            self.heat.append(self.heat_column(ts, mid, bpt[0], apt[0]))
            self.last_heat = ts
        if ts - self.last_eq_ts >= 1000:
            for k, s in (("A", self.strat["A"]), ("C", self.strat["C"]), ("T", self.taker)):
                self.eq[k].append([ts, round(s.equity_ticks(mid) * self.tick, 4)])
            self.last_eq_ts = ts
        self.n_book += 1

    # ----- algorithm terminal: every line is a state change of the simulated strategy -------------------------
    def log(self, ts, who, kind, msg):
        self.log_seq += 1
        self.algo_log.append({"seq": self.log_seq, "t": ts, "s": who, "k": kind, "m": msg})

    def px(self, ticks, extra=0):
        return f"{ticks * self.tick:,.{max(0, -int(math.floor(math.log10(self.tick)))) + extra}f}"

    @staticmethod
    def snap(mm):
        return dict(mm.target), {k: (None if v is None else v[0]) for k, v in mm.live.items()}, len(mm.fills), mm.pos

    def log_signal(self, ts, prob, move, c):
        """Model output, logged when the call changes (at most every 2 s) so the terminal stays readable."""
        if c != self.last_sig[0] and ts - self.last_sig[1] >= 2000:
            arrow = ("▼ DOWN", "● FLAT", "▲ UP")[c]
            self.log(ts, "M", "SIGNAL", f"model {arrow} {prob[c]:.0%}  (down {prob[0]:.0%} · flat {prob[1]:.0%} · up {prob[2]:.0%})"
                                         f"  E[move] {move:+.2f} ticks  · {self.decide_us[-1]:.1f} µs")
            self.last_sig = (c, ts)

    def diff_mm(self, ts, before, prob=None, pull_bid=False, pull_ask=False, move=0.0, mid=None, trade=None):
        mm, (tgt0, live0, nf0, pos0) = self.strat["C"], before
        name = {+1: "BID", -1: "ASK"}
        filled = set()
        for fts, side, fpx, q in mm.fills[nf0:]:
            filled.add(side)
            pos_after = pos0 + side * q
            how = "traded through" if trade and ((side > 0 and trade[0] < fpx) or (side < 0 and trade[0] > fpx)) else "queue reached"
            kind = "EXIT" if abs(pos_after) < abs(pos0) - 1e-12 else "FILL"
            self.log(fts, "C", kind, f"{'BUY' if side > 0 else 'SELL'} {q:g} @ {self.px(fpx)}  ({how})  position {pos0:+g} → {pos_after:+g}"
                                     f"  · maker fee {self.maker_bps:g} bp")
            if abs(pos_after) < 1e-12:
                m = mid if mid is not None else (self.mids[-1][1] if self.mids else fpx)
                self.log(fts, "C", "FLAT", f"round trip closed · inventory 0 · session P&L {mm.equity_ticks(m) * self.tick:+.4f} USDT before fees")
            pos0 = pos_after
        for side in (+1, -1):
            p0, p1 = live0[side], (None if mm.live[side] is None else mm.live[side][0])
            if p1 is not None and p1 != p0:
                qa = mm.live[side][1]
                where = "behind the visible 5 levels" if qa > 1e17 else "front of the queue" if qa <= 0 else f"{qa:g} {self.symbol[:-4]} ahead in queue"
                self.log(ts, "C", "LIVE", f"{name[side]} live @ {self.px(p1)} after {mm.lat} ms simulated latency · {where}")
        if prob is None:
            return
        for side in (+1, -1):
            t0, t1 = tgt0[side], mm.target[side]
            if t0 == t1 or side in filled and t1 is None:
                continue
            fair = mid - mm.inv_skew * mm.pos / mm.size + mm.kappa * move
            why = (f"fair {self.px(fair, 1)} = mid {self.px(mid, 1)} {mm.kappa * move:+.2f}t model {-mm.inv_skew * mm.pos / mm.size:+.2f}t inventory"
                   f" ∓ half-spread {mm.half:g}t")
            if t1 is None:
                if (side > 0 and pull_bid) or (side < 0 and pull_ask):
                    r = f"model says {'▼' if side > 0 else '▲'} {max(prob):.0%} ≥ {self.tau_pull:.0%}: step away before being run over"
                else:
                    r = f"position cap {mm.qmax * mm.size:g} reached"
                self.log(ts, "C", "CANCEL", f"{name[side]} {self.px(t0)} pulled · {r}")
            elif t0 is None:
                self.log(ts, "C", "PLACE", f"{name[side]} {mm.size:g} @ {self.px(t1)} post-only · {why}")
            else:
                self.log(ts, "C", "AMEND", f"{name[side]} {self.px(t0)} → {self.px(t1)} · {why}")

    def diff_taker(self, ts, before, prob):
        t, (st0, nf0) = self.taker, before
        if st0 == "flat" and t.state == "entering":
            self.log(ts, "T", "ENTER", f"model {'▲' if t.side > 0 else '▼'} {max(prob):.0%} ≥ {t.tau:.0%} → market {'BUY' if t.side > 0 else 'SELL'} {t.size:g} (arrives in {t.lat} ms)")
        if st0 == "holding" and t.state == "exiting":
            self.log(ts, "T", "EXIT", f"held {t.hold} book events → market {'SELL' if t.side > 0 else 'BUY'} {t.size:g} to close")
        for fts, side, fpx, q in t.fills[nf0:]:
            if t.state == "flat":
                e = t.fills[-2]
                pnl = (fpx - e[2]) * e[1] * q * self.tick
                self.log(fts, "T", "FLAT", f"{'SELL' if side < 0 else 'BUY'} {q:g} @ {self.px(fpx)} · trade P&L {pnl:+.4f} USDT before {self.taker_bps:g} bp taker fee x2")
            else:
                self.log(fts, "T", "FILL", f"{'BUY' if side > 0 else 'SELL'} {q:g} @ {self.px(fpx)} (taker, crossed the spread) · holding")

    def heat_column(self, ts, mid, bid, ask):
        """[ts, lowest tick, best bid tick, best ask tick, resting size at each tick lo..lo+2W] from the full book."""
        lo, col = int(round(mid)) - HEAT_W, [0.0] * (2 * HEAT_W + 1)
        for book in (self.book.bids, self.book.asks):
            for p, q in book.items():
                k = self.ticks(p) - lo
                if 0 <= k <= 2 * HEAT_W:
                    col[k] = round(q, 3)
        return [ts, lo, bid, ask, col]

    def on_trade_msg(self, msg):
        for t in msg["data"]:
            ts, px, q, side = int(t["T"]), self.ticks(float(t["p"])), float(t["v"]), (1 if t["S"] == "Buy" else -1)
            self.tbucket.setdefault(px, [0.0, 0.0])[0 if side > 0 else 1] += q
            self.strat["A"].on_trade(ts, px, q, side)
            before = self.snap(self.strat["C"])
            self.strat["C"].on_trade(ts, px, q, side)
            self.diff_mm(ts, before, trade=(px, q, side))
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
            "last": self.last, "series": list(self.series), **self.heat_payload(12), "log": list(self.algo_log)[-120:],   # newest heat columns only; the page keeps the history
            "equity": {k: list(v) for k, v in self.eq.items()},
            "strategies": {"A": self.summary("A", S["A"], self.maker_bps), "C": self.summary("C", S["C"], self.maker_bps),
                           "T": self.summary("T", self.taker, self.taker_bps)},
            "fills": fills[-60:],
            "latency": {"decide_p50_us": float(np.percentile(lat, 50)), "decide_p99_us": float(np.percentile(lat, 99)),
                        "feed_p50_ms": float(np.percentile(net, 50)), "feed_p99_ms": float(np.percentile(net, 99))},
            "backtest": BACKTEST,
        }

    def heat_payload(self, n=None):
        cols = list(self.heat)[-n:] if n else list(self.heat)
        t0 = cols[0][0] if cols else 0
        return {"heat": cols, "trades_heat": [r for r in self.trade_heat if r[1] > t0]}

    def publish(self):
        write_json(OUT / "state.json", self.state())
        if time.time() - self.last_hist > 5:       # full heatmap history for a page that just opened (written off the feed thread)
            self.last_hist = time.time()
            hist = {"tick": self.tick, **self.heat_payload()}
            threading.Thread(target=write_json, args=(OUT / "heat.json", hist), daemon=True).start()


def write_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, default=float, separators=(",", ":")))
    for _ in range(20):
        try:
            os.replace(tmp, path); return
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
    for page in ("results.html", "results_data.json"):     # Results page: backtests + the live session
        if (ROOT / "web" / page).exists():
            shutil.copy(ROOT / "web" / page, OUT / page)
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
    stub = type("E", (), {"book": b, "ticks": staticmethod(lambda x: int(round(x)))})()
    ts, lo, bid, ask, col = Engine.heat_column(stub, 7, 100.0, 99, 101)    # tick 1, mid 100: column covers 70..130
    assert lo == 100 - HEAT_W and col[99 - lo] == 1 and col[100 - lo] == 0 and col[101 - lo] == 5 and col[106 - lo] == 2
    model = ROOT / "models" / "ethusdt_mm.npz"
    if model.exists():                     # the published state carries every key the page reads
        eng = Engine(model, "ETHUSDT")
        book = {"b": [[f"{2000 - 0.01 * i:.2f}", "1"] for i in range(10)], "a": [[f"{2000.01 + 0.01 * i:.2f}", "2"] for i in range(10)]}
        eng.on_book_msg({"type": "snapshot", "ts": 1000, "data": book})
        eng.on_trade_msg({"data": [{"T": 1100, "p": "2000.01", "v": "0.5", "S": "Buy"}]})
        eng.on_book_msg({"type": "delta", "ts": 1400, "data": {"b": [], "a": []}})
        st = eng.state()
        need = {"strategies", "equity", "series", "heat", "trades_heat", "fills", "latency", "params", "last", "tick"}
        assert need <= st.keys(), need - st.keys()
        assert len(st["heat"]) == 2 and st["trades_heat"] == [[1000, 1400, eng.ticks(2000.01), 0.5, 0.0]]
        kinds = [e["k"] for e in st["log"] if e["s"] == "C"]
        assert "PLACE" in kinds and "LIVE" in kinds and kinds.index("PLACE") < kinds.index("LIVE"), st["log"]
    print("selftest ok: live book rebuild, taker state machine, depth heat column, published state")


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
