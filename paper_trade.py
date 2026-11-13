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

import hftsim as H
import strategies as SX
import tinyml as tm
from multiday import CHOSEN
from strategy import FEES, MarketMaker

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "paper"
WS_URL = "wss://stream.bybit.com/v5/public/linear"
P_KEYS = ["W1", "b1", "m1", "r1", "W2", "b2", "m2", "r2", "W3", "b3"]
LAT_MS = 50
X_WARM_S = 15 * 60       # X sits out while the rebuilt Binance book fills in (its +-2-3 % bands start incomplete)
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
        return self.depth(5)

    def depth(self, n):
        if len(self.bids) < n or len(self.asks) < n:
            return None
        bp = sorted(self.bids, reverse=True)[:n]
        ap = sorted(self.asks)[:n]
        if bp[0] >= ap[0]:
            return None
        return bp, [self.bids[p] for p in bp], ap, [self.asks[p] for p in ap]


LIVE_ST = ("PENDING", "OPEN")


class Blotter:
    """What a trading platform shows for one simulated account: order book (every order and its status),
    trade book (every execution) and the net position with average price and realised P&L. Prices in ticks."""

    def __init__(self, name, tick=1.0, fee_bps=0.0, keep=300):
        self.name, self.tick, self.bps = name, tick, fee_bps
        self.orders, self.trades = deque(maxlen=keep), deque(maxlen=keep)
        self.n_orders = self.n_trades = 0
        self.pos = self.avg = self.realized = self.fees = 0.0
        self.buy_q = self.buy_v = self.sell_q = self.sell_v = 0.0

    def new(self, ts, side, px, qty, typ):
        self.n_orders += 1
        o = {"id": f"{self.name}{self.n_orders:06d}", "s": self.name, "t": ts, "tu": ts, "side": side, "type": typ,
             "px": px, "qty": qty, "filled": 0.0, "avg": None, "status": "PENDING", "mods": 0, "q": None, "note": ""}
        self.orders.append(o)
        return o

    def open(self, o, ts, px, queue):
        if o["status"] == "OPEN" and o["px"] != px:
            o["mods"] += 1                                 # amended: same order id, new price, back of the new queue
        o.update(status="OPEN", px=px, tu=ts, q=queue)

    def close(self, o, ts, status, note=""):
        if o["status"] in LIVE_ST:
            o.update(status=status, tu=ts, note=note)

    def fill(self, o, ts, px, maker, q=None):
        q = o["qty"] - o["filled"] if q is None else q
        o["avg"] = px if not o["filled"] else (o["avg"] * o["filled"] + px * q) / (o["filled"] + q)
        o["filled"] = round(o["filled"] + q, 9)
        o.update(tu=ts, note="")
        if o["filled"] >= o["qty"] - 1e-12:
            o["status"] = "COMPLETE"
        self.n_trades += 1
        fee = px * self.tick * q * self.bps * 1e-4
        self.fees += fee
        self.trades.append({"id": f"T{self.name}{self.n_trades:06d}", "oid": o["id"], "s": self.name, "t": ts,
                            "side": o["side"], "px": px, "qty": q, "fee": fee, "maker": maker})
        side = o["side"]
        if side > 0:
            self.buy_q += q; self.buy_v += px * q
        else:
            self.sell_q += q; self.sell_v += px * q
        if abs(self.pos) < 1e-12 or (self.pos > 0) == (side > 0):        # opening or adding: average-cost price
            self.avg = (self.avg * abs(self.pos) + px * q) / (abs(self.pos) + q)
            self.pos = round(self.pos + side * q, 9)
            return
        close = min(q, abs(self.pos))                                    # reducing: realise against the average
        self.realized += (px - self.avg) * close * -side
        self.pos = round(self.pos + side * q, 9)
        if q > close:
            self.avg = px                                                # flipped through zero
        elif abs(self.pos) < 1e-12:
            self.pos, self.avg = 0.0, 0.0

    def position(self, mid):
        t = self.tick
        unreal = (mid - self.avg) * self.pos if self.pos else 0.0
        return {"qty": self.pos, "avg": self.avg * t if self.pos else None, "mark": mid * t,
                "unrealised": unreal * t, "realised": self.realized * t, "fees": self.fees,
                "net": (unreal + self.realized) * t - self.fees,
                "buy_qty": self.buy_q, "buy_avg": self.buy_v / self.buy_q * t if self.buy_q else None,
                "sell_qty": self.sell_q, "sell_avg": self.sell_v / self.sell_q * t if self.sell_q else None}

    def publish(self, mid, n=80):
        t = self.tick
        pt = lambda v: None if v is None else round(v * t, 8)
        orders = [{**o, "px": pt(o["px"]), "avg": pt(o["avg"])} for o in list(self.orders)[-n:]]
        trades = [{**f, "px": pt(f["px"]), "value": f["px"] * t * f["qty"]} for f in list(self.trades)[-n:]]
        return {"orders": orders, "trades": trades, "position": self.position(mid),
                "open": sum(o["status"] in LIVE_ST for o in self.orders), "n_orders": self.n_orders, "n_trades": self.n_trades}


class _Sent(list):
    """MarketMaker's per-side message queue, with a hook on every message it sends."""

    def __init__(self, mm, side):
        super().__init__()
        self.mm, self.side = mm, side

    def append(self, item):
        self.mm._sent(self.side, *item)
        super().append(item)


class TrackedMM(MarketMaker):
    """The backtest's MarketMaker, unchanged in behaviour, with every order recorded in a Blotter. Each side has one
    working order record that mirrors the strategy's live slot: a price sent to an empty side = new order (PENDING
    until the latency passes), a new price on a working order = modify (same id), None = cancel."""

    def __init__(self, name, tick, fee_bps, **kw):
        super().__init__(**kw)
        self.blotter = Blotter(name, tick, fee_bps)
        self.cur = {+1: None, -1: None}                    # side -> the working order record on that side
        self.pending = {+1: _Sent(self, +1), -1: _Sent(self, -1)}

    def _sent(self, side, act, px):
        o = self.cur[side]
        if px is not None and (o is None or o["status"] not in LIVE_ST):
            self.cur[side] = self.blotter.new(act - self.lat, side, px, self.size, "LIMIT")

    def _end(self, side, ts, status, note=""):
        if self.cur[side] is not None:
            self.blotter.close(self.cur[side], ts, status, note)
        self.cur[side] = None

    def _activate(self, ts):                               # same rules as MarketMaker._activate, plus the records
        for side in (+1, -1):
            q = self.pending[side]
            while q and q[0][0] <= ts:
                _, px = q.pop(0)
                if px is None:
                    self.live[side] = None
                    self._end(side, ts, "CANCELLED")
                    continue
                bp, _, ap, _ = self.book
                if (side == +1 and px >= ap[0]) or (side == -1 and px <= bp[0]):
                    self.live[side] = None                 # post-only order rejected
                    self._end(side, ts, "REJECTED", "post-only: would cross the spread")
                    continue
                self.live[side] = [px, self._queue_at(side, px)]
                o = self.cur[side]
                if o is None or o["status"] not in LIVE_ST:   # a modify that outlived its filled order: a new order
                    o = self.cur[side] = self.blotter.new(ts, side, px, self.size, "LIMIT")
                self.blotter.open(o, ts, px, self.live[side][1])

    def _fill(self, ts, side, px):
        super()._fill(ts, side, px)
        o = self.cur[side] or self.blotter.new(ts, side, px, self.size, "LIMIT")
        self.blotter.fill(o, ts, px, maker=True)
        self.cur[side] = None


class BinanceSignal:
    """Algorithm X's cross-venue signal, live. Binance USD-M book rebuilt from a REST snapshot + the diff stream
    (Binance's documented procedure); every 30 s, like the bookDepth archive the backtest used: size resting within
    +-2 % and +-3 % of the mid, imbalance (bid - ask) / (bid + ask) of their mean, z-scored over the last hour
    (min 4 snapshots), usable lag_s after the snapshot. Public market data only; nothing is sent."""
    REST = "https://fapi.binance.com/fapi/v1/depth?symbol={}&limit=1000"
    WS = "wss://fstream.binance.com/ws/{}@depth@500ms"

    def __init__(self, symbol, every_s=30, window_s=3600, lag_s=1):
        self.symbol, self.every, self.window, self.lag = symbol, every_s, window_s, lag_s
        self.bids, self.asks, self.buf, self.last_u, self.synced, self.snap_L = {}, {}, [], None, False, None
        self.lock = threading.Lock()
        self.hist = deque()                        # (t, imbalance) over the last window
        self.z, self.z_from, self.prev_z = float("nan"), 0.0, float("nan")
        self.next_t, self.reach, self.resyncs, self.last_imb = 0.0, (0.0, 0.0), 0, None

    def band_imbalance(self):
        """(bid - ask) / (bid + ask) of the mean of the +-2 % and +-3 % bands; also how far the rebuilt book reaches."""
        with self.lock:
            if not self.bids or not self.asks:
                return None
            bb, ba = max(self.bids), min(self.asks)
            mid = 0.5 * (bb + ba)
            b = sum((q for p, q in self.bids.items() if p >= mid * 0.97)) + sum((q for p, q in self.bids.items() if p >= mid * 0.98))
            a = sum((q for p, q in self.asks.items() if p <= mid * 1.03)) + sum((q for p, q in self.asks.items() if p <= mid * 1.02))
            self.reach = ((mid - min(self.bids)) / mid * 100, (max(self.asks) - mid) / mid * 100)
        return (b - a) / (b + a) if b + a > 0 else None

    def step(self, now):
        """Call often with wall-clock seconds; takes a snapshot every `every` s and updates the z-score."""
        if now < self.next_t or not self.synced:
            return
        self.next_t = now + self.every
        x = self.band_imbalance()
        if x is None:
            return
        self.last_imb = x
        self.hist.append((now, x))
        while self.hist and self.hist[0][0] <= now - self.window:
            self.hist.popleft()
        if len(self.hist) >= 4:
            v = np.array([h[1] for h in self.hist])
            sd = v.std(ddof=1)
            self.prev_z, self.z, self.z_from = self.value(now), ((x - v.mean()) / sd if sd > 0 else 0.0), now + self.lag

    def value(self, now):
        return self.z if now >= self.z_from else self.prev_z

    # ----- book maintenance --------------------------------------------------
    def _apply(self, m):
        for side, book in (("b", self.bids), ("a", self.asks)):
            for p, q in m[side]:
                p, q = float(p), float(q)
                if q == 0:
                    book.pop(p, None)
                else:
                    book[p] = q

    def _feed(self, m):
        """Binance's procedure: buffer, take a REST snapshot (lastUpdateId L), drop events with u < L, the first applied
        event must have U <= L <= u, then every event's pu must equal the previous u; a gap means a fresh snapshot."""
        if self.synced:
            if m["pu"] != self.last_u:
                self._restart(m)
            else:
                self._apply(m)
                self.last_u = m["u"]
        elif self.snap_L is None:
            self.buf.append(m)
        elif m["u"] >= self.snap_L:
            if m["U"] > self.snap_L:
                self._restart(m)                       # the stream skipped past the snapshot
            else:
                self._apply(m)
                self.last_u, self.synced, self.snap_L = m["u"], True, None

    def _restart(self, m):
        self.synced, self.snap_L, self.buf = False, None, [m]
        self.resyncs += 1
        threading.Thread(target=self.resync, daemon=True).start()

    def on_event(self, m):
        with self.lock:
            self._feed(m)

    def fetch_snapshot(self):
        import urllib.request
        return json.load(urllib.request.urlopen(self.REST.format(self.symbol.upper()), timeout=10))

    def resync(self):
        time.sleep(1.0)                                # let the stream buffer a few events first
        while True:
            try:
                d = self.fetch_snapshot()
                break
            except Exception as e:                     # network hiccup: keep buffering, try again
                print("binance snapshot failed:", e)
                time.sleep(5)
        with self.lock:
            self.bids = {float(p): float(q) for p, q in d["bids"]}
            self.asks = {float(p): float(q) for p, q in d["asks"]}
            self.snap_L, buf, self.buf = d["lastUpdateId"], self.buf, []
            for m in buf:
                self._feed(m)

    def run(self):
        import websocket
        while True:
            self.synced, self.snap_L, self.buf = False, None, []
            ws = websocket.WebSocketApp(self.WS.format(self.symbol.lower()), on_message=lambda w, raw: self.on_event(json.loads(raw)),
                                        on_open=lambda w: threading.Thread(target=self.resync, daemon=True).start())
            ws.run_forever(ping_interval=30, ping_timeout=10)
            time.sleep(3)


class LiveX(SX.ImbalanceMM):
    """Algorithm X exactly as tested (strategies.ImbalanceMM, frozen 9-Nov settings), except that it sits out with no
    quotes while the Binance signal is unavailable (warm-up, feed down) instead of quoting blind."""

    def targets(self, env):
        if env.aux is None or not math.isfinite(env.aux[self.zcol]):
            return {+1: None, -1: None}
        return super().targets(env)


class TrackedExchange(H.Exchange):
    """hftsim's Exchange (the simulator behind the 9-day test), driven by the live feed, with every order recorded in
    a Blotter. A modify keeps its order id while the exchange swaps the old order for the new one."""

    def __init__(self, profile, tick, name, fee_bps):
        super().__init__(profile, tick)
        self.blotter = Blotter(name, tick, fee_bps)
        self.rec = {}                                      # exchange order id -> blotter record (two ids during a modify)

    def place(self, side, px, qty, post_only=True):
        oid = super().place(side, px, qty, post_only)
        if oid:
            self.rec[oid] = self.blotter.new(self.now, side, int(px), qty, "LIMIT")
        return oid

    def replace(self, oid, px):
        new = super().replace(oid, px)
        if new and oid in self.rec:
            self.rec[new] = self.rec[oid]
        return new

    def cancel(self, oid):
        ok = super().cancel(oid)
        if ok and oid in self.rec:
            self.rec[oid]["cx"] = True
        return ok

    def _fill(self, side, px, qty, maker, oid):
        super()._fill(side, px, qty, maker, oid)
        r = self.rec.get(oid)
        if r is not None:
            self.blotter.fill(r, self.now, px, maker, qty)

    def sync(self):
        """After each event: live orders -> OPEN at their price and queue; orders gone without a fill -> CANCELLED or REJECTED."""
        for oid, r in list(self.rec.items()):
            o = self.orders.get(oid)
            if o is not None:
                if o.live:
                    self.blotter.open(r, self.now, o.px, o.queue_ahead)
                continue
            del self.rec[oid]
            if any(v is r for v in self.rec.values()):
                continue                                   # the other id of a modify still carries this order
            if r.get("cx"):
                self.blotter.close(r, self.now, "CANCELLED")
            else:
                self.blotter.close(r, self.now, "REJECTED", "post-only: would cross the spread")


class LiveTaker:
    """Incremental version of strategy.run_taker: enter at the touch after latency, hold `hold` book events."""

    def __init__(self, size, tau, hold, lat_ms, blotter=None):
        self.size, self.tau, self.hold, self.lat = size, tau, hold, lat_ms
        self.state, self.pos, self.cash_ticks, self.fills = "flat", 0.0, 0.0, []
        self.side = 0; self.t_due = 0; self.events = 0
        self.blotter, self.order = blotter or Blotter("T"), None

    def on_book(self, ts, bp1, ap1, prob):
        if self.state == "flat":
            c = int(np.argmax(prob))
            if c != 1 and prob[c] >= self.tau:
                self.state, self.side, self.t_due = "entering", (1 if c == 2 else -1), ts + self.lat
                self.order = self.blotter.new(ts, self.side, None, self.size, "MARKET")
        elif self.state == "entering" and ts >= self.t_due:
            px = ap1 if self.side > 0 else bp1
            self._fill(ts, self.side, px)
            self.state, self.events = "holding", 0
        elif self.state == "holding":
            self.events += 1
            if self.events >= self.hold:
                self.state, self.t_due = "exiting", ts + self.lat
                self.order = self.blotter.new(ts, -self.side, None, self.size, "MARKET")
        elif self.state == "exiting" and ts >= self.t_due:
            px = bp1 if self.side > 0 else ap1
            self._fill(ts, -self.side, px)
            self.state = "flat"

    def _fill(self, ts, side, px):
        self.blotter.fill(self.order, ts, px, maker=False)
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
            "A": TrackedMM("A", self.tick, self.maker_bps, size=self.unit, half=float(m["A_half"]), inv_skew=float(m["A_inv"]),
                           use_signal=False, pull=False, latency_ms=LAT_MS),
            "C": TrackedMM("C", self.tick, self.maker_bps, size=self.unit, half=float(m["C_half"]), inv_skew=float(m["C_inv"]),
                           kappa=float(m["C_kappa"]), use_signal=True, pull=True, latency_ms=LAT_MS),
        }
        self.taker = LiveTaker(self.unit, float(m["tau_taker"]), int(m["T_hold"]), LAT_MS, Blotter("T", self.tick, self.taker_bps))
        self.tape = deque(maxlen=60)               # market time & sales: [ts, price, qty, aggressor side]
        # algorithm X: the 9-day tested strategy and simulator, frozen settings, HFT-firm profile as tested
        name = next((k for k in CHOSEN if k.split()[0] == symbol and "perpetual" in k), None)
        self.bn, self.xex, self.xk, self.x_logged_z = BinanceSignal(symbol), None, 0, 0.0
        self.xwarm_until = time.time() + X_WARM_S
        if name:
            self.xprof, self.xkw = H.profiles(*FEES[name])["hft_firm"], CHOSEN[name]["X"]
            self.xex = TrackedExchange(self.xprof, self.tick, "X", self.xprof.maker_bps)
            self.xex.strategy = LiveX(size=self.unit, **self.xkw)
            self.xex.strategy.on_start(self.xex)
            self.xex.signal = (0.0, False, False, None)
        self.params = {"C": {"half": float(m["C_half"]), "inv_skew": float(m["C_inv"]), "kappa": float(m["C_kappa"]), "tau_pull": self.tau_pull},
                       "A": {"half": float(m["A_half"]), "inv_skew": float(m["A_inv"])},
                       "T": {"tau": float(m["tau_taker"]), "hold_events": int(m["T_hold"])}}
        self.st, self.xb, self.xq, self.h1, self.h2, self.lg = tm.buffers({"mu": self.mu, "P": self.P})
        self.book = LiveBook()
        self.series = deque(maxlen=900)            # ~90 s of book events
        self.heat, self.last_heat, self.last_hist = deque(maxlen=HEAT_N), 0, 0.0   # depth heatmap columns
        self.tbucket, self.trade_heat = {}, deque(maxlen=8000)   # traded volume per price per column: [t0, t1, tick, buy, sell]
        self.eq = {k: deque(maxlen=3600) for k in ("A", "C", "T", "X")}
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
        if self.xex is not None:
            now = time.time()
            warm = now >= self.xwarm_until
            if warm:
                self.bn.step(now)                      # history starts once the rebuilt book is complete
            z = self.bn.value(now) if warm else math.nan
            self.xex.aux = (math.nan, math.nan, z, math.nan)
            nt, self.xk = self.xex.blotter.n_trades, self.xk + 1
            self.xex.on_book(ts, list(bpt), list(bq), list(apt), list(aq), self.xk)
            self.xex.sync()
            self.log_x(ts, nt, z)
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
            if self.xex is not None:
                self.eq["X"].append([ts, round(self.xex.equity(), 4)])
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
            self.tape.append([ts, float(t["p"]), q, side])
            self.strat["A"].on_trade(ts, px, q, side)
            before = self.snap(self.strat["C"])
            self.strat["C"].on_trade(ts, px, q, side)
            self.diff_mm(ts, before, trade=(px, q, side))
            if self.xex is not None:
                nt, self.xk = self.xex.blotter.n_trades, self.xk + 1
                self.xex.on_trade(ts, px, q, side, self.xk)
                self.xex.sync()
                self.log_x(ts, nt)
            self.n_trades += 1

    def log_x(self, ts, n_before, z=None):
        """Terminal lines for X: each new Binance reading and what it does to the quotes, and every fill."""
        bl, new = self.xex.blotter, self.xex.blotter.n_trades - n_before
        for f in (list(bl.trades)[-new:] if new else []):
            self.log(f["t"], "X", "FILL", f"{'BUY' if f['side'] > 0 else 'SELL'} {f['qty']:g} @ {self.px(f['px'])}  order {f['oid']}"
                                         f"  position {bl.pos:+g}  · rebate {-f['fee']:+.4f} USDT")
        if z is not None and math.isfinite(z) and self.bn.z_from != self.x_logged_z:
            self.x_logged_z = self.bn.z_from
            k = self.xkw
            lean, half = k["c1_bp"] * z, k["half_bp"]
            act = ("bid joins the best bid, ask stands back: expecting a rise" if lean > half else
                   "ask joins the best ask, bid stands back: expecting a fall" if lean < -half else "both sides stand back from the touch")
            self.log(ts, "X", "SIGNAL", f"Binance ±2–3 % book imbalance {self.bn.last_imb:+.3f} → z {z:+.2f} → fair value "
                                        f"{lean:+.2f} bp vs mid ({k['c1_bp']:g} bp per z) · {act}")

    def summary_x(self):
        ex, bn, now = self.xex, self.bn, time.time()
        f = [(x.ts, x.side, x.px, x.qty) for x in ex.fills]
        vol = sum(x.px * x.qty for x in ex.fills) * self.tick
        gross = ex.equity() if ex.bp is not None else 0.0
        mk, n_mk = self._markout(f)
        z = bn.value(now)
        return {"gross": gross, "fees": ex.fees, "net": gross - ex.fees, "fees_vip0": vol * self.maker_bps * 1e-4,
                "net_vip0": gross - vol * self.maker_bps * 1e-4, "fills": len(f), "position": ex.pos, "volume": vol,
                "markout10s_ticks": mk, "markout_n": n_mk,
                "signal": {"z": z if math.isfinite(z) else None, "imbalance": bn.last_imb, "snapshots": len(bn.hist),
                           "synced": bn.synced, "reach_pct": [round(v, 2) for v in bn.reach], "resyncs": bn.resyncs,
                           "warmup_s_left": max(0, int(self.xwarm_until - now))}}

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
        if self.xex is not None:
            fills += [{"t": x.ts, "strategy": "X", "side": x.side, "price": x.px * self.tick, "qty": x.qty} for x in self.xex.fills[-40:]]
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
                           "T": self.summary("T", self.taker, self.taker_bps),
                           **({"X": self.summary_x()} if self.xex is not None else {})},
            "x": None if self.xex is None else {**self.xkw, "profile": self.xprof.name, "maker_bps": self.xprof.maker_bps,
                                               "max_pos": self.xex.strategy.max_pos, "unit": self.unit},
            "fills": fills[-60:],
            "latency": {"decide_p50_us": float(np.percentile(lat, 50)), "decide_p99_us": float(np.percentile(lat, 99)),
                        "feed_p50_ms": float(np.percentile(net, 50)), "feed_p99_ms": float(np.percentile(net, 99))},
            "backtest": BACKTEST,
            "desk": self.desk(),
        }

    def desk(self):
        """Trading-platform view: market depth (10 levels), time & sales, and each strategy's orders, trades, position."""
        mid_t = self.mids[-1][1] if self.mids else 0.0
        d = self.book.depth(10)
        books = {k: s.blotter.publish(mid_t) for k, s in (("C", self.strat["C"]), ("A", self.strat["A"]), ("T", self.taker))}
        if self.xex is not None:
            books["X"] = self.xex.blotter.publish(mid_t)
        return {"depth": None if d is None else dict(zip(("bp", "bq", "ap", "aq"), d)), "tape": list(self.tape)[-40:], "books": books}

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
    for page in ("results.html", "results_data.json", "fx.js", "fx.css"):   # Results page + shared 3D effects
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
    # blotter: average price, realised P&L across a flip, order statuses
    bl = Blotter("X", tick=0.5, fee_bps=0.0)
    for side, px in ((+1, 100), (+1, 104), (-1, 110), (-1, 98), (+1, 96)):
        bl.fill(bl.new(0, side, px, 1.0, "LIMIT"), 1, px, maker=True)
    # long 2 @ 102 -> sell 1 @ 110 (+8) -> sell 1 @ 98 (-4) -> flat -> buy 1 @ 96: long 1 @ 96, realised 4 ticks = 2.0
    p = bl.position(100)
    assert p["qty"] == 1 and p["avg"] == 48.0 and p["realised"] == 2.0 and p["unrealised"] == 2.0
    assert all(o["status"] == "COMPLETE" for o in bl.orders) and len(bl.trades) == 5
    # TrackedMM trades exactly like the backtest's MarketMaker, and its blotter reconciles with its fills and P&L
    rng = np.random.default_rng(1)
    kw = dict(size=1.0, half=0.5, inv_skew=0.4, kappa=1.0, use_signal=True, pull=True, latency_ms=50)
    base, tr = MarketMaker(**kw), TrackedMM("C", 1.0, 0.0, **kw)
    mid = 1000
    for i in range(4000):
        ts = i * 20
        mid += int(rng.integers(-1, 2)) if rng.random() < 0.3 else 0
        bp, ap = tuple(mid - k for k in range(5)), tuple(mid + 1 + k for k in range(5))
        bq = aq = tuple(float(rng.integers(1, 6)) for _ in range(5))
        move, pb, pa = float(rng.normal()), bool(rng.random() < 0.05), bool(rng.random() < 0.05)
        for mm in (base, tr):
            mm.on_book(ts, bp, bq, ap, aq, move, pb, pa)
        if rng.random() < 0.5:
            agg, q = (1 if rng.random() < 0.5 else -1), float(rng.integers(1, 8))
            for mm in (base, tr):
                mm.on_trade(ts + 5, ap[0] if agg > 0 else bp[0], q, agg)
    assert base.fills == tr.fills and len(tr.fills) > 50
    trades = list(tr.blotter.trades)
    assert [(f["t"], f["side"], f["px"]) for f in trades] == [(f[0], f[1], f[2]) for f in tr.fills[-len(trades):]]
    pos = tr.blotter.position(mid)
    assert abs(pos["qty"] - tr.pos) < 1e-9 and abs(pos["realised"] + pos["unrealised"] - tr.equity_ticks(mid)) < 1e-6
    st_ = {o["status"] for o in tr.blotter.orders}
    assert {"COMPLETE", "CANCELLED"} <= st_ and st_ <= {"PENDING", "OPEN", "COMPLETE", "CANCELLED", "REJECTED"}, st_
    assert sum(o["status"] in LIVE_ST for o in tr.blotter.orders) <= 2
    # Binance book sync (offline): stale events dropped, the event straddling the snapshot applied, gaps resync
    ev = lambda U, u, pu, b: {"U": U, "u": u, "pu": pu, "b": b, "a": []}
    snap = {"lastUpdateId": 105, "bids": [["10", "1"]], "asks": [["11", "1"]]}
    s1 = BinanceSignal("ETHUSDT"); s1.fetch_snapshot = lambda: snap
    s1.on_event(ev(100, 103, 99, [["9", "1"]])); s1.on_event(ev(104, 107, 103, [["10", "2"]]))
    s1.resync()
    assert s1.synced and s1.bids == {10.0: 2.0} and s1.last_u == 107
    s1.on_event(ev(108, 110, 107, [["10", "0"], ["8", "3"]]))
    assert s1.bids == {8.0: 3.0}
    s2 = BinanceSignal("ETHUSDT"); s2.fetch_snapshot = lambda: snap   # snapshot newer than everything buffered
    s2.on_event(ev(100, 103, 99, [])); s2.resync()
    assert not s2.synced
    s2.on_event(ev(104, 107, 103, [["10", "4"]]))
    assert s2.synced and s2.bids[10.0] == 4.0
    s2.fetch_snapshot = lambda: (_ for _ in ()).throw(OSError("offline"))   # the resync thread a gap starts stays offline
    s2.on_event(ev(115, 118, 112, []))
    assert not s2.synced and s2.resyncs == 1
    # Binance signal: band imbalance, 1-hour z-score from >= 4 snapshots, usable only lag_s after each snapshot
    bn = BinanceSignal("ETHUSDT", every_s=30, lag_s=1)
    bn.synced, imbs = True, []
    for i, (bb, aa) in enumerate([(10, 10), (12, 8), (9, 11), (15, 5), (10, 10)]):
        bn.bids, bn.asks = {100.0: 1.0, 99.0: bb, 50.0: 99.0}, {101.0: 1.0, 102.0: aa, 200.0: 99.0}   # 50 / 200: outside 3 %
        bn.step(1000 + 30 * i)
        imbs.append((bb - aa) / (bb + aa + 2))
    v = np.array(imbs)
    assert abs(bn.last_imb - v[-1]) < 1e-12 and abs(bn.z - (v[-1] - v.mean()) / v.std(ddof=1)) < 1e-9
    z4 = (v[3] - v[:4].mean()) / v[:4].std(ddof=1)
    assert abs(bn.value(1120.5) - z4) < 1e-9 and abs(bn.value(1121) - bn.z) < 1e-12
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
        need = {"strategies", "equity", "series", "heat", "trades_heat", "fills", "latency", "params", "last", "tick", "desk"}
        dk = st["desk"]
        assert set(dk["books"]) == {"A", "C", "T", "X"} and len(dk["depth"]["bp"]) == 10 and dk["tape"][-1][1] == 2000.01
        assert dk["books"]["C"]["orders"] and dk["books"]["C"]["orders"][0]["status"] in LIVE_ST
        json.dumps(st, default=float, allow_nan=False)
        # algorithm X live: z = +2 leans 4 bp up, so X bids at the best bid; a seller hits it; maker rebate is earned
        eng2 = Engine(model, "ETHUSDT")
        eng2.xwarm_until, eng2.bn.next_t = 0, 1e18
        eng2.bn.synced, eng2.bn.z, eng2.bn.z_from = True, 2.0, 0.0
        eng2.on_book_msg({"type": "snapshot", "ts": 1000, "data": book})
        o = [r for r in eng2.xex.blotter.orders if r["side"] > 0][0]
        assert o["px"] == eng2.ticks(2000.00) and [r["side"] for r in eng2.xex.blotter.orders] == [1, -1]
        eng2.on_trade_msg({"data": [{"T": 1050, "p": "2000.00", "v": "0.5", "S": "Sell"}]})
        xb = eng2.xex.blotter
        assert xb.n_trades == 1 and xb.pos == 0.1 and o["status"] == "COMPLETE" and list(xb.trades)[0]["fee"] < 0
        sx = eng2.state()
        assert sx["strategies"]["X"]["fills"] == 1 and sx["desk"]["books"]["X"]["n_trades"] == 1
        assert any(e["s"] == "X" and e["k"] == "FILL" for e in sx["log"])
        json.dumps(sx, default=float, allow_nan=False)
        eng3 = Engine(model, "ETHUSDT")                  # warming up: X sends nothing
        eng3.on_book_msg({"type": "snapshot", "ts": 1000, "data": book})
        assert eng3.xex.blotter.n_orders == 0 and eng3.state()["strategies"]["X"]["signal"]["warmup_s_left"] > 0
        assert need <= st.keys(), need - st.keys()
        assert len(st["heat"]) == 2 and st["trades_heat"] == [[1000, 1400, eng.ticks(2000.01), 0.5, 0.0]]
        kinds = [e["k"] for e in st["log"] if e["s"] == "C"]
        assert "PLACE" in kinds and "LIVE" in kinds and kinds.index("PLACE") < kinds.index("LIVE"), st["log"]
    print("selftest ok: live book rebuild, blotter (orders, trades, positions) matches the backtest market maker, "
          "taker state machine, depth heat column, published state")


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
        if eng.xex is not None:
            threading.Thread(target=eng.bn.run, daemon=True).start()   # Binance public depth feed for algorithm X
        serve(a.port)
        stream(eng, a.symbol)
