"""HFT simulation environment: replay real order books and trades with switchable infrastructure.

    python hftsim.py               # every strategy x every infrastructure profile, tuned per profile -> results/hftsim/
    python hftsim.py --quick       # first part of the test day only
    python hftsim.py --selftest

Write a strategy in strategies.py; this file is the exchange. It replays Bybit book snapshots (top 5 levels,
~100 ms) and every trade print (ms), keeps a "shadow" touch that trades update between snapshots, and simulates
your orders with queue position, latency, fees, rate limits and partial fills.

Infrastructure profiles (what an HFT firm buys, as switches):
  retail        50 ms order/cancel latency, back of the queue, Bybit VIP0 fees, 10 msgs/s, decides on book updates only
  hft_firm      10 us latency (co-location), first in the queue at a price level that just formed (it wins the race),
                maker rebate, low taker fee, 2,000 msgs/s, reacts to every trade
  frictionless  zero latency, front of every queue, zero fees: an upper bound, not a realistic setting
The hft_firm numbers are assumptions that stand in for co-location and a market-maker fee tier; they are not
measured, and the queue-race rule is the most optimistic one. Treat results as "what infrastructure is worth".
Research and simulation only: nothing here connects to an exchange.
"""
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "hftsim"
FAR = 10 ** 12


@dataclass
class Profile:
    name: str
    order_latency_ms: float
    cancel_latency_ms: float
    maker_bps: float
    taker_bps: float
    queue: str                 # "back" | "race" | "front"
    race_window_ms: float      # "race": a level younger than this is joined at the front
    max_msgs_per_s: int
    react_to_trades: bool      # strategy gets on_trade callbacks (a slow trader only sees book snapshots)


def profiles(retail_maker, retail_taker):
    return {
        "retail": Profile("retail", 50.0, 50.0, retail_maker, retail_taker, "back", 0.0, 10, False),
        "hft_firm": Profile("hft_firm", 0.01, 0.01, -0.5, 2.0, "race", 100.0, 2000, True),
        "frictionless": Profile("frictionless", 0.0, 0.0, 0.0, 0.0, "front", 0.0, 10 ** 9, True),
    }


@dataclass
class Order:
    oid: int
    side: int                  # +1 buy, -1 sell
    px: int                    # ticks
    qty: float
    filled: float = 0.0
    queue_ahead: float = 0.0
    live: bool = False
    queue_at_join: float = -1.0       # queue ahead when the order went live (what a firm estimates from the feed)

    @property
    def remaining(self):
        return self.qty - self.filled


@dataclass
class Fill:
    ts: int
    side: int
    px: int
    qty: float
    maker: bool
    oid: int = -1


class Strategy:
    """Override what you need. env gives: book (bp, bq, ap, aq in ticks/sizes), position, signal, place/cancel/market."""
    def on_start(self, env): pass
    def on_book(self, env): pass
    def on_trade(self, env, px, qty, aggressor): pass
    def on_fill(self, env, fill): pass


class Exchange:
    def __init__(self, profile, tick, rng_seed=0):
        self.p, self.tick = profile, tick
        self.now, self.seq = 0, 0
        self.bp = self.ap = None
        self.bq = self.aq = None
        self.born = {}                         # (side, price) -> time the level appeared at the touch
        self.orders, self.pending = {}, []     # pending: (act_time, created_seq, kind, payload)
        self.next_oid = 1
        self.pos = 0.0
        self.cash_ticks = 0.0
        self.fees = 0.0                        # quote currency, positive = paid
        self.fills = []
        self.msgs = []
        self.messages = self.rejected = 0
        self.signal = None                     # set by the runner per book event: (move, pull_bid, pull_ask, prob)
        self.aux = None                        # optional extra per-event features (e.g. deep-book imbalance z-scores)
        self.strategy = None

    # ----- the strategy-facing API ------------------------------------------
    def place(self, side, px, qty, post_only=True):
        """Limit order at px (ticks). Becomes live after the order latency. Returns an order id or None (rate cap)."""
        if not self._msg_ok():
            return None
        oid = self.next_oid; self.next_oid += 1
        self.orders[oid] = Order(oid, side, int(px), qty)
        self.pending.append((self.now + self.p.order_latency_ms, self.seq, "place", (oid, post_only)))
        return oid

    def replace(self, oid, px):
        """Amend: move an open order to a new price in one message (the old one stays live until this arrives).
        Returns the new order id, or None if rate-capped (the old order is then left untouched)."""
        old = self.orders.get(oid)
        if old is None or not self._msg_ok():
            return None
        new = self.next_oid; self.next_oid += 1
        self.orders[new] = Order(new, old.side, int(px), old.remaining)
        self.pending.append((self.now + self.p.order_latency_ms, self.seq, "replace", (oid, new)))
        return new

    def cancel(self, oid):
        if oid is None or oid not in self.orders or not self._msg_ok():
            return False
        self.pending.append((self.now + self.p.cancel_latency_ms, self.seq, "cancel", oid))
        return True

    def market(self, side, qty):
        """Immediate-or-cancel against the book once it arrives (after the order latency); pays the taker fee."""
        if not self._msg_ok():
            return False
        self.pending.append((self.now + self.p.order_latency_ms, self.seq, "market", (side, qty)))
        return True

    @property
    def mid(self):
        return 0.5 * (self.bp[0] + self.ap[0])

    def equity(self):
        """Mark-to-mid P&L in quote currency, before fees."""
        return (self.cash_ticks + self.pos * self.mid) * self.tick

    # ----- internals --------------------------------------------------------
    def _msg_ok(self):
        while self.msgs and self.msgs[0] <= self.now - 1000:
            self.msgs.pop(0)
        if len(self.msgs) >= self.p.max_msgs_per_s:
            self.rejected += 1
            return False
        self.msgs.append(self.now); self.messages += 1
        return True

    def _level_qty(self, side, px):
        prices, sizes = (self.bp, self.bq) if side > 0 else (self.ap, self.aq)
        for p, q in zip(prices, sizes):
            if p == px:
                return q
        return None

    def _join_queue(self, o):
        best = self.bp[0] if o.side > 0 else self.ap[0]
        improves = o.px > best if o.side > 0 else o.px < best
        if improves or self.p.queue == "front":
            return 0.0
        q = self._level_qty(o.side, o.px)
        if q is None:
            return float(FAR)                  # behind the visible book: effectively unfillable
        if self.p.queue == "race" and self.now - self.born.get((o.side, o.px), -FAR) <= self.p.race_window_ms:
            return 0.0                         # a level that just formed: the fastest firm arrives first
        return float(q)

    def _activate(self, k, book=False):
        """Apply orders whose latency has passed. Sub-ms actions also beat the next book snapshot, but never a
        trade print in the same millisecond: prints sharing a timestamp are one sweep, matched atomically."""
        if not self.pending:
            return
        keep = []
        fast = lambda lat: lat < 1.0           # sub-ms latency: in effect before the next data message
        for item in self.pending:
            act, created, kind, payload = item
            due = act <= self.now or (book and created < k and fast(self.p.order_latency_ms if kind != "cancel" else self.p.cancel_latency_ms))
            if not due:
                keep.append(item); continue
            if kind == "place":
                oid, post_only = payload
                o = self.orders.get(oid)
                if o is None:
                    continue
                crosses = o.px >= self.ap[0] if o.side > 0 else o.px <= self.bp[0]
                if crosses and post_only:
                    self.orders.pop(oid, None); self.rejected += 1
                    continue
                o.live, o.queue_ahead = True, self._join_queue(o)
                o.queue_at_join = o.queue_ahead
            elif kind == "cancel":
                o = self.orders.pop(payload, None)
            elif kind == "replace":
                old_id, new_id = payload
                old, new = self.orders.pop(old_id, None), self.orders.get(new_id)
                if new is None:
                    continue
                if old is None:                    # old order already filled: the amend fails
                    self.orders.pop(new_id, None)
                    continue
                new.qty = old.remaining
                crosses = new.px >= self.ap[0] if new.side > 0 else new.px <= self.bp[0]
                if crosses:
                    self.orders.pop(new_id, None); self.rejected += 1
                    continue
                new.live, new.queue_ahead = True, self._join_queue(new)
                new.queue_at_join = new.queue_ahead
            elif kind == "market":
                self._take(*payload)
        self.pending = keep

    def _take(self, side, qty):
        prices, sizes = (self.ap, self.aq) if side > 0 else (self.bp, self.bq)
        left = qty
        for p, q in zip(prices, sizes):
            if left <= 0 or p >= FAR or p <= -FAR:
                break
            take = min(left, q)
            if take > 0:
                self._fill(side, p, take, maker=False, oid=-1)
                left -= take
        # anything beyond five visible levels is not filled (conservative)

    def _fill(self, side, px, qty, maker, oid):
        f = Fill(self.now, side, px, qty, maker, oid)
        self.fills.append(f)
        self.pos += side * qty
        self.cash_ticks -= side * px * qty
        bps = self.p.maker_bps if maker else self.p.taker_bps
        self.fees += px * self.tick * qty * bps * 1e-4
        if self.strategy:
            self.strategy.on_fill(self, f)

    def on_trade(self, ts, px, qty, aggressor, k):
        self.now, self.seq = ts, k
        if self.bp is None:
            return
        self._activate(k)
        # shadow touch: trades eat the visible levels between snapshots
        if aggressor > 0:
            while self.ap[0] < px:
                self._pop(-1)
            if self.ap[0] == px:
                self.aq[0] -= qty
                if self.aq[0] <= 0:
                    self._pop(-1)
        else:
            while self.bp[0] > px:
                self._pop(+1)
            if self.bp[0] == px:
                self.bq[0] -= qty
                if self.bq[0] <= 0:
                    self._pop(+1)
        # our resting orders on the side this trade can hit
        side = -aggressor
        for o in list(self.orders.values()):
            if not o.live or o.side != side:
                continue
            if (side > 0 and px < o.px) or (side < 0 and px > o.px):
                q = o.remaining                        # traded through our price
            elif px == o.px:
                take = qty - o.queue_ahead
                o.queue_ahead = max(0.0, o.queue_ahead - qty)
                q = min(o.remaining, take) if take > 0 else 0.0
            else:
                continue
            if q > 0:
                o.filled += q
                if o.remaining <= 1e-12:
                    self.orders.pop(o.oid, None)
                self._fill(side, o.px, q, maker=True, oid=o.oid)
        if self.p.react_to_trades and self.strategy:
            self.strategy.on_trade(self, px, qty, aggressor)

    def _pop(self, side):
        """Top level consumed: the next level becomes the touch (a new level is born there)."""
        if side > 0:
            self.bp, self.bq = self.bp[1:] + [-FAR], self.bq[1:] + [0.0]
            self.born.setdefault((+1, self.bp[0]), self.now)
        else:
            self.ap, self.aq = self.ap[1:] + [FAR], self.aq[1:] + [0.0]
            self.born.setdefault((-1, self.ap[0]), self.now)

    def on_book(self, ts, bp, bq, ap, aq, k):
        self.now, self.seq = ts, k
        first = self.bp is None
        for p in bp[:1]:
            if first or p != self.bp[0]:
                self.born.setdefault((+1, p), ts)
        for p in ap[:1]:
            if first or p != self.ap[0]:
                self.born.setdefault((-1, p), ts)
        self.bp, self.bq, self.ap, self.aq = list(bp), list(bq), list(ap), list(aq)
        if len(self.born) > 50_000:
            self.born = {key: t for key, t in self.born.items() if ts - t < 60_000}
        self._activate(k, book=True)
        for o in list(self.orders.values()):           # snapshot: crossed = filled; cancellations ahead shrink the queue
            if not o.live:
                continue
            if (o.side > 0 and self.ap[0] <= o.px) or (o.side < 0 and self.bp[0] >= o.px):
                q = o.remaining
                self.orders.pop(o.oid, None)
                self._fill(o.side, o.px, q, maker=True, oid=o.oid)
                continue
            lvl = self._level_qty(o.side, o.px)
            best = self.bp[0] if o.side > 0 else self.ap[0]
            if (o.side > 0 and o.px > best) or (o.side < 0 and o.px < best):
                o.queue_ahead = 0.0
            elif lvl is not None:
                o.queue_ahead = min(o.queue_ahead, lvl)
        if self.strategy:
            self.strategy.on_book(self)


def book_imbalance_z(ts, R, windows=(600, 3600)):
    """Causal z-score of (total bid size - total ask size) over every published level (R cols 21, 22), sampled each
    second; an event at time t sees only seconds that had fully completed. Returns (n, len(windows))."""
    import pandas as pd
    x = pd.Series(R[:, 21] - R[:, 22], index=pd.to_datetime(ts, unit="ms"))
    s = x.resample("1s").last().ffill()
    sec = pd.to_datetime((ts // 1000 - 1) * 1000, unit="ms")
    cols = []
    for w in windows:
        r = s.rolling(w, min_periods=w // 4)
        cols.append(((s - r.mean()) / r.std()).reindex(sec).to_numpy())
    return np.column_stack(cols)


def binance_imbalance_z(ts, symbol, day, window="1h", lag_s=1):
    """Cross-venue signal from Binance's public bookDepth archive (USD-M futures, ~30 s snapshots of the size resting
    within +-0.2..5 % of the price): imbalance (bid - ask) / (bid + ask) in the +-2.5 % band (mean of the 2 % and 3 %
    bands) and the +-1 % band, z-scored over `window`, usable `lag_s` after each snapshot. Returns (n, 2); NaN if absent."""
    import zipfile
    import pandas as pd
    f = ROOT / "raw" / "binance" / "bookDepth" / f"{symbol}-bookDepth-{day}.zip"
    if not f.exists():
        return np.full((len(ts), 2), np.nan)
    z = zipfile.ZipFile(f)
    d = pd.read_csv(z.open(z.namelist()[0]))
    d["timestamp"] = pd.to_datetime(d["timestamp"])                    # UTC, like the Bybit timestamps
    p = d.pivot_table(index="timestamp", columns="percentage", values="depth")
    bands = [((p[-2.0] + p[-3.0]) / 2, (p[2.0] + p[3.0]) / 2), (p[-1.0], p[1.0])]
    at = pd.to_datetime(ts, unit="ms")
    cols = []
    for b, a in bands:
        x = (b - a) / (b + a)
        r = x.rolling(window, min_periods=4)
        zz = ((x - r.mean()) / r.std())
        zz.index = zz.index + pd.Timedelta(seconds=lag_s)
        cols.append(zz.reindex(at, method="ffill").to_numpy())
    return np.column_stack(cols)


def replay(ex, strategy, ts, B, trades, signals, eq_every_ms=1000, aux=None):
    """Merge book events and trades in time order (trades at a timestamp before the snapshot that reflects them)."""
    tts, tpx, tqty, tside = trades
    move, pull_b, pull_a, prob = signals
    bp_, bq_, ap_, aq_ = B
    ex.strategy = strategy
    strategy.on_start(ex)
    j, nT, k = 0, len(tts), 0
    eq_t, eq_v, last = [], [], -FAR
    for i in range(len(ts)):
        t = int(ts[i])
        while j < nT and tts[j] <= t:
            k += 1; ex.on_trade(int(tts[j]), int(tpx[j]), float(tqty[j]), int(tside[j]), k); j += 1
        k += 1
        ex.signal = (move[i], pull_b[i], pull_a[i], prob[i] if prob is not None else None)
        ex.aux = aux[i] if aux is not None else None
        ex.on_book(t, bp_[i], bq_[i], ap_[i], aq_[i], k)
        if t - last >= eq_every_ms:
            eq_t.append(t); eq_v.append(ex.equity()); last = t
    return np.array(eq_t), np.array(eq_v)


def report(ex, label, ts, mid, eq_v):
    f = ex.fills
    out = {"strategy": label, "fills": len(f), "maker_fills": sum(x.maker for x in f), "messages": ex.messages,
           "rejected": ex.rejected}
    gross = ex.equity()
    out.update(gross_pnl=gross, fees=ex.fees, net_pnl=gross - ex.fees,
               volume=sum(x.px * x.qty for x in f) * ex.tick, final_position=ex.pos)
    if f:
        ft = np.array([x.ts for x in f]); fs = np.array([x.side for x in f]); fp = np.array([x.px for x in f])
        mk = np.array([x.maker for x in f])
        i0 = np.clip(np.searchsorted(ts, ft, side="right") - 1, 0, len(ts) - 1)
        i10 = np.clip(np.searchsorted(ts, ft + 10_000), 0, len(ts) - 1)
        if mk.any():
            out["maker_spread_capture_ticks"] = float(np.mean((fs * (mid[i0] - fp))[mk]))
            out["maker_markout_10s_ticks"] = float(np.mean((fs * (mid[i10] - fp))[mk]))
    eq = eq_v
    out["max_drawdown"] = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    return out


# ----------------------------------------------------------------------------- experiment
def run_market(name, quick):
    import pandas as pd
    import strategies as S
    from strategy import MARKETS, FEES, load_trades, model_probs, prep, slice_trades, signals
    tr_f, te_f, trd_te, fmt, unit, _ = MARKETS[name]
    trd_tr = trd_te.replace("2026-11-10", "2026-11-09")
    tr, te = np.load(ROOT / "data" / "markets" / tr_f), np.load(ROOT / "data" / "markets" / te_f)
    tick = float(te["tick"])
    R_tr, ts_tr, R_te, ts_te = tr["R"], tr["ts"].astype(np.int64), te["R"], te["ts"].astype(np.int64)
    if quick:
        R_te, ts_te = R_te[:150_000], ts_te[:150_000]
    tail0, p_tail, p_test, class_move, info = model_probs(R_tr, R_te, tick)
    info.pop("QM", None)
    Dtail, Dtest = prep(R_tr[tail0:], ts_tr[tail0:], tick), prep(R_te, ts_te, tick)
    T_tail = slice_trades(load_trades(ROOT / "raw" / "trades" / trd_tr, fmt, tick), Dtail["ts"][0], Dtail["ts"][-1])
    T_test = slice_trades(load_trades(ROOT / "raw" / "trades" / trd_te, fmt, tick), Dtest["ts"][0], Dtest["ts"][-1])
    bsym = name.split()[0]                     # Binance USD-M symbol for the cross-venue signal
    Ztail = np.column_stack([book_imbalance_z(ts_tr, R_tr), binance_imbalance_z(ts_tr, bsym, "2026-11-09")])[tail0:]
    Ztest = np.column_stack([book_imbalance_z(ts_te, R_te), binance_imbalance_z(ts_te, bsym, "2026-11-10")])
    print(f"  aux signals: Bybit 200-level z (10 min, 1 h), Binance +-2.5 % and +-1 % z (1 h); "
          f"Binance coverage {np.isfinite(Ztest[:, 2]).mean():.0%} of test events")
    as_lists = lambda D: tuple([list(r) for r in side] for side in D["B"])
    Btail, Btest = as_lists(Dtail), as_lists(Dtest)

    def sig(prob, q):
        conf = prob.max(1)[prob.argmax(1) != 1]
        return (*signals(prob, class_move, np.quantile(conf, q) if q else 9.0), prob)

    def one(D, B, T, prob, prof, strat_cls, kw, q, label, aux=None):
        ex = Exchange(prof, tick)
        st = strat_cls(size=unit, **kw)
        eq_t, eq_v = replay(ex, st, D["ts"], B, T, sig(prob, q), aux=aux)
        r = report(ex, label, D["ts"], D["mid"], eq_v)
        r.update(profile=prof.name, market=name, q=q, **{k: v for k, v in kw.items()})
        return r, (eq_t, eq_v)

    rm, rt = FEES[name]
    rows, tunes, curves, hft_kw = [], [], {}, {}
    for prof in profiles(rm, rt).values():
        t0 = time.time()
        grid = []
        for half in (0.5, 1.5, 2.5):
            for inv in (0.2, 0.6):
                grid.append(("A", S.QuotingMM, dict(half=half, inv_skew=inv), None))
                for kappa in (1.0, 3.0):
                    for q in (0.8, 0.95):
                        grid.append(("C", S.NSQMM, dict(half=half, inv_skew=inv, kappa=kappa, guard=prof.react_to_trades), q))
        for inv in (0.6, 2.0):                     # Q grid fixed from the day-1 per-fill diagnosis, same for every market
            for kappa in (0.0, 1.0, 3.0):
                for tau in (0.1, 0.15, 0.25, 0.35):
                    for imb in (0.0, 0.5):
                        grid.append(("Q", S.QueueAwareMM, dict(half=0.5, inv_skew=inv, kappa=kappa, tau=tau, imb_min=imb,
                                                               max_ahead=0.5, guard=prof.react_to_trades), None))
        for half in (0.25, 0.5, 1.0):              # O: hftbacktest imbalance market maker, prices in bp of the mid
            for skew in (0.02, 0.1):
                for c1, zcol in ((0.0, 0), (0.5, 0), (0.5, 1), (2.0, 0), (2.0, 1)):   # O: Bybit's own 200-level book
                    grid.append(("O", S.ImbalanceMM, dict(half_bp=half, c1_bp=c1, skew_bp=skew, zcol=zcol), None))
                for c1, zcol in ((0.5, 2), (2.0, 2), (5.0, 2), (2.0, 3), (5.0, 3)):   # X: Binance +-2.5 % / +-1 % bands
                    grid.append(("X", S.ImbalanceMM, dict(half_bp=half, c1_bp=c1, skew_bp=skew, zcol=zcol), None))
        for q in (0.9, 0.97, 0.995):
            for hold in (10, 30):
                grid.append(("T", S.SignalTaker, dict(hold=hold), q))
        res = []
        for key, cls, kw, q in grid:
            r, _ = one(Dtail, Btail, T_tail, p_tail, prof, cls, kw, q, key, aux=Ztail)
            res.append(r)
        tune = pd.DataFrame(res); tunes.append(tune)
        for key, cls in (("A", S.QuotingMM), ("C", S.NSQMM), ("Q", S.QueueAwareMM), ("O", S.ImbalanceMM), ("X", S.ImbalanceMM),
                         ("T", S.SignalTaker)):
            best = tune[tune.strategy == key].sort_values("net_pnl").iloc[-1]   # tuned on net P&L: fees are part of the profile
            kw = {c: best[c] for c in ("half", "inv_skew", "kappa", "guard", "hold", "tau", "imb_min", "max_ahead",
                                       "half_bp", "c1_bp", "skew_bp", "zcol") if c in best.index and not pd.isna(best[c])}
            for c in ("hold", "zcol"):
                if c in kw: kw[c] = int(kw[c])
            if "guard" in kw: kw["guard"] = bool(kw["guard"])
            r, c = one(Dtest, Btest, T_test, p_test, prof, cls, kw, None if pd.isna(best["q"]) else best["q"], NAMES[key], aux=Ztest)
            rows.append(r); curves[(prof.name, key)] = c
            if key in ("Q", "O", "X") and prof.name == "hft_firm":
                hft_kw[key] = kw
            print(f"  {prof.name:12s} {NAMES[key]:28s} fills {r['fills']:7,d}  gross {r['gross_pnl']:9.2f}  "
                  f"fees {r['fees']:9.2f}  net {r['net_pnl']:9.2f}")
        print(f"  ({prof.name}: tuned {len(grid)} settings and tested in {time.time() - t0:.0f}s)")
    # robustness: the Q and O settings chosen for hft_firm, on the test day, with the profile's optimistic assumptions removed
    from dataclasses import replace
    hft = profiles(rm, rt)["hft_firm"]
    for label, prof in (("hft_firm, no maker rebate", replace(hft, maker_bps=0.0)),
                        ("hft_firm, 10 ms queue race", replace(hft, race_window_ms=10.0)),
                        ("hft_firm, back of every queue", replace(hft, queue="back")),
                        ("hft_firm, 1 ms latency", replace(hft, order_latency_ms=1.0, cancel_latency_ms=1.0))):
        for key, cls in (("Q", S.QueueAwareMM), ("O", S.ImbalanceMM), ("X", S.ImbalanceMM)):
            r, _ = one(Dtest, Btest, T_test, p_test, prof, cls, hft_kw[key], None, NAMES[key], aux=Ztest)
            r["profile"] = label; rows.append(r)
            print(f"  {label:30s} {key}  fills {r['fills']:7,d}  gross {r['gross_pnl']:9.2f}  fees {r['fees']:9.2f}  net {r['net_pnl']:9.2f}")
    return rows, tunes, curves


NAMES = {"A": "A  market maker, no model", "C": "C  NSQ-MM", "Q": "Q  NSQ-MM-Q (queue-aware)",
         "O": "O  imbalance MM (hftbacktest port)", "X": "X  imbalance MM, Binance signal", "T": "T  signal taker"}


def plot(rows, OUTP):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    d = pd.DataFrame(rows)
    markets = list(dict.fromkeys(d.market))
    fig, axs = plt.subplots(1, len(markets), figsize=(6 * len(markets), 4), squeeze=False)
    cols = dict(zip(NAMES.values(), ("#9aa5a0", "#00b36b", "#e0a100", "#8b5cd6", "#d6336c", "#2a78d6")))
    profs = ["retail", "hft_firm", "frictionless"]
    for ax, m in zip(axs[0], markets):
        x = np.arange(len(profs)); w = 0.13
        for i, (s, col) in enumerate(cols.items()):
            v = [d[(d.market == m) & (d.profile == p) & (d.strategy == s)].net_pnl.iloc[0] for p in profs]
            ax.bar(x + (i - 2.5) * w, v, w, color=col, label=s)
        ax.axhline(0, color="#777", lw=1)
        ax.set_xticks(x, ["retail", "HFT firm", "frictionless"])
        ax.set(title=m, ylabel="net P&L after fees (USDT)")
        ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(OUTP / "profiles_net_pnl.png", dpi=120); plt.close()


def selftest():
    import strategies as S

    class Fixed(Strategy):
        def __init__(self): self.done = False
        def on_book(self, env):
            if not self.done:
                env.place(+1, 100, 1.0); self.done = True
    bp, bq, ap, aq = [100, 99, 98, 97, 96], [5.0] * 5, [101, 102, 103, 104, 105], [5.0] * 5
    P = profiles(2.0, 5.5)
    # retail: 50 ms latency, back of a 5-lot queue
    ex = Exchange(P["retail"], 0.01); ex.strategy = Fixed()
    ex.on_book(0, bp, bq, ap, aq, 1)
    ex.on_trade(10, 100, 3.0, -1, 2)              # before the order is live: nothing
    assert not ex.fills and not ex.orders[1].live
    ex.on_book(60, bp, bq, ap, aq, 3)              # live now, 5 ahead
    assert ex.orders[1].queue_ahead == 5.0
    ex.on_trade(70, 100, 5.5, -1, 4)               # 5 ahead consumed, 0.5 of ours filled (partial)
    assert abs(ex.fills[0].qty - 0.5) < 1e-12 and ex.orders[1].remaining == 0.5
    # HFT firm: a level that just formed is joined at the front; fills and maker rebate
    ex = Exchange(P["hft_firm"], 0.01); ex.strategy = Fixed()
    ex.on_book(0, bp, bq, ap, aq, 1)               # level 100 born at t=0, order placed
    ex.on_trade(1, 100, 0.4, -1, 2)                # order live before this message; front of queue
    assert ex.fills and abs(ex.fills[0].qty - 0.4) < 1e-12 and ex.fees < 0
    # shadow touch: a buy sweep removes the ask level between snapshots
    ex.on_trade(2, 102, 3.0, +1, 3)                # a print at 102 means 101 is gone; 3 of 5 at 102 traded
    assert ex.ap[0] == 102 and abs(ex.aq[0] - 2.0) < 1e-12
    # taker: market buy walks the shadow book and pays the taker fee
    ex2 = Exchange(P["hft_firm"], 0.01)
    ex2.on_book(0, bp, bq, ap, aq, 1); ex2.market(+1, 7.0); ex2.on_book(1, bp, bq, ap, aq, 2)
    assert [(f.px, f.qty) for f in ex2.fills] == [(101, 5.0), (102, 2.0)] and ex2.fees > 0
    # amend: one message, the old order stays live until the new one arrives, never two live orders
    ex4 = Exchange(P["retail"], 0.01); ex4.on_book(0, bp, bq, ap, aq, 1)
    oid = ex4.place(+1, 100, 1.0); ex4.on_book(60, bp, bq, ap, aq, 2)
    new = ex4.replace(oid, 99); ex4.on_book(80, bp, bq, ap, aq, 3)
    assert oid in ex4.orders and ex4.orders[oid].live and new in ex4.orders and not ex4.orders[new].live
    ex4.on_book(120, bp, bq, ap, aq, 4)
    assert oid not in ex4.orders and ex4.orders[new].live and ex4.orders[new].px == 99 and ex4.messages == 2
    # strategies run end to end on synthetic data
    # a sweep is atomic: prints sharing a millisecond cannot be dodged by a cancel sent between them
    ex5 = Exchange(P["hft_firm"], 0.01); ex5.on_book(0, bp, bq, ap, aq, 1)
    ex5.on_trade(5, 100, 0.0, -1, 2); oid = ex5.place(+1, 100, 1.0); ex5.cancel(oid)
    ex5.on_trade(5, 100, 4.0, -1, 3)                                    # same ms: the place and the cancel wait
    assert not ex5.fills and oid in ex5.orders and not ex5.orders[oid].live
    # imbalance MM: mid 10,000.5 ticks = ~1 tick per bp; alpha +-2 with c1 = 1 bp shifts the reservation price by +-2 ticks
    ex6 = Exchange(P["hft_firm"], 0.01)
    ex6.on_book(0, [10000 - i for i in range(5)], [5.0] * 5, [10001 + i for i in range(5)], [5.0] * 5, 1)
    om = S.ImbalanceMM(size=1.0, half_bp=0.4, c1_bp=1.0, skew_bp=0.0)
    ex6.aux = (2.0, 0.0); assert om.targets(ex6) == {+1: 10000, -1: 10003}
    ex6.aux = (-2.0, 0.0); assert om.targets(ex6) == {+1: 9998, -1: 10001}
    for cls, kw in ((S.QuotingMM, {}), (S.NSQMM, {"guard": True}), (S.QueueAwareMM, {"tau": 0.5, "guard": True}),
                    (S.ImbalanceMM, {}), (S.SignalTaker, {})):
        ex3 = Exchange(P["hft_firm"], 0.01)
        n = 50
        prob = np.tile([0.1, 0.2, 0.7], (n, 1))
        replay(ex3, cls(size=1.0, **kw), np.arange(n) * 100, ([bp] * n, [bq] * n, [ap] * n, [aq] * n),
               (np.array([150, 250]), np.array([100, 101]), np.array([10.0, 10.0]), np.array([-1, 1])),
               (np.full(n, 0.5), np.zeros(n, bool), np.zeros(n, bool), prob))
    print("selftest ok: latency, queue back/race/front, partial fills, shadow touch, taker walk, fees, strategies")


if __name__ == "__main__":
    import pandas as pd
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--quick", action="store_true")
    ap_.add_argument("--selftest", action="store_true")
    a = ap_.parse_args()
    if a.selftest:
        selftest()
    else:
        from strategy import MARKETS
        OUT.mkdir(parents=True, exist_ok=True)
        allrows, alltunes = [], []
        for name in MARKETS:
            print(name)
            rows, tunes, _ = run_market(name, a.quick)
            allrows += rows; alltunes += tunes
            pd.DataFrame(allrows).to_csv(OUT / "summary.csv", index=False)
            pd.concat(alltunes).to_csv(OUT / "tuning.csv", index=False)
        plot(allrows, OUT)
        print("wrote", OUT)
