"""NSQ-MM: prediction-driven, inventory-aware market making, with an event-driven backtester.

    python strategy.py                 # backtest on every market with book + trade ticks, writes results/trading/
    python strategy.py --selftest      # offline check of the fill / queue / PnL logic

The algorithm (one class, MarketMaker, shared by the backtest and the live paper trader):
  fair  = mid + kappa * E[move]          E[move] from the TinyML model: sum_c p(c) * mean move of class c (ticks)
              - inv_skew * position       inventory control (Avellaneda-Stoikov reservation-price idea)
  quote bid = floor(fair - half), ask = ceil(fair + half), post-only, one order per side
  pull  the bid when the model confidently calls DOWN, the ask when it confidently calls UP
        (confidence threshold certified on calibration data; protects against adverse selection)
  risk  position cap, order-rate cap, keep queue priority when the price is unchanged

Fill model (the part that makes a market-making backtest honest):
  * an order joining a price level goes to the BACK of the visible queue at that level
  * it fills only when aggressive trades at that price exceed the quantity ahead of it,
    or when trades go through its price; cancellations ahead shrink the queue
  * orders and cancels take effect only after the latency (default 50 ms)
  * fees are applied per fill; post-fill markouts measure adverse selection
Research and paper trading only: nothing here sends orders to any exchange.
"""
import argparse
import gzip
import math
import time
from pathlib import Path

import numpy as np

import tinyml as tm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "trading"
H = 10                                    # prediction horizon, book events

# market -> (book npz train day, book npz test day, trades file of the test day, trade format, unit size, maker fee bps)
MARKETS = {
    "ETHUSDT perpetual (Bybit)": ("bybit_linear_ETHUSDT_2026-11-09.npz", "bybit_linear_ETHUSDT_2026-11-10.npz",
                                  "ETHUSDT_linear_2026-11-10.csv.gz", "linear", 0.1, 2.0),
    "BTCUSDT spot (Bybit)": ("bybit_spot_BTCUSDT_2026-11-09.npz", "bybit_spot_BTCUSDT_2026-11-10.npz",
                             "BTCUSDT_spot_2026-11-10.csv.gz", "spot", 0.002, 10.0),
}


# ----------------------------------------------------------------------------- the algorithm
class MarketMaker:
    """Event-driven quoting logic + simulated exchange-side fills. Prices are integer ticks."""

    def __init__(self, size=1.0, qmax=5, half=0.5, inv_skew=0.4, kappa=1.0, use_signal=True, pull=True,
                 latency_ms=50, max_msgs_per_s=10):
        self.size, self.qmax, self.half, self.inv_skew = size, qmax, half, inv_skew
        self.kappa, self.use_signal, self.pull = kappa, use_signal, pull
        self.lat, self.max_msgs = latency_ms, max_msgs_per_s
        self.pos = 0.0                      # base units (e.g. ETH)
        self.cash_ticks = 0.0               # quote cash in tick units x base units
        self.fills = []                     # (ts, side +1 buy / -1 sell, price_ticks, qty)
        self.live = {+1: None, -1: None}    # side -> [price, queue_ahead]
        self.pending = {+1: [], -1: []}     # side -> [(activate_ts, price or None)]
        self.target = {+1: None, -1: None}
        self.msg_ts = []
        self.book = None
        self.messages = 0

    # ----- exchange side ----------------------------------------------------
    def _queue_at(self, side, px):
        bp, bq, ap, aq = self.book
        if side == +1:
            if px > bp[0]:
                return 0.0
            for i in range(5):
                if bp[i] == px:
                    return float(bq[i])
            return 1e18                     # deeper than the visible book: effectively never filled
        if px < ap[0]:
            return 0.0
        for i in range(5):
            if ap[i] == px:
                return float(aq[i])
        return 1e18

    def _activate(self, ts):
        for side in (+1, -1):
            q = self.pending[side]
            while q and q[0][0] <= ts:
                _, px = q.pop(0)
                if px is None:
                    self.live[side] = None
                    continue
                bp, _, ap, _ = self.book
                if (side == +1 and px >= ap[0]) or (side == -1 and px <= bp[0]):
                    self.live[side] = None  # would cross: post-only order rejected
                    continue
                self.live[side] = [px, self._queue_at(side, px)]

    def _fill(self, ts, side, px):
        self.fills.append((ts, side, px, self.size))
        self.pos += side * self.size
        self.cash_ticks -= side * px * self.size
        self.live[side] = None
        self.target[side] = None            # re-quote on the next decision

    def on_trade(self, ts, px, qty, aggressor):
        """aggressor +1 = buyer lifted asks (can fill our ask), -1 = seller hit bids (can fill our bid)."""
        if self.book is None:
            return
        self._activate(ts)
        side = -aggressor                    # our resting side that this trade can hit
        o = self.live[side]
        if o is None:
            return
        opx = o[0]
        if (side == +1 and px < opx) or (side == -1 and px > opx):
            self._fill(ts, side, opx)        # traded through our price
        elif px == opx:
            o[1] -= qty
            if o[1] < 0:
                self._fill(ts, side, opx)

    # ----- decision side ----------------------------------------------------
    def on_book(self, ts, bp, bq, ap, aq, move=0.0, pull_bid=False, pull_ask=False):
        self.book = (bp, bq, ap, aq)
        self._activate(ts)
        for side in (+1, -1):                # cancellations ahead shrink our queue; crossed book = filled
            o = self.live[side]
            if o is None:
                continue
            if side == +1 and ap[0] <= o[0] or side == -1 and bp[0] >= o[0]:
                self._fill(ts, side, o[0])
                continue
            o[1] = min(o[1], self._queue_at(side, o[0]))
        mid = 0.5 * (bp[0] + ap[0])
        fair = mid - self.inv_skew * self.pos / self.size
        if self.use_signal:
            fair += self.kappa * move
        want = {+1: math.floor(fair - self.half), -1: math.ceil(fair + self.half)}
        want[+1] = min(want[+1], ap[0] - 1)
        want[-1] = max(want[-1], bp[0] + 1)
        if self.pos >= self.qmax * self.size or (self.pull and pull_bid):
            want[+1] = None
        if self.pos <= -self.qmax * self.size or (self.pull and pull_ask):
            want[-1] = None
        for side in (+1, -1):
            if want[side] == self.target[side]:
                continue                     # unchanged: keep queue priority, send nothing
            while self.msg_ts and self.msg_ts[0] <= ts - 1000:
                self.msg_ts.pop(0)
            if len(self.msg_ts) >= self.max_msgs:
                continue                     # order-rate cap
            self.msg_ts.append(ts)
            self.messages += 1
            self.target[side] = want[side]
            self.pending[side].append((ts + self.lat, want[side]))

    def equity_ticks(self, mid):
        return self.cash_ticks + self.pos * mid


# ----------------------------------------------------------------------------- data + signals
def load_trades(path, fmt, tick):
    import pandas as pd
    if fmt == "linear":
        d = pd.read_csv(path, usecols=["timestamp", "side", "size", "price"])
        ts = np.round(d.timestamp.to_numpy() * 1000).astype(np.int64)
        side = np.where(d.side.to_numpy() == "Buy", 1, -1)
        qty, px = d["size"].to_numpy(float), d.price.to_numpy(float)
    else:
        d = pd.read_csv(path, usecols=["timestamp", "price", "volume", "side"])
        ts = d.timestamp.to_numpy().astype(np.int64)
        side = np.where(d.side.str.lower().to_numpy() == "buy", 1, -1)
        qty, px = d.volume.to_numpy(float), d.price.to_numpy(float)
    pt = np.round(px / tick).astype(np.int64)
    o = np.lexsort((side, pt, ts))
    ts, pt, side, qty = ts[o], pt[o], side[o], qty[o]
    key = np.r_[True, (ts[1:] != ts[:-1]) | (pt[1:] != pt[:-1]) | (side[1:] != side[:-1])]
    g = np.cumsum(key) - 1                   # aggregate prints with the same time, price and side
    return ts[key], pt[key], np.bincount(g, weights=qty), side[key]


def model_probs(train_R, test_R, tick):
    """Train the integer-only student on the first 85% of the train day. Returns class probabilities for the
    train day's untouched tail (used only to tune strategy parameters) and for the whole test day."""
    X_all = tm.run_features(train_R, tick)
    m = 0.5 * (train_R[:, 0] + train_R[:, 10]) / tick
    d = np.round(tm.future_move(m, H), 9)
    keep = (np.arange(len(train_R)) >= tm.W) & np.isfinite(d)
    rows = np.flatnonzero(keep)
    Xa, d = X_all[keep], d[keep]
    n = len(Xa); cut = int(0.85 * n)
    alpha = float(np.quantile(np.abs(d[:cut]), 1 / 3))
    y = 1 + (d > alpha).astype(int) - (d < -alpha).astype(int)
    mu, sd = Xa[:cut].mean(0), Xa[:cut].std(0) + 1e-9
    es = slice(cut, cut + (n - cut) // 2)
    ws = tm.train_mlp((Xa[:cut] - mu) / sd, y[:cut], (Xa[es] - mu) / sd, y[es])
    QM = tm.quantize_mlp(ws, mu, sd, Xa[:cut])
    class_move = np.array([d[:cut][y[:cut] == c].mean() for c in range(3)])
    tail0 = rows[cut]
    prob = lambda X: tm.softmax(tm.qmlp_forward(QM, X)[0] * QM["out_scale"])
    p_tail = prob(X_all[tail0:])
    p_test = prob(tm.run_features(test_R, tick))
    info = {"alpha_ticks": alpha, "class_move": class_move.round(3).tolist(), "train_events": cut,
            "tail_events": len(train_R) - tail0, "bytes": sum(v for k, v in tm.model_bytes(QM).items() if k != "macs")}
    info["QM"] = QM
    return tail0, p_tail, p_test, class_move, info


def signals(prob, class_move, tau):
    """Expected move (ticks) and quote-pull flags from class probabilities. tau = confidence needed to pull."""
    pred, conf = prob.argmax(1), prob.max(1)
    return prob @ class_move, (pred == 0) & (conf >= tau), (pred == 2) & (conf >= tau)


def run_taker(ts, bp, ap, prob, tau, hold, lat_ms, size):
    """Directional taker: on a confident up/down call, buy at the ask / sell at the bid once the order arrives
    (after latency), hold `hold` book events, exit by crossing back. One position at a time. Prices in ticks."""
    pred, conf = prob.argmax(1), prob.max(1)
    go = np.flatnonzero((pred != 1) & (conf >= tau))
    n, trades, free_at = len(ts), [], -1
    for i in go:
        if i < free_at:
            continue
        k = np.searchsorted(ts, ts[i] + lat_ms)
        e = k + hold
        x = np.searchsorted(ts, ts[min(e, n - 1)] + lat_ms) if e < n else n
        if x >= n:
            break
        s = 1 if pred[i] == 2 else -1
        entry = ap[k] if s > 0 else bp[k]
        exitp = bp[x] if s > 0 else ap[x]
        trades.append((ts[k], s, entry, exitp))
        free_at = x + 1
    t = np.array(trades, dtype=float).reshape(-1, 4)
    pnl_ticks = (t[:, 1] * (t[:, 3] - t[:, 2]) * size) if len(t) else np.zeros(0)
    notional = ((t[:, 2] + t[:, 3]) * size).sum() if len(t) else 0.0
    return t, pnl_ticks, notional


# ----------------------------------------------------------------------------- backtest
def run(book_ts, B, trades, mm, move, pull_bid, pull_ask, eq_every_ms=1000):
    """Replay book events and trades in time order (trades at a timestamp before the book snapshot that reflects them)."""
    tts, tpx, tqty, tside = trades
    j, nT = 0, len(tts)
    bp_, bq_, ap_, aq_ = B
    eq_t, eq_v, last = [], [], -10**18
    for i in range(len(book_ts)):
        ts = book_ts[i]
        while j < nT and tts[j] <= ts:
            mm.on_trade(tts[j], tpx[j], tqty[j], tside[j]); j += 1
        mm.on_book(ts, bp_[i], bq_[i], ap_[i], aq_[i], move[i], pull_bid[i], pull_ask[i])
        if ts - last >= eq_every_ms:
            eq_t.append(ts); eq_v.append(mm.equity_ticks(0.5 * (bp_[i][0] + ap_[i][0]))); last = ts
    return np.array(eq_t), np.array(eq_v)


def evaluate(name, mm, book_ts, mid_ticks, tick, eq_t, eq_v, fee_bps):
    """PnL decomposition, fees, adverse selection (post-fill markouts), risk."""
    f = np.array(mm.fills, dtype=float).reshape(-1, 4)
    out = {"strategy": name, "fills": len(f), "messages": mm.messages}
    end_mid = mid_ticks[-1]
    gross_ticks = mm.cash_ticks + mm.pos * end_mid
    px = tick                                                        # 1 tick in quote currency
    notional = (f[:, 2] * f[:, 3]).sum() * px if len(f) else 0.0
    out["gross_pnl"] = gross_ticks * px
    for label, bps in (("net_vip0", fee_bps), ("net_zero_fee", 0.0), ("net_rebate_0.5bp", -0.5)):
        out[label] = gross_ticks * px - notional * bps * 1e-4
    out["volume_quote"] = notional
    if len(f):
        idx0 = np.clip(np.searchsorted(book_ts, f[:, 0], side="right") - 1, 0, len(book_ts) - 1)
        out["spread_capture_ticks"] = float(np.mean(f[:, 1] * (mid_ticks[idx0] - f[:, 2])))
        for hs in (1, 10, 60):
            k = np.clip(np.searchsorted(book_ts, f[:, 0] + hs * 1000), 0, len(book_ts) - 1)
            out[f"markout_{hs}s_ticks"] = float(np.mean(f[:, 1] * (mid_ticks[k] - f[:, 2])))
    eq = eq_v * px
    dd = eq - np.maximum.accumulate(eq) if len(eq) else np.array([0.0])
    out["max_drawdown"] = float(dd.min())
    r = np.diff(eq)
    out["sharpe_daily"] = float(r.mean() / r.std() * np.sqrt(len(r))) if len(r) > 2 and r.std() > 0 else float("nan")
    out["final_position"] = mm.pos
    return out


FEES = {  # Bybit VIP0 published rates in basis points (maker, taker); check the current schedule before relying on them
    "ETHUSDT perpetual (Bybit)": (2.0, 5.5),
    "BTCUSDT spot (Bybit)": (10.0, 10.0),
}


def prep(R, ts, tick):
    tq = lambda a: np.round(a / tick).astype(np.int64)
    bp, ap = tq(R[:, 0:5]), tq(R[:, 10:15])
    B = (list(map(tuple, bp.tolist())), list(map(tuple, R[:, 5:10].tolist())),
         list(map(tuple, ap.tolist())), list(map(tuple, R[:, 15:20].tolist())))
    return {"ts": ts, "B": B, "mid": 0.5 * (bp[:, 0] + ap[:, 0]), "bp1": bp[:, 0], "ap1": ap[:, 0]}


def slice_trades(trades, t0, t1):
    m = (trades[0] >= t0) & (trades[0] <= t1)
    return tuple(a[m] for a in trades)


def mm_run(D, trades, prob, class_move, tick, unit, fee, lat, label, half, inv, kappa=0.0, q=None):
    if q is not None or kappa > 0:
        conf = prob.max(1)[prob.argmax(1) != 1]
        tau = np.quantile(conf, q) if q is not None else 9.0
        move, pb, pa = signals(prob, class_move, tau)
    else:
        n = len(D["ts"]); move, pb, pa = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    mm = MarketMaker(size=unit, half=half, inv_skew=inv, kappa=kappa, use_signal=kappa > 0, pull=q is not None, latency_ms=lat)
    eq_t, eq_v = run(D["ts"], D["B"], trades, mm, move, pb, pa)
    r = evaluate(label, mm, D["ts"], D["mid"], tick, eq_t, eq_v, fee)
    r.update(half=half, inv_skew=inv, kappa=kappa, pull_quantile=q, latency_ms=lat)
    return r, (eq_t, eq_v * tick)


def taker_run(D, prob, tick, unit, taker_bps, lat, label, q, hold):
    conf = prob.max(1)[prob.argmax(1) != 1]
    tau = np.quantile(conf, q)
    t, pnl, notional = run_taker(D["ts"], D["bp1"], D["ap1"], prob, tau, hold, lat, unit)
    gross = pnl.sum() * tick
    fees = notional * tick * taker_bps * 1e-4
    r = {"strategy": label, "fills": 2 * len(t), "gross_pnl": gross, "volume_quote": notional * tick,
         "net_zero_fee": gross, "net_vip0": gross - fees, "net_rebate_0.5bp": gross - fees,
         "hit_rate": float((pnl > 0).mean()) if len(pnl) else float("nan"),
         "avg_gross_ticks_per_trade": float(pnl.mean() / unit) if len(pnl) else float("nan"),
         "pull_quantile": q, "hold_events": hold, "latency_ms": lat}
    eq = np.cumsum(pnl) * tick
    r["max_drawdown"] = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    return r, (t[:, 0] if len(t) else np.zeros(0), eq)


def backtest_market(name, quick=False):
    """Tune every strategy on the train day's untouched tail, then test each chosen setting once on the next day."""
    import pandas as pd
    tr_f, te_f, trd_te, fmt, unit, _ = MARKETS[name]
    trd_tr = trd_te.replace("2026-11-10", "2026-11-09")
    maker_bps, taker_bps = FEES[name]
    tr, te = np.load(ROOT / "data" / "markets" / tr_f), np.load(ROOT / "data" / "markets" / te_f)
    tick = float(te["tick"])
    R_tr, ts_tr, R_te, ts_te = tr["R"], tr["ts"].astype(np.int64), te["R"], te["ts"].astype(np.int64)
    if quick:
        R_te, ts_te = R_te[:150_000], ts_te[:150_000]
    t0 = time.time()
    tail0, p_tail, p_test, class_move, info = model_probs(R_tr, R_te, tick)
    print(f"{name}: model {info['bytes']} B trained on {info['train_events']:,} events; tuning on {info['tail_events']:,}-event tail ({time.time() - t0:.0f}s)")
    Dtail, Dtest = prep(R_tr[tail0:], ts_tr[tail0:], tick), prep(R_te, ts_te, tick)
    T_tail = slice_trades(load_trades(ROOT / "raw" / "trades" / trd_tr, fmt, tick), Dtail["ts"][0], Dtail["ts"][-1])
    T_test = slice_trades(load_trades(ROOT / "raw" / "trades" / trd_te, fmt, tick), Dtest["ts"][0], Dtest["ts"][-1])

    # ---- tuning on the tail (objective: gross P&L, before fees, the same for every strategy)
    t1 = time.time()
    tune = []
    for half in (0.5, 1.5, 2.5):
        for inv in (0.2, 0.6):
            tune.append(mm_run(Dtail, T_tail, p_tail, class_move, tick, unit, maker_bps, 50, "A", half, inv)[0])
            for kappa in (1.0, 3.0):
                for q in (0.6, 0.8, 0.95):
                    tune.append(mm_run(Dtail, T_tail, p_tail, class_move, tick, unit, maker_bps, 50, "C", half, inv, kappa, q)[0])
    for q in (0.9, 0.97, 0.995):
        for hold in (10, 30):
            tune.append(taker_run(Dtail, p_tail, tick, unit, taker_bps, 50, "T", q, hold)[0])
    tune = pd.DataFrame(tune).assign(market=name)
    best = {k: tune[tune.strategy == k].sort_values("gross_pnl").iloc[-1] for k in ("A", "C", "T")}
    print(f"  tuned {len(tune)} settings on the tail in {time.time() - t1:.0f}s: "
          + "; ".join(f"{k} gross {v.gross_pnl:.2f}" for k, v in best.items()))

    # ---- test day, each chosen setting run once
    A, C, T = best["A"], best["C"], best["T"]
    rows, curves = [], {}

    def add(r_c):
        r, c = r_c
        r["market"] = name; rows.append(r); curves[r["strategy"]] = c
        print(f"  {r['strategy']:46s} fills {r['fills']:7,d}  gross {r['gross_pnl']:9.2f}  net VIP0 {r['net_vip0']:10.2f}")
    add(mm_run(Dtest, T_test, p_test, class_move, tick, unit, maker_bps, 50, "A  market maker, no model", A.half, A.inv_skew))
    add(mm_run(Dtest, T_test, p_test, class_move, tick, unit, maker_bps, 50, "C  NSQ-MM (model skew + pulling)",
               C.half, C.inv_skew, C.kappa, C.pull_quantile))
    add(taker_run(Dtest, p_test, tick, unit, taker_bps, 50, "T  TinyML signal taker", T.pull_quantile, int(T.hold_events)))
    for L in (5, 200, 1000):
        add(mm_run(Dtest, T_test, p_test, class_move, tick, unit, maker_bps, L, f"C  NSQ-MM, latency {L} ms",
                   C.half, C.inv_skew, C.kappa, C.pull_quantile))
        add(taker_run(Dtest, p_test, tick, unit, taker_bps, L, f"T  signal taker, latency {L} ms", T.pull_quantile, int(T.hold_events)))
    for r in rows:
        r.update(unit=unit, maker_fee_bps=maker_bps, taker_fee_bps=taker_bps)
    clean = lambda v: None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
    info.update(tuned={k: {c: clean(v[c]) for c in ("half", "inv_skew", "kappa", "pull_quantile", "hold_events") if c in v.index}
                       for k, v in best.items()}, trades_test=len(T_test[0]), book_events_test=len(R_te))
    # save exactly what was tested, for the live paper trader
    QM = info.pop("QM")
    conf_test = p_test.max(1)[p_test.argmax(1) != 1]
    np.savez(ROOT / "models" / f"{name.split()[0].lower()}_mm.npz", tick=tick, unit=unit, class_move=class_move,
             mu=QM["mu"], a=QM["a"], out_scale=QM["out_scale"],
             tau_pull=np.quantile(np.r_[p_tail.max(1)[p_tail.argmax(1) != 1]], C.pull_quantile),
             tau_taker=np.quantile(np.r_[p_tail.max(1)[p_tail.argmax(1) != 1]], T.pull_quantile),
             A_half=A.half, A_inv=A.inv_skew, C_half=C.half, C_inv=C.inv_skew, C_kappa=C.kappa,
             T_hold=int(T.hold_events), maker_bps=maker_bps, taker_bps=taker_bps,
             **dict(zip(["W1", "b1", "m1", "r1", "W2", "b2", "m2", "r2", "W3", "b3"], QM["P"])))
    return rows, curves, info, tune


def plots(name, curves, slug):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("A  market maker, no model", "#9aa5a0"), ("C  NSQ-MM (model skew + pulling)", "#00b36b"), ("T  TinyML signal taker", "#2a78d6")]
    fig, ax = plt.subplots(figsize=(9, 4))
    t0 = min(c[0][0] for c in curves.values() if len(c[0]))
    for k, col in show:
        t, v = curves[k]
        if len(t):
            ax.plot((np.asarray(t) - t0) / 3.6e6, v, color=col, lw=2, label=k)
    ax.axhline(0, color="#999", lw=1)
    ax.set(xlabel="hours into the test day", ylabel="P&L before fees (USDT)",
           title=f"{name}: test day, parameters tuned on the previous day")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / f"equity_{slug}.png", dpi=120); plt.close()


def selftest():
    mm = MarketMaker(size=1.0, latency_ms=0, use_signal=False, pull=False, inv_skew=0.0)
    bp, bq, ap, aq = (100, 99, 98, 97, 96), (5, 5, 5, 5, 5), (101, 102, 103, 104, 105), (5, 5, 5, 5, 5)
    mm.on_book(0, bp, bq, ap, aq)              # decides bid 100, ask 101
    mm.on_book(1, bp, bq, ap, aq)              # orders go live behind 5 lots each
    assert mm.live[+1] == [100, 5.0] and mm.live[-1] == [101, 5.0]
    mm.on_trade(2, 100, 3, -1)                 # 3 lots sold at 100: still 2 ahead of us
    assert mm.live[+1][1] == 2.0 and not mm.fills
    mm.on_trade(3, 100, 3, -1)                 # 3 more: queue exhausted, we are filled
    assert mm.fills == [(3, +1, 100, 1.0)] and mm.pos == 1.0
    mm.on_trade(4, 102, 1, +1)                 # buyer trades through our ask at 101: filled there
    assert mm.fills[-1] == (4, -1, 101, 1.0) and mm.pos == 0.0
    assert mm.equity_ticks(100.5) == 1.0       # bought 100, sold 101: one tick earned
    # latency: an order decided at t lives only from t + latency
    mm2 = MarketMaker(size=1.0, latency_ms=50, use_signal=False, pull=False)
    mm2.on_book(0, bp, bq, ap, aq)
    mm2.on_trade(10, 99, 1, -1)
    assert not mm2.fills and mm2.live[+1] is None
    # pulling: a confident DOWN call removes the bid
    mm3 = MarketMaker(size=1.0, latency_ms=0)
    mm3.on_book(0, bp, bq, ap, aq, move=-0.2, pull_bid=True)
    mm3.on_book(1, bp, bq, ap, aq, move=-0.2, pull_bid=True)
    assert mm3.live[+1] is None and mm3.live[-1] is not None
    print("selftest ok: queue position, trade-through fills, latency, pulling, P&L")


if __name__ == "__main__":
    import json
    import pandas as pd
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--selftest", action="store_true")
    ap_.add_argument("--quick", action="store_true")
    a = ap_.parse_args()
    if a.selftest:
        selftest()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        allrows, tunes, meta = [], [], {}
        for name in MARKETS:
            rows, curves, info, tune = backtest_market(name, a.quick)
            allrows += rows; tunes.append(tune); meta[name] = info
            plots(name, curves, name.split()[0].lower())
            pd.DataFrame(allrows).to_csv(OUT / "backtest.csv", index=False)
            pd.concat(tunes).to_csv(OUT / "tuning.csv", index=False)
        (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print("wrote", OUT)
