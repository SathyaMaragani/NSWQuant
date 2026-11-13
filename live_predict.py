"""Real-time NIFTY mid-price prediction on the recorder's live stream, with a local dashboard.

READ-ONLY: it reads the CSV that collect_kite.py writes; it never talks to the broker and places no orders.

    python live_predict.py train                      # fit the integer-only model on recorded NIFTY ticks
    python live_predict.py live   [--speed 20] [--port 8765]   # live during NSE hours, looping replay when closed
    python live_predict.py replay data/nifty_fut_YYYYMMDD.csv [--speed 20] [--port 8765]
    python live_predict.py --selftest

Dashboard: http://127.0.0.1:<port>/  (served from results/live/, local machine only)
Every prediction is scored once its outcome is known (mean mid over the next h ticks vs the mid now, the
same future-only label as in training). Live predictions are logged to results/live/predictions_<date>_live.csv.
"""
import argparse
import csv
import datetime as dt
import http.server
import json
import math
import os
import shutil
import sys
import threading
import time
from collections import deque
from contextlib import closing
from functools import partial
from pathlib import Path

import numpy as np

import tinyml as tm
from nse_session import IST, next_open, session_hours

ROOT = Path(__file__).resolve().parent
DATA, MODELS, LIVE = ROOT / "data", ROOT / "models", ROOT / "results" / "live"
MODEL = MODELS / "nifty_int8.npz"
H = 10                      # horizon in ticks
CLASSES = ("down", "flat", "up")
P_KEYS = ["W1", "b1", "m1", "r1", "W2", "b2", "m2", "r2", "W3", "b3"]


def load_sessions():
    import pandas as pd
    files = sorted(DATA.glob("nifty_fut_*.csv"))
    if not files:
        sys.exit("No recorded NIFTY files in data/. Run collect_kite.py during market hours first.")
    df = pd.concat([pd.read_csv(f).assign(session=i) for i, f in enumerate(files)], ignore_index=True)
    df = tm.clean_book(df)
    return [g[tm.BOOK_COLS].to_numpy(np.float64) for _, g in df.groupby("session", sort=True)], files


def tick_size():
    meta = DATA / "meta.json"
    return float(json.loads(meta.read_text())["tick_size"]) if meta.exists() else 0.05


def train():
    sessions, files = load_sessions()
    tick = tick_size()
    Xs, Ds = [], []
    for R in sessions:
        X = tm.run_features(R, tick)
        d = tm.future_move(0.5 * (R[:, 0] + R[:, 10]) / tick, H)
        keep = (np.arange(len(R)) >= tm.W) & np.isfinite(d)
        Xs.append(X[keep]); Ds.append(d[keep])
    X, D = np.vstack(Xs), np.round(np.concatenate(Ds), 9)
    if len(X) < 1000:
        sys.exit(f"Only {len(X)} usable ticks recorded; need at least 1,000.")
    cut = int(0.8 * len(X))                      # chronological: fit on the first 80%, check on the last 20%
    alpha = float(np.quantile(np.abs(D[:cut]), 1 / 3))
    y = 1 + (D > alpha).astype(int) - (D < -alpha).astype(int)
    mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-9
    Z = (X - mu) / sd
    ws = tm.train_mlp(Z[:cut], y[:cut], Z[cut:], y[cut:], bs=256)
    QM = tm.quantize_mlp(ws, mu, sd, X[:cut])
    _, p = tm.qmlp_forward(QM, X[cut:])
    yv = y[cut:]
    acc = float((p == yv).mean())
    base = float(np.bincount(yv, minlength=3).max() / len(yv))
    persist = np.where(X[cut:, 8] > 0, 2, np.where(X[cut:, 8] < 0, 0, 1))
    MODELS.mkdir(exist_ok=True)
    np.savez(MODEL, mu=QM["mu"], a=QM["a"], out_scale=QM["out_scale"], tick=tick, alpha=alpha, horizon=H,
             trained_ticks=len(X), files=np.array([f.name for f in files]),
             val_accuracy=acc, val_majority=base, val_persistence=float((persist == yv).mean()),
             **dict(zip(P_KEYS, QM["P"])))
    print(f"trained on {cut:,} ticks, checked on {len(yv):,} later ticks: accuracy {acc:.3f} "
          f"(majority {base:.3f}, persistence {(persist == yv).mean():.3f}); alpha {alpha:.3f} ticks -> {MODEL.name}")


class Engine:
    """Streams book rows through the deployed integer-only model and keeps the dashboard state."""

    def __init__(self, mode, symbol, replay=None, log_predictions=True):
        m = np.load(MODEL)
        self.mode, self.symbol = mode, symbol
        self.stream_id = str(time.time_ns())
        self.replay = replay
        self.tick, self.alpha, self.out_scale = float(m["tick"]), float(m["alpha"]), float(m["out_scale"])
        self.mu, self.a = m["mu"], m["a"]
        self.P = tuple(m[k] for k in P_KEYS)
        W1, W2, W3 = self.P[0], self.P[4], self.P[8]
        self.info = {"trained_ticks": int(m["trained_ticks"]), "val_accuracy": float(m["val_accuracy"]),
                     "val_majority": float(m["val_majority"]), "val_persistence": float(m["val_persistence"]),
                     "arch": f"{W1.shape[1]}→{W1.shape[0]}→{W2.shape[0]}→{W3.shape[0]}",
                     "bytes": sum(v for k, v in tm.model_bytes({"mu": self.mu, "P": self.P}).items() if k != "macs"),
                     "runtime": "numba-compiled, keep-warm"}
        self.hist = {k: deque(maxlen=120) for k in ("acc", "lat", "lag", "conf")}   # one point per dashboard refresh
        self.st, self.xb, self.xq, self.h1, self.h2, self.lg = tm.buffers({"mu": self.mu, "P": self.P})
        self.mids = []                                   # every mid since start (for outcomes)
        self.pending = deque()                           # (index, class, confidence)
        self.series = deque(maxlen=300)
        self.compute_ns, self.lag_ms = deque(maxlen=5000), deque(maxlen=5000)
        self.conf = np.zeros((3, 3), int)
        self.hits = deque(maxlen=200)
        self.conf_hits = []                              # outcomes of confident directional calls
        self.n = 0
        # compile once so the first live tick isn't timed with JIT compilation
        tm.tick_predict(np.zeros(tm.ST_SIZE), np.ones(23), self.tick, self.mu, self.a, self.xb.copy(), self.xq.copy(),
                        *self.P, self.h1.copy(), self.h2.copy(), self.lg.copy())
        self.log = None
        if log_predictions:
            LIVE.mkdir(parents=True, exist_ok=True)
            logp = LIVE / f"predictions_{dt.datetime.now(IST):%Y%m%d}_{mode}.csv"
            new = mode != "live" or not logp.exists()      # live appends across restarts
            self.log = open(logp, "w" if new else "a", newline="")
            self.w = csv.writer(self.log)
            if new:
                self.w.writerow(["exch_ts", "mid", "pred", "conf", "compute_ns", "lag_ms", "outcome"])

    def close(self):
        if self.log is not None:
            self.log.close()

    def keep_warm(self, seconds):
        """Run the hot path on a dummy tick while waiting, so code and data stay in cache (HFT keep-warm).
        Uses separate buffers: the live feature state is never touched. Burns one CPU core while idle."""
        if not hasattr(self, "_dummy"):
            self._dummy = tm.buffers({"mu": self.mu, "P": self.P}), (np.array(self._last_r) if hasattr(self, "_last_r") else np.ones(23))
        (st, xb, xq, h1, h2, lg), r = self._dummy
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            tm.tick_predict(st, r, self.tick, self.mu, self.a, xb, xq, *self.P, h1, h2, lg)

    def warm(self, r):
        """Advance the feature state without predicting (history before the live start)."""
        tm.feat_update(self.st, r, self.tick, self.xb)
        self.mids.append(0.5 * (r[0] + r[10]))
        self._last_r = r

    def on_row(self, r, exch_ts, recv_ns):
        t0 = time.perf_counter_ns()
        cls = tm.tick_predict(self.st, r, self.tick, self.mu, self.a, self.xb, self.xq, *self.P, self.h1, self.h2, self.lg)
        ns = time.perf_counter_ns() - t0
        z = self.lg * self.out_scale
        p = np.exp(z - z.max()); p /= p.sum()
        conf = float(p[cls])
        self._probs = [round(float(v), 4) for v in p]
        lag = (time.time_ns() - recv_ns) / 1e6 if (self.mode == "live" and recv_ns) else None
        mid = 0.5 * (r[0] + r[10])
        i = len(self.mids)
        self.mids.append(mid)
        self.pending.append((i, int(cls), conf))
        self.compute_ns.append(ns)
        if lag is not None:
            self.lag_ms.append(lag)
        self.n += 1
        self._last_r = r
        self.series.append({"t": exch_ts, "mid": mid, "bid": float(r[0]), "ask": float(r[10]), "pred": int(cls),
                            "conf": round(conf, 3), "out": None, "i": i,
                            "lv": [r[0:5].tolist(), r[5:10].tolist(), r[10:15].tolist(), r[15:20].tolist()]})  # depth heatmap
        self._resolve()
        if self.log is not None:
            self.w.writerow([exch_ts, mid, CLASSES[cls], round(conf, 4), ns, None if lag is None else round(lag, 2), ""])

    def _resolve(self):
        while self.pending and self.pending[0][0] + H < len(self.mids):
            i, cls, conf = self.pending.popleft()
            d = round((np.mean(self.mids[i + 1:i + 1 + H]) - self.mids[i]) / self.tick, 9)   # rounded: no float flips at ±alpha
            y = 1 + int(d > self.alpha) - int(d < -self.alpha)
            self.conf[y, cls] += 1
            self.hits.append(int(cls == y))
            if cls != 1 and conf >= 0.5:
                self.conf_hits.append(int(cls == y))
            for s in self.series:
                if s["i"] == i:
                    s["out"] = y
                    break

    def state(self):
        n_res = int(self.conf.sum())
        cn = np.array(self.compute_ns) if self.compute_ns else np.array([0])
        lg = np.array(self.lag_ms) if self.lag_ms else None
        last = self.series[-1] if self.series else None
        return {
            "mode": self.mode, "symbol": self.symbol, "horizon": H, "updated": dt.datetime.now(IST).strftime("%H:%M:%S"),
            "ticks": self.n, "resolved": n_res, "model": self.info, "replay": self.replay, "stream_id": self.stream_id,
            "last": last and {**{k: last[k] for k in ("t", "mid", "conf")}, "pred": CLASSES[last["pred"]],
                              "probs": getattr(self, "_probs", None)},
            "accuracy": (np.trace(self.conf) / n_res) if n_res else None,
            "rolling": (float(np.mean(self.hits)) if self.hits else None),
            "majority": (float(self.conf.sum(1).max() / n_res) if n_res else None),
            "confident": {"n": len(self.conf_hits), "accuracy": float(np.mean(self.conf_hits)) if self.conf_hits else None},
            "confusion": self.conf.tolist(),
            "latency": {"compute_p50_us": float(np.percentile(cn, 50)) / 1000, "compute_p99_us": float(np.percentile(cn, 99)) / 1000,
                        "lag_p50_ms": None if lg is None else float(np.percentile(lg, 50)),
                        "lag_p99_ms": None if lg is None else float(np.percentile(lg, 99))},
            "series": [{k: s[k] for k in ("t", "mid", "bid", "ask", "pred", "conf", "out", "lv")} for s in self.series],
            "book": self._book(),
            "hist": {k: list(v) for k, v in self.hist.items()},
        }

    def _book(self):
        r = getattr(self, "_last_r", None)
        if r is None:
            return None
        bq, aq = r[5:10], r[15:20]
        return {"bp": r[0:5].tolist(), "bq": bq.tolist(), "ap": r[10:15].tolist(), "aq": aq.tolist(),
                "spread": float(r[10] - r[0]), "imb1": float((r[5] - r[15]) / (r[5] + r[15])),
                "depth": float(bq.sum() + aq.sum()), "tick": self.tick}

    def publish(self):
        s = self.state()
        for k, v in (("acc", s["rolling"]), ("lat", s["latency"]["compute_p50_us"]),
                     ("lag", s["latency"]["lag_p50_ms"]), ("conf", s["confident"]["n"])):
            if v is not None:
                self.hist[k].append(round(float(v), 4))
        s["hist"] = {k: list(v) for k, v in self.hist.items()}
        publish_state(s)
        if self.log is not None:
            self.log.flush()


def publish_state(state):
    tmp = LIVE / "state.json.tmp"
    tmp.write_text(json.dumps(state))
    for _ in range(20):                # Windows can refuse the swap while the server reads the old file
        try:
            os.replace(tmp, LIVE / "state.json")
            return
        except PermissionError:
            time.sleep(0.005)


def parse(header):
    idx = {c: i for i, c in enumerate(header)}
    cols = [idx[c] for c in tm.BOOK_COLS]
    its, irv = idx.get("exch_ts"), idx.get("recv_ns")

    def row(fields):
        r = np.array([float(fields[c] or 0) for c in cols])
        ts = fields[its] if its is not None else ""
        if len(ts) == 13 and ts.isascii() and ts.isdigit():   # Upstox uses Unix milliseconds
            ts = dt.datetime.fromtimestamp(int(ts) / 1000, IST).isoformat(timespec="seconds")
        return r, ts, int(fields[irv]) if irv is not None and fields[irv] else 0
    return row


def valid(r):
    return r[0] > 0 and r[10] > r[0] and r[5] > 0 and r[15] > 0


def read_tick(parse_row, line):
    try:
        tick = parse_row(next(csv.reader([line])))
    except (IndexError, ValueError, csv.Error):
        return None
    return tick if np.isfinite(tick[0]).all() and valid(tick[0]) else None


def has_ticks(path):
    try:
        with path.open(newline="") as f:
            header = f.readline()
            if not header.endswith("\n"):
                return False
            row = parse(next(csv.reader([header])))
            return any(line.endswith("\n") and read_tick(row, line) is not None for line in f)
    except (OSError, KeyError, ValueError, csv.Error):
        return False


def latest_recording(day):
    return next((p for p in sorted(DATA.glob("nifty_fut_????????.csv"), reverse=True)
                 if p.stem[-8:].isdigit() and p.stem[-8:] <= f"{day:%Y%m%d}" and has_ticks(p)), None)


def serve(port):
    LIVE.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "web" / "dashboard.html", LIVE / "index.html")
    shutil.copytree(ROOT / "web" / "vendor", LIVE / "vendor", dirs_exist_ok=True)   # charts work offline

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Quiet, directory=str(LIVE)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"dashboard: http://127.0.0.1:{srv.server_port}/")
    return srv


def follow(path, eng, replay_speed=None, warm_rows=True, keepwarm=True, stop_at=None):
    """Tail a growing CSV (live) or replay a finished one at recorded pace / speed."""
    def session_ended():
        return stop_at is not None and dt.datetime.now(IST) >= stop_at

    last_pub, prev_ts, buf = float("-inf"), None, ""
    warming = warm_rows and replay_speed is None
    with open(path, newline="") as f:
        row = parse(next(csv.reader([f.readline()])))
        while True:
            line = f.readline()
            if not line:
                if replay_speed is not None or session_ended():
                    break
                if warming:
                    warming = False
                    print(f"warmed on {len(eng.mids):,} earlier ticks; predicting live")
                if time.monotonic() - last_pub > 1:
                    eng.publish()
                    last_pub = time.monotonic()
                eng.keep_warm(0.02) if keepwarm else time.sleep(0.02)
                continue
            buf += line
            if not buf.endswith("\n"):
                continue                              # partial line still being written
            tick, buf = read_tick(row, buf), ""
            if tick is None:
                continue
            r, ts, rv = tick
            if warming:
                eng.warm(r)
                continue
            if replay_speed is not None:
                delay = max(0.0, (rv - prev_ts) / 1e9 / replay_speed) if rv and prev_ts else 0
                if rv:
                    prev_ts = rv
                if stop_at is not None:
                    # Keep presentations moving through gaps in the recorder and stop at the next open.
                    remaining = (stop_at - dt.datetime.now(IST)).total_seconds()
                    delay = min(delay, max(0.0, remaining), 2.0)
                if delay > 0:
                    time.sleep(delay)
                if session_ended():
                    break
            eng.on_row(r, ts, rv)
            if time.monotonic() - last_pub > 0.25:
                eng.publish()
                last_pub = time.monotonic()
    eng.publish()
    s = eng.state()
    print(f"done: {s['ticks']:,} predictions, {s['resolved']:,} scored, accuracy "
          f"{s['accuracy'] if s['accuracy'] is None else round(s['accuracy'], 3)} (majority {s['majority']})")


def instrument_meta():
    try:
        meta = json.loads((DATA / "meta.json").read_text())
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def run_dashboard(speed, keepwarm):
    waiting = None
    while True:
        now = dt.datetime.now(IST)
        meta = instrument_meta()
        symbol = meta.get("tradingsymbol", "NIFTY FUT")
        hours = session_hours(now.date(), meta.get("expiry"))
        if hours and hours[0] <= now.time() < hours[1]:
            path = DATA / f"nifty_fut_{now:%Y%m%d}.csv"
            if has_ticks(path):
                waiting = None
                close = dt.datetime.combine(now.date(), hours[1], IST)
                with closing(Engine("live", symbol)) as eng:
                    follow(path, eng, keepwarm=keepwarm, stop_at=close)
                continue
            message = "Waiting for today’s live recorder"
        else:
            path = latest_recording(now.date())
            if path is not None:
                waiting = None
                opening = next_open(now)
                date = path.stem[-8:]
                replay = {"speed": speed, "automatic": True, "date": f"{date[:4]}-{date[4:6]}-{date[6:]}"}
                with closing(Engine("replay", "NIFTY FUT", replay=replay, log_predictions=False)) as eng:
                    follow(path, eng, replay_speed=speed, stop_at=opening)
                time.sleep(max(0.0, min(2.0, (opening - dt.datetime.now(IST)).total_seconds())))
                continue
            message = "Market closed · no recording available"
        if message != waiting:
            publish_state({"mode": "waiting", "symbol": symbol, "message": message})
        waiting = message
        time.sleep(1)


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    R = tm._simulate(3000, 3, 0.05)
    X = tm.run_features(R, 0.05)
    d = tm.future_move(0.5 * (R[:, 0] + R[:, 10]) / 0.05, H)
    ok = np.isfinite(d)
    y = 1 + (d[ok] > 0.3).astype(int) - (d[ok] < -0.3).astype(int)
    ws = [rng.normal(size=(16, tm.NF)).astype(np.float32) * .3, np.zeros(16, np.float32),
          rng.normal(size=(16, 16)).astype(np.float32) * .3, np.zeros(16, np.float32),
          rng.normal(size=(3, 16)).astype(np.float32) * .3, np.zeros(3, np.float32)]
    QM = tm.quantize_mlp(ws, X.mean(0), X.std(0) + 1e-9, X)
    global MODEL, LIVE
    tmpd = Path(tempfile.mkdtemp())
    MODEL, LIVE = tmpd / "m.npz", tmpd / "live"
    np.savez(MODEL, mu=QM["mu"], a=QM["a"], out_scale=QM["out_scale"], tick=0.05, alpha=0.3, horizon=H, trained_ticks=1,
             files=np.array(["x"]), val_accuracy=0.0, val_majority=0.0, val_persistence=0.0, **dict(zip(P_KEYS, QM["P"])))
    eng = Engine("replay", "TEST")
    for r in R:
        eng.on_row(r, "", 0)
    expected = tm.run_pipeline(R, 0.05, QM["mu"], QM["a"], *QM["P"])
    # the engine's predictions equal the batch pipeline, and every prediction with a known future is scored
    eng.log.close()
    import pandas as pd
    log = pd.read_csv(next(LIVE.glob("predictions_*.csv")))
    got = log.pred.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    assert (got == expected).all()
    assert eng.conf.sum() == len(R) - H
    d = np.round(d, 9)
    yy = 1 + (d > 0.3).astype(int) - (d < -0.3).astype(int)
    assert np.trace(eng.conf) == int((expected[:len(R) - H] == yy[:len(R) - H]).sum())
    print("selftest ok: live engine == batch pipeline; outcomes scored with the training label")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["train", "live", "replay"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--speed", type=float, default=20.0, help="replay speed multiplier (default: 20)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-keepwarm", action="store_true", help="sleep while idle instead of keeping the hot path warm")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not math.isfinite(a.speed) or a.speed <= 0:
        ap.error("--speed must be a finite positive number")
    if a.selftest:
        selftest()
    elif a.cmd == "train":
        train()
    elif a.cmd in ("live", "replay"):
        if not MODEL.exists():
            sys.exit("No model yet. Run: python live_predict.py train")
        if a.cmd == "replay":
            if not a.file:
                ap.error("replay needs a recorded CSV path")
            path = Path(a.file)
            if not has_ticks(path):
                ap.error(f"{path} has no usable recorded ticks")
            LIVE = ROOT / "results" / "replay"
        srv = serve(a.port)
        try:
            publish_state({"mode": "waiting", "symbol": "NIFTY FUT", "message": "Loading market data"})
            if a.cmd == "live":
                run_dashboard(a.speed, keepwarm=not a.no_keepwarm)
            else:
                sym = instrument_meta().get("tradingsymbol", "NIFTY FUT")
                with closing(Engine("replay", sym, replay={"speed": a.speed})) as eng:
                    follow(path, eng, replay_speed=a.speed)
        except KeyboardInterrupt:
            pass
        finally:
            srv.shutdown()
            srv.server_close()
    else:
        ap.print_help()
