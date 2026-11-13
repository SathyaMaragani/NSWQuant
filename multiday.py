"""Out-of-sample check across days: the settings chosen on 2026-11-09 (hftsim.py), frozen, replayed on every other
day we have under the hft_firm profile. Nothing is re-tuned here.

    python multiday.py            # -> results/hftsim/multiday.csv, multiday_curves.json, x_delay.csv

X  = imbalance market maker with the Binance +-2.5 % / +-1 % signal (the hftsim choice for each market)
X0 = the same quoting with the signal switched off (c1 = 0): what the Binance signal itself is worth
Q  = NSQ-MM-Q, the TinyML queue-aware market maker (model trained once on 2026-11-09, as in hftsim)
Data per day: data/markets/bybit_*_<day>.npz (fetch_bybit.py), raw/trades/*_<day>.csv.gz,
raw/binance/bookDepth/*-bookDepth-<day>.zip (data.binance.vision). Days with missing files are skipped.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import math

import numpy as np
import pandas as pd

import hftsim as H
import strategies as S
import tinyml as tm
from strategy import FEES, MARKETS, load_trades, model_probs, prep, signals, slice_trades

TUNE_DAY = "2026-11-09"
DAYS = ["2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-07", "2026-11-08",
        "2026-11-10", "2026-11-11", "2026-11-12"]
CHOSEN = {   # hftsim.py, hft_firm, tuned on the 2026-11-09 tail (results/hftsim/summary.csv)
    "ETHUSDT perpetual (Bybit)": {"X": dict(half_bp=1.0, c1_bp=2.0, skew_bp=0.02, zcol=2),
                                  "Q": dict(half=0.5, inv_skew=0.6, kappa=0.0, tau=0.25, imb_min=0.5, max_ahead=0.5, guard=True)},
    "BTCUSDT spot (Bybit)": {"X": dict(half_bp=0.5, c1_bp=5.0, skew_bp=0.02, zcol=3),
                             "Q": dict(half=0.5, inv_skew=2.0, kappa=3.0, tau=0.25, imb_min=0.5, max_ahead=0.5, guard=True)},
}


def files(name, day):
    tr_f, _, trd_te, fmt, unit, _ = MARKETS[name]
    book = H.ROOT / "data" / "markets" / tr_f.replace(TUNE_DAY, day)
    trades = H.ROOT / "raw" / "trades" / trd_te.replace("2026-11-10", day)
    depth = H.ROOT / "raw" / "binance" / "bookDepth" / f"{name.split()[0]}-bookDepth-{day}.zip"
    return book, trades, depth, fmt, unit


def run_day(args):
    name, day, QM, class_move = args
    book, trades, depth, fmt, unit = files(name, day)
    if not (book.exists() and trades.exists() and depth.exists()):
        return [], None
    d = np.load(book)
    R, ts, tick = d["R"], d["ts"].astype(np.int64), float(d["tick"])
    D = prep(R, ts, tick)
    T = slice_trades(load_trades(trades, fmt, tick), D["ts"][0], D["ts"][-1])
    B = tuple([list(r) for r in side] for side in D["B"])
    Z = np.column_stack([H.book_imbalance_z(ts, R), H.binance_imbalance_z(ts, name.split()[0], day)])
    prob = tm.softmax(tm.qmlp_forward(QM, tm.run_features(R, tick))[0] * QM["out_scale"])
    sig = (*signals(prob, class_move, 9.0), prob)
    x = CHOSEN[name]["X"]
    runs = {"X": (S.ImbalanceMM, x), "X0": (S.ImbalanceMM, {**x, "c1_bp": 0.0}), "Q": (S.QueueAwareMM, CHOSEN[name]["Q"])}
    rows, curves = [], {}
    for key, (cls, kw) in runs.items():
        ex = H.Exchange(H.profiles(*FEES[name])["hft_firm"], tick)
        eq_t, eq = H.replay(ex, cls(size=unit, **kw), D["ts"], B, T, sig, aux=Z)
        curves[key] = [[int(t), round(float(v), 3)] for t, v in zip(eq_t[::300], eq[::300])]   # 5-min, before fees
        r = H.report(ex, key, D["ts"], D["mid"], eq)
        move = (D["mid"][-1] - D["mid"][0]) * tick
        rows.append({"market": name.split()[0], "day": day, "strategy": key, "fills": r["fills"], "gross_pnl": r["gross_pnl"],
                     "rebate": -r["fees"], "net_pnl": r["net_pnl"], "max_drawdown": r["max_drawdown"],
                     "markout_10s_ticks": r.get("maker_markout_10s_ticks"), "day_move_usd": move,
                     "final_position": r["final_position"]})
    return rows, {"market": name.split()[0], "day": day, "curves": curves}


def delay_check(day="2026-11-10", lags=(1, 30, 60, 300, 3600)):
    """The X setting on the test day with the Binance signal arriving later and later: a real minutes-ahead signal
    fades gradually; a timestamp leak would vanish at the first delay."""
    out = []
    for name in MARKETS:
        book, trades, depth, fmt, unit = files(name, day)
        d = np.load(book)
        R, ts, tick = d["R"], d["ts"].astype(np.int64), float(d["tick"])
        D = prep(R, ts, tick)
        T = slice_trades(load_trades(trades, fmt, tick), D["ts"][0], D["ts"][-1])
        B = tuple([list(r) for r in side] for side in D["B"])
        n, zb = len(ts), H.book_imbalance_z(ts, R)
        sig = (np.zeros(n), np.zeros(n, bool), np.zeros(n, bool), None)
        for lag in lags:
            Z = np.column_stack([zb, H.binance_imbalance_z(ts, name.split()[0], day, lag_s=lag)])
            ex = H.Exchange(H.profiles(*FEES[name])["hft_firm"], tick)
            _, eq = H.replay(ex, S.ImbalanceMM(size=unit, **CHOSEN[name]["X"]), D["ts"], B, T, sig, aux=Z)
            r = H.report(ex, "X", D["ts"], D["mid"], eq)
            out.append({"market": name.split()[0], "day": day, "signal_delay_s": lag, "fills": r["fills"],
                        "gross_pnl": r["gross_pnl"], "net_pnl": r["net_pnl"], "max_drawdown": r["max_drawdown"]})
            print(f"  delay {name.split()[0]} {lag:5d}s  net {r['net_pnl']:8.2f}")
    pd.DataFrame(out).to_csv(H.ROOT / "results" / "hftsim" / "x_delay.csv", index=False)


def main():
    jobs = []
    for name in MARKETS:
        tr_f, *_ = MARKETS[name]
        t = np.load(H.ROOT / "data" / "markets" / tr_f)
        _, _, _, class_move, info = model_probs(t["R"], t["R"][:1000], float(t["tick"]))   # same training as hftsim
        jobs += [(name, day, info["QM"], class_move) for day in DAYS]
    with ProcessPoolExecutor(max_workers=min(12, len(jobs))) as pool:
        res = list(pool.map(run_day, jobs))
    rows = [r for rs, _ in res for r in rs]
    curves = [c for rs, c in res if rs]
    (H.ROOT / "results" / "hftsim" / "multiday_curves.json").write_text(json.dumps(curves, separators=(",", ":")))
    df = pd.DataFrame(rows).sort_values(["market", "day", "strategy"])
    out = H.ROOT / "results" / "hftsim" / "multiday.csv"
    df.to_csv(out, index=False)
    piv = df.pivot_table(index=["market", "day"], columns="strategy", values=["net_pnl", "gross_pnl"]).round(2)
    print(piv.to_string())
    print()
    for (m, k), g in df.groupby(["market", "strategy"]):
        n, mu, sd = len(g), g.net_pnl.mean(), g.net_pnl.std(ddof=1)
        t = mu / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        print(f"{m} {k:3s} days {n}  net total {g.net_pnl.sum():9.2f}  mean/day {mu:8.2f}  t {t:5.2f}  "
              f"positive days {int((g.net_pnl > 0).sum())}/{n}  gross total {g.gross_pnl.sum():9.2f}")
    w = df.pivot_table(index=["market", "day"], columns="strategy", values="net_pnl")
    for m, g in w.groupby(level=0):
        diff = (g["X"] - g["X0"]).dropna()
        if len(diff) > 1:
            print(f"{m} signal value X - X0: total {diff.sum():.2f}, better on {int((diff > 0).sum())}/{len(diff)} days, "
                  f"t {diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))):.2f}")
    delay_check()
    print("wrote", out)


if __name__ == "__main__":
    main()
