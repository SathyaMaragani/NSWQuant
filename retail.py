"""Algorithm D: can a normal retail account make money on ETH with the Binance depth signal, at base fees?

    python retail.py tune        # grid on the 10 November development days -> results/retail/tuning.csv, chosen.json
    python retail.py holdout     # the chosen setting, frozen, once on 14 unseen days (20 Oct - 2 Nov)
    python retail.py --selftest

Everything runs on hftsim's "retail" profile: 50 ms order/cancel latency, back of every queue, Bybit VIP0 fees
(maker 2 bp, taker 5.5 bp, no rebate), 10 messages a second, decisions on book updates only. Order size 1 ETH.
Market making cannot pay a 2 bp maker fee on a 1-tick (0.05 bp) spread, so D is directional: it trades the
Binance +-2.5 % depth-imbalance signal over tens of minutes (strategies.SignalSwing).
Research and simulation only: nothing here connects to an exchange.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

import hftsim as H
import strategies as S
from strategy import FEES, load_trades, prep, slice_trades

OUT = H.ROOT / "results" / "retail"
MARKET = "ETHUSDT perpetual (Bybit)"
SIZE = 1.0                                          # ETH per order (about 2,000 USDT)
DEV = ["2026-11-%02d" % d for d in range(3, 13)]
HOLDOUT = ["2026-10-%02d" % d for d in range(20, 32)] + ["2026-11-01", "2026-11-02"]
GRID1 = [dict(k_in=k, hold_s=h, entry_s=e, exit_s=x)          # round 1 (36 settings): best +113.66, 4/10 days up
         for k, h, e, x in product((1.0, 1.5, 2.0), (900, 1800, 3600), (60, 300), (60, 600))]
GRID = [dict(k_in=k, hold_s=h, entry_s=300, exit_s=600, k_out=ko, extend=ex)   # round 2 (16): hold while the signal holds
        for k, h, ko, ex in product((2.0, 2.5), (1800, 3600), (1.0, 0.0), (False, True))]


def load_day(day):
    d = np.load(H.ROOT / "data" / "markets" / f"bybit_linear_ETHUSDT_{day}.npz")
    R, ts, tick = d["R"], d["ts"].astype(np.int64), float(d["tick"])
    D = prep(R, ts, tick)
    T = slice_trades(load_trades(H.ROOT / "raw" / "trades" / f"ETHUSDT_linear_{day}.csv.gz", "linear", tick), D["ts"][0], D["ts"][-1])
    B = tuple([list(r) for r in side] for side in D["B"])
    Z = np.column_stack([H.book_imbalance_z(ts, R), H.binance_imbalance_z(ts, "ETHUSDT", day)])
    return D, T, B, Z, tick


def run(day_data, kw, day):
    D, T, B, Z, tick = day_data
    n = len(D["ts"])
    sig = (np.zeros(n), np.zeros(n, bool), np.zeros(n, bool), None)
    ex = H.Exchange(H.profiles(*FEES[MARKET])["retail"], tick)
    st = S.SignalSwing(size=SIZE, **kw)
    _, eq = H.replay(ex, st, D["ts"], B, T, sig, aux=Z)
    r = H.report(ex, "D", D["ts"], D["mid"], eq)
    notional = r["volume"]
    return {"day": day, **kw, "net": r["net_pnl"], "gross": r["gross_pnl"], "fees": r["fees"], "fills": r["fills"],
            "maker_fills": r["maker_fills"], "volume": notional, "final_pos": r["final_position"],
            "max_drawdown": r["max_drawdown"], "round_trips": st.trips, "market_exits": st.market_exits,
            "binance_coverage": float(np.isfinite(Z[:, 2]).mean())}


def run_days(args):
    day, kws = args
    t0 = time.time()
    dd = load_day(day)
    rows = [run(dd, kw, day) for kw in kws]
    print(f"  {day}: {len(kws)} settings in {time.time() - t0:.0f}s", flush=True)
    return rows


def parallel(days, kws):
    with ProcessPoolExecutor(max_workers=min(len(days), 14)) as pool:
        return [r for rows in pool.map(run_days, [(d, kws) for d in days]) for r in rows]


def summarise(d, keys):
    import pandas as pd
    g = d.groupby(keys)
    s = g.agg(net=("net", "sum"), gross=("gross", "sum"), fees=("fees", "sum"), fills=("fills", "sum"),
              days_up=("net", lambda v: int((v > 0).sum())), days=("net", "size"), worst_day=("net", "min")).reset_index()
    s["t"] = g.net.apply(lambda v: v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 and v.std() > 0 else np.nan).values
    return s


def tune():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"tuning {len(GRID)} settings on {len(DEV)} development days (retail profile, base fees)")
    d = pd.DataFrame(parallel(DEV, GRID))
    d.to_csv(OUT / "tuning_days.csv", index=False)
    keys = list(GRID[0])
    s = summarise(d, keys).sort_values("net", ascending=False)
    s.to_csv(OUT / "tuning.csv", index=False)
    print(s.head(12).round(2).to_string(index=False))
    best = {k: (int(v) if float(v).is_integer() and k != "k_in" else float(v)) for k, v in s.iloc[0][keys].items()}
    (OUT / "chosen.json").write_text(json.dumps({"chosen_on": DEV, "setting": best, "size_eth": SIZE}, indent=1))
    print("chosen:", best)


def holdout():
    import pandas as pd
    c = json.loads((OUT / "chosen.json").read_text())
    kw = c["setting"]
    days = [d for d in HOLDOUT if (H.ROOT / "data" / "markets" / f"bybit_linear_ETHUSDT_{d}.npz").exists()]
    print(f"holdout: frozen {kw} on {len(days)} unseen days")
    d = pd.DataFrame(parallel(days, [kw])).sort_values("day")
    d.to_csv(OUT / "holdout.csv", index=False)
    print(d[["day", "net", "gross", "fees", "fills", "final_pos", "max_drawdown", "binance_coverage"]].round(2).to_string(index=False))
    s = summarise(d.assign(k=0), ["k"]).iloc[0]
    print(f"TOTAL net {s.net:+.2f} USDT · {int(s.days_up)}/{int(s.days)} days up · t = {s.t:.2f} · gross {s.gross:+.2f} · fees {s.fees:.2f}")


def selftest():
    """SignalSwing on a flat synthetic market: enters passively on the signal, exits passively, falls back to market."""
    P = H.profiles(*FEES[MARKET])["retail"]
    n = 400
    ts = np.arange(n) * 1000
    bp, bq, ap, aq = [100, 99, 98, 97, 96], [5.0] * 5, [101, 102, 103, 104, 105], [5.0] * 5
    B = ([bp] * n, [bq] * n, [ap] * n, [aq] * n)
    z = np.full((n, 4), np.nan); z[5:60, 2] = 2.0; z[60:, 2] = 0.0
    sells = [(t, 100, 6.0, -1) for t in (20_500,)]              # a seller hits the bid: our 1 ETH behind 5 fills
    buys = [(t, 101, 6.0, 1) for t in (100_500,)]                # a buyer lifts the ask: our exit fills
    tr = sorted(sells + buys)
    T = tuple(np.array(c) for c in zip(*tr))
    sig = (np.zeros(n), np.zeros(n, bool), np.zeros(n, bool), None)
    ex = H.Exchange(P, 0.01)
    st = S.SignalSwing(size=1.0, k_in=1.5, hold_s=60, entry_s=120, exit_s=600)
    H.replay(ex, st, ts, B, T, sig, aux=z)
    f = [(x.side, x.px, x.qty, x.maker) for x in ex.fills]
    assert f == [(1, 100, 1.0, True), (-1, 101, 1.0, True)], f
    assert ex.pos == 0 and ex.fees > 0                           # bought the bid, sold the ask, paid two maker fees
    ex2 = H.Exchange(P, 0.01)                                    # no buyer arrives: the exit falls back to market
    st2 = S.SignalSwing(size=1.0, k_in=1.5, hold_s=60, entry_s=120, exit_s=30)
    T2 = tuple(np.array(c) for c in zip(*sells))
    H.replay(ex2, st2, ts, B, T2, sig, aux=z)
    f2 = [(x.side, x.px, x.qty, x.maker) for x in ex2.fills]
    assert f2 == [(1, 100, 1.0, True), (-1, 100, 1.0, False)], f2
    assert ex2.pos == 0 and st2.state == "flat"
    print("selftest ok: passive entry on the signal, passive exit, market fallback, retail fees")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["tune", "holdout"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.cmd == "tune":
        tune()
    elif a.cmd == "holdout":
        holdout()
