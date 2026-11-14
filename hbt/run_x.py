"""Algorithm X + Model on hftbacktest (github.com/nkaz001/hftbacktest, 2.4.4): one replay window, recorded for a viewer.

    hbt/.venv/Scripts/python.exe hbt/run_x.py 2026-11-12 14:00 15:00    # needs hbt/data/ETHUSDT_2026-11-12_1400_1500.npz
    -> results/hbt/x_replay.json (summary, our orders and fills, series), heat.bin (book), trades.bin (market trades)

Same algorithm and frozen settings as the 9-day test (strategies.ImbalanceMM, multiday.CHOSEN for ETH) and the same
Binance +-2.5 % depth-imbalance z (hftsim.binance_imbalance_z: 1 h z, usable 1 s after each snapshot). Re-quoted once a
second; post-only (GTX) limit orders; order id = price tick, as in hftbacktest's own examples; sits out when the signal
is missing. Fills, queue position and latency come from hftbacktest's models, not ours:
  power-probability queue model (n = 3), 1 ms order entry and response latency, no partial fills,
  maker fee -0.5 bp (the market-maker rebate tier X was tested with), taker 2 bp. The summary also reports retail fees.
Simulation only: nothing here connects to an exchange.
"""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numba import njit
from hftbacktest import BUY, BUY_EVENT, FILLED, GTX, LIMIT, NEW, SELL, BacktestAsset, HashMapMarketDepthBacktest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hftsim import binance_imbalance_z  # noqa: E402  (the project's signal code, unchanged)

OUT = ROOT / "results" / "hbt"
X = dict(half_bp=1.0, c1_bp=2.0, skew_bp=0.02, max_pos=20, size=0.1)    # multiday.CHOSEN["ETHUSDT perpetual (Bybit)"]["X"]
# scalp variant (not the tested X): 0.5 ETH orders, same 2 ETH cap, tighter quotes, re-quoted every 10 ms
SCALP = dict(half_bp=0.3, c1_bp=2.0, skew_bp=0.02, max_pos=4, size=0.5)
TICK, LOT = 0.01, 0.01
LATENCY_NS = 1_000_000
MAKER_FEE, TAKER_FEE, RETAIL_MAKER = -0.00005, 0.0002, 0.0002
STEP_NS, QUOTE_NS, HEAT_NS, HEAT_W = 100_000_000, 1_000_000_000, 500_000_000, 150


@njit
def run(hbt, z_t0, z, half_bp, c1_bp, skew_bp, max_pos, size, step_ns, quote_ns, z_min, rec, heat, heat_t, heat_lo, fills, mtr, counts):
    tick = hbt.depth(0).tick_size
    n_rec = n_heat = n_fill = n_tr = 0
    next_q = 0
    next_h = 0
    next_r = 0
    zz = np.nan
    while hbt.elapse(step_ns) == 0:
        now = hbt.current_timestamp
        depth = hbt.depth(0)
        bb, ba = depth.best_bid_tick, depth.best_ask_tick
        # market trades since the last step
        lt = hbt.last_trades(0)
        for i in range(len(lt)):
            if n_tr < mtr.shape[0]:
                mtr[n_tr, 0] = lt[i].exch_ts
                mtr[n_tr, 1] = lt[i].px
                mtr[n_tr, 2] = lt[i].qty
                mtr[n_tr, 3] = 1.0 if (lt[i].ev & BUY_EVENT) == BUY_EVENT else -1.0
                n_tr += 1
        hbt.clear_last_trades(0)
        # our fills (no partial fills: an order fills completely), recorded before inactive orders are dropped
        vals = hbt.orders(0).values()
        while vals.has_next():
            o = vals.get()
            if o.status == FILLED and n_fill < fills.shape[0]:
                fills[n_fill, 0] = o.exch_timestamp
                fills[n_fill, 1] = 1.0 if o.side == BUY else -1.0
                fills[n_fill, 2] = o.exec_price_tick * tick
                fills[n_fill, 3] = o.exec_qty
                fills[n_fill, 4] = o.order_id
                n_fill += 1
        hbt.clear_inactive_orders(0)
        if bb < 0 or ba > 10 ** 15 or ba <= bb:
            continue                                  # one side of the book is empty
        mid = (bb + ba) / 2.0
        pos = hbt.position(0)
        if now >= next_q:                             # algorithm X: re-quote once a second
            next_q = now + quote_ns
            k = (now - z_t0) // 1_000_000_000
            zz = z[k] if 0 <= k < z.shape[0] else np.nan
            tpb = mid * 1e-4                          # ticks per basis point
            want_b = want_a = False
            nb = na = 0
            if np.isfinite(zz):
                res = mid + c1_bp * tpb * zz - skew_bp * tpb * pos / size
                nb = min(int(math.floor(res - half_bp * tpb)), bb)    # post-only, never better than the touch
                na = max(int(math.ceil(res + half_bp * tpb)), ba)
                want_b = pos < max_pos * size - 1e-9
                want_a = pos > -max_pos * size + 1e-9
                if abs(zz) < z_min:                   # weak signal: only the side that reduces the position
                    want_b = want_b and pos < -1e-9
                    want_a = want_a and pos > 1e-9
            vals = hbt.orders(0).values()
            while vals.has_next():
                o = vals.get()
                if o.cancellable and ((o.side == BUY and (not want_b or o.price_tick != nb)) or
                                      (o.side == SELL and (not want_a or o.price_tick != na))):
                    hbt.cancel(0, o.order_id, False)
                    counts[1] += 1
            orders = hbt.orders(0)
            if want_b and not orders.__contains__(nb):
                hbt.submit_buy_order(0, nb, nb * tick, size, GTX, LIMIT, False)
                counts[0] += 1
            if want_a and not orders.__contains__(na):
                hbt.submit_sell_order(0, na, na * tick, size, GTX, LIMIT, False)
                counts[0] += 1
        # series: book top, our working orders, position, equity, signal
        if now >= next_r and n_rec < rec.shape[0]:
            next_r = now + STEP_NS
            ob = oa = np.nan
            vals = hbt.orders(0).values()
            while vals.has_next():
                o = vals.get()
                if o.status == NEW:
                    if o.side == BUY:
                        ob = o.price_tick * tick
                    else:
                        oa = o.price_tick * tick
            sv = hbt.state_values(0)
            rec[n_rec, 0] = now
            rec[n_rec, 1] = bb * tick
            rec[n_rec, 2] = ba * tick
            rec[n_rec, 3] = ob
            rec[n_rec, 4] = oa
            rec[n_rec, 5] = pos
            rec[n_rec, 6] = sv.balance + pos * mid * tick - sv.fee
            rec[n_rec, 7] = zz
            rec[n_rec, 8] = sv.trading_value
            n_rec += 1
        # full book +-HEAT_W ticks around the mid, every HEAT_NS
        if now >= next_h and n_heat < heat.shape[0]:
            next_h = now + HEAT_NS
            lo = int(round(mid)) - HEAT_W
            for j in range(2 * HEAT_W + 1):
                t = lo + j
                heat[n_heat, j] = depth.bid_qty_at_tick(t) if t <= bb else (depth.ask_qty_at_tick(t) if t >= ba else 0.0)
            heat_t[n_heat] = now
            heat_lo[n_heat] = lo
            n_heat += 1
    return n_rec, n_heat, n_fill, n_tr


def main(day, start, end, variant="x", symbol="ETHUSDT"):
    cfg, step_ns, quote_ns = (SCALP, 10_000_000, 10_000_000) if variant == "scalp" else (X, STEP_NS, QUOTE_NS)
    if variant in ("chosen", "chosen2"):          # the setting hbt/tune.py picked on a different hour (round 1 / round 2)
        c = json.loads((OUT / ("chosen.json" if variant == "chosen" else "chosen_round2.json")).read_text())
        cfg, quote_ns = c["settings"], c["quote_ms"] * 1_000_000
        step_ns = 10_000_000 if c["quote_ms"] < 100 else STEP_NS
    data = ROOT / "hbt" / "data" / f"{symbol}_{day}_{start.replace(':', '')}_{end.replace(':', '')}.npz"
    t0 = int(datetime.fromisoformat(f"{day}T{start}:00").replace(tzinfo=timezone.utc).timestamp() * 1e9)
    t1 = int(datetime.fromisoformat(f"{day}T{end}:00").replace(tzinfo=timezone.utc).timestamp() * 1e9)
    day0 = int(datetime.fromisoformat(f"{day}T00:00:00").replace(tzinfo=timezone.utc).timestamp() * 1000)
    grid = day0 + 1000 * np.arange(86_400, dtype=np.int64)          # one value per second of the day (ms)
    z = binance_imbalance_z(grid, symbol, day)[:, 0].astype(np.float64)
    print(f"Binance z: {np.isfinite(z).mean():.0%} of the day's seconds")
    asset = (BacktestAsset().data([str(data)]).linear_asset(1.0)
             .constant_order_latency(LATENCY_NS, LATENCY_NS)
             .power_prob_queue_model(3.0)
             .no_partial_fill_exchange()
             .trading_value_fee_model(MAKER_FEE, TAKER_FEE)
             .tick_size(TICK).lot_size(LOT)
             .last_trades_capacity(50_000))
    hbt = HashMapMarketDepthBacktest([asset])
    secs = (t1 - t0) // 1_000_000_000 + 60
    rec = np.full((secs * 10, 9), np.nan)
    heat = np.zeros((secs * 2, 2 * HEAT_W + 1), np.float32)
    heat_t, heat_lo = np.zeros(secs * 2, np.int64), np.zeros(secs * 2, np.int64)
    fills, mtr = np.zeros((100_000, 5)), np.zeros((2_000_000, 4))
    counts = np.zeros(2, np.int64)
    n_rec, n_heat, n_fill, n_tr = run(hbt, day0 * 1_000_000, z, cfg["half_bp"], cfg["c1_bp"], cfg["skew_bp"], cfg["max_pos"], cfg["size"], step_ns, quote_ns, cfg.get("z_min", 0.0),
                                      rec, heat, heat_t, heat_lo, fills, mtr, counts)
    sv = hbt.state_values(0)
    hbt.close()
    rec, heat, fills, mtr = rec[:n_rec], heat[:n_heat], fills[:n_fill], mtr[:n_tr]
    mid = (rec[-1, 1] + rec[-1, 2]) / 2
    gross = sv.balance + sv.position * mid
    summary = {"orders_sent": int(counts[0]), "cancels": int(counts[1]), "fills": int(n_fill), "market_trades": int(n_tr),
               "position": float(sv.position), "traded_value": float(sv.trading_value), "gross": float(gross),
               "fees": float(sv.fee), "net": float(gross - sv.fee), "net_retail": float(gross - sv.trading_value * RETAIL_MAKER)}
    meta = {"engine": "hftbacktest 2.4.4 (HashMapMarketDepthBacktest)", "symbol": symbol, "day": day, "window_utc": [start, end],
            "window_choice": "the busiest hour of the day by market trade count (chosen before running X)",
            "models": {"queue": "power-probability (n = 3)", "latency_ms": LATENCY_NS / 1e6, "exchange": "no partial fills",
                       "maker_bp": MAKER_FEE * 1e4, "taker_bp": TAKER_FEE * 1e4, "feed_latency_ms": 1.0},
            "settings": cfg, "variant": variant, "quote_ms": quote_ns / 1e6, "tick": TICK, "heat_w": HEAT_W, "heat_ms": HEAT_NS // 1_000_000, "summary": summary,
            "t0_ms": t0 // 1_000_000, "t1_ms": t1 // 1_000_000,
            "series": {"t": (rec[:, 0] // 1_000_000).astype(np.int64).tolist(), "bid": rec[:, 1].round(2).tolist(), "ask": rec[:, 2].round(2).tolist(),
                       "our_bid": [None if math.isnan(v) else round(v, 2) for v in rec[:, 3]],
                       "our_ask": [None if math.isnan(v) else round(v, 2) for v in rec[:, 4]],
                       "pos": rec[:, 5].round(4).tolist(), "equity": rec[:, 6].round(4).tolist(),
                       "z": [None if math.isnan(v) else round(v, 3) for v in rec[:, 7]], "value": rec[:, 8].round(2).tolist()},
            "fills": [{"t": int(f[0] // 1_000_000), "side": int(f[1]), "px": round(float(f[2]), 2), "qty": round(float(f[3]), 4),
                       "id": int(f[4])} for f in fills],
            "heat_t": (heat_t[:n_heat] // 1_000_000).tolist(), "heat_lo": heat_lo[:n_heat].tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "x_replay.json").write_text(json.dumps(meta, separators=(",", ":"), allow_nan=False))
    heat.astype(np.float32).tofile(OUT / "heat.bin")
    tr = np.column_stack([(mtr[:, 0] // 1_000_000) - t0 // 1_000_000, mtr[:, 1], mtr[:, 2], mtr[:, 3]]).astype(np.float64)
    tr.tofile(OUT / "trades.bin")
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT} : {n_rec} series rows, {n_heat} book columns, {n_fill} fills, {n_tr} market trades")


if __name__ == "__main__":
    main(*sys.argv[1:5])
