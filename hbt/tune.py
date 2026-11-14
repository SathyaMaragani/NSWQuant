"""Quick honest tuning of the 0.5 ETH variant on one hour, to be tested once on a different hour.

    hbt/.venv/Scripts/python.exe hbt/tune.py 2026-11-12 13:00 14:00     # -> results/hbt/tune_<window>.csv, chosen.json
"""
import json
import sys
from datetime import datetime, timezone
from itertools import product

import numpy as np
import pandas as pd
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

import run_x as R


def one(day, start, end, cfg, step_ns, quote_ns, z, day0):
    data = R.ROOT / "hbt" / "data" / f"ETHUSDT_{day}_{start.replace(':', '')}_{end.replace(':', '')}.npz"
    asset = (BacktestAsset().data([str(data)]).linear_asset(1.0).constant_order_latency(R.LATENCY_NS, R.LATENCY_NS)
             .power_prob_queue_model(3.0).no_partial_fill_exchange().trading_value_fee_model(R.MAKER_FEE, R.TAKER_FEE)
             .tick_size(R.TICK).lot_size(R.LOT).last_trades_capacity(50_000))
    hbt = HashMapMarketDepthBacktest([asset])
    rec = np.full((40_000, 9), np.nan)
    heat = np.zeros((1, 2 * R.HEAT_W + 1), np.float32)
    n_rec, _, n_fill, _ = R.run(hbt, day0 * 1_000_000, z, cfg["half_bp"], cfg["c1_bp"], cfg["skew_bp"], cfg["max_pos"], cfg["size"],
                                step_ns, quote_ns, cfg.get("z_min", 0.0), rec, heat, np.zeros(1, np.int64), np.zeros(1, np.int64),
                                np.zeros((200_000, 5)), np.zeros((1, 4)), np.zeros(2, np.int64))
    sv = hbt.state_values(0)
    mid = (rec[n_rec - 1, 1] + rec[n_rec - 1, 2]) / 2
    gross = sv.balance + sv.position * mid
    hbt.close()
    return {"fills": int(n_fill), "gross": gross, "net": gross - sv.fee, "net_retail": gross - sv.trading_value * R.RETAIL_MAKER,
            "max_dd": float(np.nanmin(rec[:n_rec, 6] - np.fmax.accumulate(rec[:n_rec, 6])))}


# round 1 (16 settings, z_min 0): all lost; round 2 adds the signal threshold z_min, 1 s re-quotes
GRID = list(product((1.0, 2.0, 3.0), (2.0, 5.0), (1000,), (1.0, 1.5, 2.0)))


def main(day, start, end):
    day0 = int(datetime.fromisoformat(f"{day}T00:00:00").replace(tzinfo=timezone.utc).timestamp() * 1000)
    z = R.binance_imbalance_z(day0 + 1000 * np.arange(86_400, dtype=np.int64), "ETHUSDT", day)[:, 0].astype(np.float64)
    rows = []
    for half, c1, q, zm in GRID:
        cfg = dict(half_bp=half, c1_bp=c1, skew_bp=0.02, max_pos=4, size=0.5, z_min=zm)
        r = one(day, start, end, cfg, 10_000_000 if q == 10 else R.STEP_NS, q * 1_000_000, z, day0)
        rows.append({**cfg, "quote_ms": q, **r})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows).sort_values("net", ascending=False)
    tag = f"{day}_{start.replace(':', '')}_{end.replace(':', '')}"
    d.to_csv(R.OUT / f"tune_{tag}_round2.csv", index=False)
    best = d.iloc[0]
    chosen = {"tuned_on": [day, start, end], "settings": {k: float(best[k]) for k in ("half_bp", "c1_bp", "skew_bp", "size", "z_min")} | {"max_pos": int(best["max_pos"])},
              "quote_ms": int(best["quote_ms"]), "tuning_net": float(best["net"]), "tried": len(d)}
    (R.OUT / "chosen_round2.json").write_text(json.dumps(chosen, indent=1))
    print(d.round(2).to_string(index=False)); print("chosen:", chosen)


if __name__ == "__main__":
    main(*sys.argv[1:4])
