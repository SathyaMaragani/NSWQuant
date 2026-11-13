"""Collect every saved result into web/results_data.json for the Results page (web/results.html).

    python make_results.py

Reads only files the experiments wrote: results/hftsim/{summary,multiday,x_delay}.csv, multiday_curves.json,
results/markets_summary.csv, results/deployment_latency.csv, results/latency.csv. Missing files are skipped.
"""
import json
import math
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, OUT = ROOT / "results", ROOT / "web" / "results_data.json"


def read(p):
    p = RES / p
    return pd.read_csv(p) if p.exists() else None


def clean(x):
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, 4)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def records(df, cols):
    return [] if df is None else df[[c for c in cols if c in df.columns]].to_dict("records")


def main():
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "tune_day": "2026-11-09", "test_day": "2026-11-10"}

    s = read("hftsim/summary.csv")
    if s is not None:
        s["market"] = s.market.str.split().str[0]
        s["key"] = s.strategy.str.split().str[0]
        out["strategies"] = records(s, ["market", "profile", "key", "strategy", "fills", "gross_pnl", "fees", "net_pnl",
                                        "maker_markout_10s_ticks", "maker_spread_capture_ticks", "max_drawdown",
                                        "half_bp", "c1_bp", "skew_bp", "zcol"])

    m = read("hftsim/multiday.csv")
    if m is not None:
        out["days"] = records(m, ["market", "day", "strategy", "fills", "gross_pnl", "rebate", "net_pnl", "max_drawdown",
                                  "markout_10s_ticks", "day_move_usd", "final_position"])
        summ = []
        for (mk, k), g in m.groupby(["market", "strategy"]):
            n, mu, sd = len(g), g.net_pnl.mean(), g.net_pnl.std(ddof=1)
            summ.append({"market": mk, "strategy": k, "days": n, "net_total": g.net_pnl.sum(), "gross_total": g.gross_pnl.sum(),
                         "net_mean": mu, "net_t": mu / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None,
                         "positive_days": int((g.net_pnl > 0).sum()), "worst_day": g.net_pnl.min(), "best_day": g.net_pnl.max(),
                         "max_drawdown": g.max_drawdown.min()})
        out["days_summary"] = summ
    c = RES / "hftsim" / "multiday_curves.json"
    if c.exists():
        out["curves"] = json.loads(c.read_text())

    d = read("hftsim/x_delay.csv")
    out["delay"] = records(d, ["market", "day", "signal_delay_s", "fills", "gross_pnl", "net_pnl", "max_drawdown"])

    a = read("markets_summary.csv")
    if a is not None:
        a = a[(a.k == 10) & (a.labels == "future-only")]
        acc = []
        for mk, g in a.groupby("market", sort=False):
            get = lambda name: g.loc[g.model == name, "accuracy"].squeeze() if (g.model == name).any() else None
            f1 = g.loc[g.model == "mlp_int8", "macro_f1"]
            acc.append({"market": mk, "model": get("mlp_int8"), "macro_f1": f1.squeeze() if len(f1) else None,
                        "majority": get("majority"), "persistence": get("persistence"), "logreg": get("logreg"),
                        "top50": get("mlp_int8 @ top 50% conf"), "top25": get("mlp_int8 @ top 25% conf"),
                        "top10": get("mlp_int8 @ top 10% conf")})
        out["accuracy"] = acc

    dl = read("deployment_latency.csv")
    if dl is not None:
        out["latency"] = records(dl.rename(columns={"path": "name"}), ["name", "p50_us", "p99_us", "in_kernel_ns"])
    lt = read("latency.csv")
    if lt is not None:
        lt = lt.rename(columns={lt.columns[0]: "name"})
        out["latency_models"] = records(lt[lt.name != "timer overhead"], ["name", "p50_us", "p99_us", "in_kernel_ns"])
    out["model_bytes"] = 532

    OUT.write_text(json.dumps(clean(out), separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB): " + ", ".join(k for k in out if k not in ("generated",)))


if __name__ == "__main__":
    main()
