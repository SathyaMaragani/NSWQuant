"""Market loaders shared by the notebook and experiments.py: each market -> (train sessions, test sessions, tick[, extras]).
A session is a raw book array in tinyml.BOOK_COLS order; nothing crosses a session boundary."""
import glob
import json
from pathlib import Path

import numpy as np

import tinyml as tm

ROOT = Path(__file__).resolve().parent
FI_COL = {10: 40, 50: 43}          # FI-2010 official label columns: 40..44 = horizons 10, 20, 30, 50, 100 events
NIFTY, FI_NAME = "India · NIFTY futures (NSE)", "Nordic equities · FI-2010 (Nasdaq Nordic)"
BTC, ETH, SIM = "Crypto spot · BTCUSDT (Bybit)", "Crypto perpetual · ETHUSDT (Bybit)", "Synthetic (simulator)"


def split_sessions(S):
    """Train on earlier sessions, test on the last one; a single session is split 80/20 in time."""
    if len(S) >= 2:
        return S[:-1], S[-1:]
    i = int(0.8 * len(S[0]))
    return [S[0][:i]], [S[0][i:]]


def market_dataset(sessions, tick, horizons, extras=None):
    """Raw sessions -> features, future moves (ticks, rounded so ±alpha never flips on float noise), optional extras.
    Drops warm-up rows and rows whose future is unknown."""
    Xs, Ds, Es = [], {h: [] for h in horizons}, []
    for i, R in enumerate(sessions):
        X = tm.run_features(R, tick)
        m = 0.5 * (R[:, 0] + R[:, 10]) / tick
        d = {h: np.round(tm.future_move(m, h), 9) for h in horizons}
        keep = (np.arange(len(R)) >= tm.W) & np.all([np.isfinite(d[h]) for h in horizons], axis=0)
        Xs.append(X[keep])
        for h in horizons:
            Ds[h].append(d[h][keep])
        if extras is not None:
            Es.append(extras[i][keep])
    E = np.concatenate(Es) if extras is not None else None
    return np.vstack(Xs), {h: np.concatenate(Ds[h]) for h in horizons}, E


def fi_sessions(a):
    """FI-2010 rows (levels as askP, askV, bidP, bidV) -> one book array per stock segment + official labels."""
    R = np.zeros((len(a), len(tm.BOOK_COLS)))
    for i in range(tm.LEVELS):
        R[:, i], R[:, 5 + i], R[:, 10 + i], R[:, 15 + i] = a[:, 4 * i + 2], a[:, 4 * i + 3], a[:, 4 * i], a[:, 4 * i + 1]
    assert (R[:, 10] > R[:, 0]).all()
    mid = 0.5 * (R[:, 0] + R[:, 10])
    cuts = np.r_[0, np.flatnonzero(np.abs(np.diff(np.log(mid))) > 0.03) + 1, len(R)]   # price jump = next stock
    off = np.stack([(3 - a[:, c]).astype(int) for c in FI_COL.values()], axis=1)       # 0 down, 1 flat, 2 up
    return [R[s:e] for s, e in zip(cuts[:-1], cuts[1:])], [off[s:e] for s, e in zip(cuts[:-1], cuts[1:])]


def load_markets(include_sim=True, nifty_min_rows=5000):
    """All markets available on disk: name -> (train sessions, test sessions, tick[, extras])."""
    import pandas as pd
    M = {}
    files = sorted(glob.glob(str(ROOT / "data" / "nifty_fut_*.csv")))
    if files:
        df = pd.concat([pd.read_csv(f).assign(session=i) for i, f in enumerate(files)], ignore_index=True)
        if len(df) >= nifty_min_rows:
            df = tm.clean_book(df)
            meta = ROOT / "data" / "meta.json"
            tick = float(json.loads(meta.read_text())["tick_size"]) if meta.exists() else 0.05
            M[NIFTY] = (*split_sessions([g[tm.BOOK_COLS].to_numpy(np.float64) for _, g in df.groupby("session", sort=True)]), tick)
    fi = ROOT / "fi2010" / "fi2010.npz"
    if fi.exists():
        z = np.load(fi)
        (fa, oa), (fb, ob) = fi_sessions(z["train"]), fi_sessions(z["test"])
        M[FI_NAME] = (fa, fb, 1e-4, (oa, ob))                          # train days 1-7, test days 8-10
    for cat, sym, label in [("spot", "BTCUSDT", BTC), ("linear", "ETHUSDT", ETH)]:
        fs = sorted(glob.glob(str(ROOT / "data" / "markets" / f"bybit_{cat}_{sym}_*.npz")))
        if fs:
            M[label] = (*split_sessions([np.load(f)["R"] for f in fs]), float(np.load(fs[0])["tick"]))
    if include_sim:
        sim = [g[tm.BOOK_COLS].to_numpy(np.float64) for _, g in tm.simulate_lob(tick=0.05).groupby("session")]
        M[SIM] = (*split_sessions(sim), 0.05)
    return M
