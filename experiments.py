"""Heavy experiments behind the notebook's later sections (phases B, C and E of the plan). Writes results/experiments/.

    python experiments.py              # all markets on disk
    python experiments.py --quick      # smaller budgets, for a smoke test

Per market, horizon k = 10 events, chronological split, labels from future mids only:
  B  teacher vs student   a DeepLOB-style CNN+GRU teacher over 50-event windows of features and book shape
                          vs the integer-only student, with and without knowledge distillation
     QAT                  post-training vs quantisation-aware training at 8-bit and 4-bit weights
     Pareto search        hidden width x number of features -> accuracy vs bytes vs nanoseconds
  C  feature importance   permutation importance (macro-F1 drop) on the calibration set
  E  decision value       research-only: gross edge of confident calls vs the spread they would cross,
                          and how much edge survives if the prediction arrives 1-20 events late
"""
import argparse
import copy
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import markets as mkt
import tinyml as tm

OUT = Path(__file__).resolve().parent / "results" / "experiments"
H, T_WIN, SEED = 10, 50, 0
DELAYS = [0, 1, 2, 5, 10, 20]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


# ----------------------------------------------------------------------------- data
def delayed_move(m, h, L):
    """Mean mid over [t+L+1, t+L+h] minus mid at t+L: the outcome if the prediction lands L events late."""
    c = np.concatenate([[0.0], np.cumsum(m)])
    d = np.full(len(m), np.nan)
    n = len(m) - L - h
    if n > 0:
        d[:n] = (c[L + h + 1:L + h + 1 + n] - c[L + 1:L + 1 + n]) / h - m[L:L + n]
    return d


def build(sessions, tick):
    """Features, delayed moves (ticks), book shape for the teacher, spread (ticks) and session ids."""
    parts = []
    for i, R in enumerate(sessions):
        X = tm.run_features(R, tick)
        mid = 0.5 * (R[:, 0] + R[:, 10])
        m = mid / tick
        dL = np.stack([np.round(delayed_move(m, H, L), 9) for L in DELAYS], 1)
        book = np.c_[(R[:, 10:15] - mid[:, None]) / tick, (mid[:, None] - R[:, 0:5]) / tick,
                     np.log1p(R[:, 15:20]), np.log1p(R[:, 5:10])]
        keep = (np.arange(len(R)) >= tm.W) & np.isfinite(dL[:, 0])
        parts.append((X[keep], dL[keep], book[keep], ((R[:, 10] - R[:, 0]) / tick)[keep], np.full(keep.sum(), i)))
    return [np.concatenate([p[j] for p in parts]) for j in range(5)]


# ----------------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, n_in, hidden=(16, 16), wbits=None):
        super().__init__()
        dims = [n_in, *hidden, 3]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.wbits = wbits

    def _w(self, lin, last):
        if not self.wbits:
            return lin.weight
        qmax = 2 ** (self.wbits - 1) - 1                # fake quantisation with a straight-through estimator,
        w = lin.weight                                  # same scales as tinyml.quantize_mlp
        s = (w.abs().max() if last else w.abs().amax(1, keepdim=True)) / qmax
        s = torch.clamp(s, min=1e-12)
        return w + (torch.clamp(torch.floor(w / s + 0.5), -qmax, qmax) * s - w).detach()

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            last = i == len(self.lins) - 1
            x = F.linear(x, self._w(lin, last), lin.bias)
            if not last:
                x = F.relu(x)
        return x

    def weights(self):
        return [p.detach().numpy().astype(np.float32).copy() for lin in self.lins for p in (lin.weight, lin.bias)]


def train_student(Xtr, ytr, Xva, yva, hidden=(16, 16), t_logits=None, lam=0.5, temp=2.0, wbits=None, init=None,
                  epochs=40, patience=4, bs=1024, lr=3e-3):
    """Cross-entropy, optionally mixed with distillation from teacher logits (Hinton et al.), optionally QAT."""
    torch.manual_seed(SEED)
    net = MLP(Xtr.shape[1], hidden, wbits)
    if init is not None:
        net.load_state_dict(init.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xtr.astype(np.float32)), torch.from_numpy(ytr).long()
    Tl = None if t_logits is None else torch.from_numpy(t_logits.astype(np.float32))
    Xv, yv = torch.from_numpy(Xva.astype(np.float32)), torch.from_numpy(yva).long()
    best, bad, state = np.inf, 0, None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            out = net(Xt[idx])
            loss = F.cross_entropy(out, yt[idx])
            if Tl is not None:
                kd = F.kl_div(F.log_softmax(out / temp, 1), F.softmax(Tl[idx] / temp, 1), reduction="batchmean")
                loss = (1 - lam) * loss + lam * temp * temp * kd
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = F.cross_entropy(net(Xv), yv).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return net


class Teacher(nn.Module):
    """DeepLOB-style: temporal convolutions over the window, then a GRU; ~25k parameters."""
    def __init__(self, c_in, ch=32, hid=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c_in, ch, 3, padding=1), nn.LeakyReLU(0.01),
                                  nn.Conv1d(ch, ch, 3, padding=1), nn.LeakyReLU(0.01))
        self.gru = nn.GRU(ch, hid, batch_first=True)
        self.out = nn.Linear(hid, 3)

    def forward(self, x):                            # x: (B, T, C)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, hn = self.gru(h)
        return self.out(hn[-1])


def windows(S, idx):
    return S[torch.as_tensor(idx)[:, None] - torch.arange(T_WIN - 1, -1, -1)[None, :]]


def valid_windows(sess):
    j = np.arange(len(sess))
    ok = j >= T_WIN - 1
    ok[T_WIN - 1:] &= sess[T_WIN - 1:] == sess[:len(sess) - T_WIN + 1]
    return ok


def train_teacher(S, y, tr_idx, va_idx, epochs, bs=512):
    torch.manual_seed(SEED)
    net = Teacher(S.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    yt = torch.from_numpy(y).long()
    best, bad, state = np.inf, 0, None
    for _ in range(epochs):
        net.train()
        perm = np.random.default_rng(SEED).permutation(tr_idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(net(windows(S, b)), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        vl = F.cross_entropy(teacher_logits(net, S, va_idx), yt[va_idx]).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= 2:
                break
    net.load_state_dict(state)
    return net.eval()


@torch.no_grad()
def teacher_logits(net, S, idx, bs=4096):
    net.eval()
    return torch.cat([net(windows(S, idx[i:i + bs])) for i in range(0, len(idx), bs)]) if len(idx) else torch.zeros(0, 3)


@nb.njit(cache=True)
def _loop_qmlp(XQ, W1, b1, m1, r1, W2, b2, m2, r2, W3, b3, h1, h2, lg):
    s = 0
    for i in range(XQ.shape[0]):
        s += tm.qmlp_predict(XQ[i], W1, b1, m1, r1, W2, b2, m2, r2, W3, b3, h1, h2, lg)
    return s


def model_ns(QM, Xraw):
    XQ = np.clip(np.floor((Xraw - QM["mu"]) * QM["a"] + 0.5), -127, 127).astype(np.int64)
    W1, W2, W3 = QM["P"][0], QM["P"][4], QM["P"][8]
    bufs = (np.zeros(W1.shape[0], np.int64), np.zeros(W2.shape[0], np.int64), np.zeros(W3.shape[0], np.int64))
    _loop_qmlp(XQ[:100], *QM["P"], *bufs)
    best = np.inf
    for _ in range(3):
        t0 = time.perf_counter_ns(); _loop_qmlp(XQ, *QM["P"], *bufs); best = min(best, time.perf_counter_ns() - t0)
    return best / len(XQ)


def mf1(y, p):
    return f1_score(y, p, average="macro")


def int8_eval(net, mu, sd, Xcal_raw, Xte_raw, yte, wbits=8):
    QM = tm.quantize_mlp(net.weights(), mu, sd, Xcal_raw, wbits=wbits)
    lg, p = tm.qmlp_forward(QM, Xte_raw)
    return QM, lg, p, mf1(yte, p), accuracy_score(yte, p)


# ----------------------------------------------------------------------------- per market
def run_market(name, spec, quick):
    t0 = time.time()
    train_S, test_S, tick = spec[:3]
    Xa, Da, Ba, Sa, Ia = build(train_S, tick)
    Xb, Db, Bb, Sb, Ib = build(test_S, tick)
    cap = 120_000 if quick else 400_000
    if len(Xa) > cap:
        Xa, Da, Ba, Sa, Ia = (v[-cap:] for v in (Xa, Da, Ba, Sa, Ia))
    n = len(Xa)
    cut = int(0.85 * n); es = slice(cut, cut + (n - cut) // 2); cal = slice(cut + (n - cut) // 2, n)
    alpha = float(np.quantile(np.abs(Da[:cut, 0]), 1 / 3))
    lab = lambda d: 1 + (d > alpha).astype(int) - (d < -alpha).astype(int)
    ya, yb = lab(Da[:, 0]), lab(Db[:, 0])
    mu, sd = Xa[:cut].mean(0), Xa[:cut].std(0) + 1e-9
    Za, Zb = (Xa - mu) / sd, (Xb - mu) / sd
    rows = {}

    # ---- student, plain
    stu = train_student(Za[:cut], ya[:cut], Za[es], ya[es])
    QM, lg_s, p_s, f1_s, acc_s = int8_eval(stu, mu, sd, Xa[:cut], Xb, yb)

    # ---- teacher over windows of z-scored features + book shape
    bmu, bsd = Ba[:cut].mean(0), Ba[:cut].std(0) + 1e-9
    seq_a = np.clip(np.c_[Za, (Ba - bmu) / bsd], -10, 10).astype(np.float32)
    seq_b = np.clip(np.c_[Zb, (Bb - bmu) / bsd], -10, 10).astype(np.float32)
    Sa_t, Sb_t = torch.from_numpy(seq_a), torch.from_numpy(seq_b)
    okA, okB = valid_windows(Ia), valid_windows(Ib)
    tr_idx = np.flatnonzero(okA[:cut])
    tr_idx = tr_idx[-(40_000 if quick else 200_000):]
    es_idx = np.flatnonzero(okA)[(np.flatnonzero(okA) >= es.start) & (np.flatnonzero(okA) < es.stop)]
    t1 = time.time()
    teacher = train_teacher(Sa_t, ya, tr_idx, es_idx, epochs=2 if quick else 6)
    t_train = time.time() - t1
    te_idx = np.flatnonzero(okB)
    lt_b = teacher_logits(teacher, Sb_t, te_idx).numpy()
    p_t = lt_b.argmax(1)
    f1_t, acc_t = mf1(yb[te_idx], p_t), accuracy_score(yb[te_idx], p_t)
    n_params_t = sum(p.numel() for p in teacher.parameters())
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)                         # single-sample latency, single thread, like the student
    one = windows(Sb_t, te_idx[:1])
    with torch.no_grad():
        for _ in range(20): teacher(one)
        lat = []
        for k in range(200):
            w = windows(Sb_t, te_idx[k:k + 1]); s0 = time.perf_counter_ns(); teacher(w); lat.append(time.perf_counter_ns() - s0)
    torch.set_num_threads(n_threads)
    t_lat_us = float(np.percentile(lat, 50)) / 1000

    # ---- distilled student: teacher logits on every training row that has a full window
    all_idx = np.flatnonzero(okA)
    lt_a = np.zeros((n, 3), np.float32)
    lt_a[all_idx] = teacher_logits(teacher, Sa_t, all_idx).numpy()
    has = np.zeros(n, bool); has[all_idx] = True
    tr_kd = np.flatnonzero(has[:cut])
    kd = train_student(Za[tr_kd], ya[tr_kd], Za[es], ya[es], t_logits=lt_a[tr_kd])
    QMk, lg_k, p_k, f1_k, acc_k = int8_eval(kd, mu, sd, Xa[:cut], Xb, yb)
    rows["teacher"] = [
        {"market": name, "model": "teacher: DeepLOB-style CNN+GRU, fp32", "params": n_params_t, "bytes": 4 * n_params_t,
         "macro_f1": f1_t, "accuracy": acc_t, "latency_us_p50": t_lat_us, "note": f"trained {t_train:.0f}s, {len(tr_idx):,} windows"},
        {"market": name, "model": "student: int8 MLP", "params": sum(w.size for w in stu.weights()), "bytes": sum(v for k, v in tm.model_bytes(QM).items() if k != "macs"),
         "macro_f1": mf1(yb[te_idx], p_s[te_idx]), "accuracy": accuracy_score(yb[te_idx], p_s[te_idx]), "latency_us_p50": model_ns(QM, Xb[:50000]) / 1000, "note": "same test rows as teacher"},
        {"market": name, "model": "student distilled: int8 MLP", "params": sum(w.size for w in kd.weights()), "bytes": sum(v for k, v in tm.model_bytes(QMk).items() if k != "macs"),
         "macro_f1": mf1(yb[te_idx], p_k[te_idx]), "accuracy": accuracy_score(yb[te_idx], p_k[te_idx]), "latency_us_p50": model_ns(QMk, Xb[:50000]) / 1000, "note": "same test rows as teacher"}]
    best_net, best_QM, best_lg, best_p = (kd, QMk, lg_k, p_k) if f1_k >= f1_s else (stu, QM, lg_s, p_s)

    # ---- QAT vs PTQ at 8 and 4 bits (fine-tune the best student with fake-quantised weights)
    rows["qat"] = []
    for bits in (8, 4):
        _, _, _, f1_ptq, _ = int8_eval(best_net, mu, sd, Xa[:cut], Xb, yb, wbits=bits)
        qat = train_student(Za[:cut], ya[:cut], Za[es], ya[es], wbits=bits, init=best_net, epochs=3 if quick else 8, lr=1e-3)
        _, _, _, f1_qat, _ = int8_eval(qat, mu, sd, Xa[:cut], Xb, yb, wbits=bits)
        rows["qat"].append({"market": name, "weight_bits": bits, "float_f1": f1_k if best_net is kd else f1_s,
                            "PTQ_f1": f1_ptq, "QAT_f1": f1_qat})

    # ---- permutation importance on the calibration set (deployed integer model)
    rng = np.random.default_rng(SEED)
    Xc, yc = Xa[cal], ya[cal]
    base = mf1(yc, tm.qmlp_forward(best_QM, Xc)[1])
    imp = []
    for j in range(tm.NF):
        Xp = Xc.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp.append(base - mf1(yc, tm.qmlp_forward(best_QM, Xp)[1]))
    rows["importance"] = [{"market": name, "feature": tm.FEATURES[j], "f1_drop": imp[j]} for j in range(tm.NF)]
    order = list(np.argsort(imp)[::-1])

    # ---- Pareto: width x number of features (top-k by importance), plain cross-entropy
    rows["pareto"] = []
    widths = (4, 16) if quick else (4, 8, 16, 32)
    for kf in ((6, tm.NF) if quick else (6, 10, tm.NF)):
        cols = sorted(order[:kf])
        for wd in widths:
            net = train_student(Za[:cut][:, cols], ya[:cut], Za[es][:, cols], ya[es], hidden=(wd, wd), epochs=25)
            Qp = tm.quantize_mlp(net.weights(), mu[cols], sd[cols], Xa[:cut][:, cols])
            pp = tm.qmlp_forward(Qp, Xb[:, cols])[1]
            mb = tm.model_bytes(Qp)
            rows["pareto"].append({"market": name, "features": kf, "width": wd, "macs": mb["macs"],
                                   "bytes": sum(v for k, v in mb.items() if k != "macs"), "model_ns": model_ns(Qp, Xb[:50000][:, cols]),
                                   "macro_f1": mf1(yb, pp), "feature_set": ",".join(tm.FEATURES[c] for c in cols)})

    # ---- E: research-only decision value of the deployed model's confident calls
    prob = tm.softmax(best_lg * best_QM["out_scale"])
    conf, pred = prob.max(1), best_p
    dirn = np.where(pred == 2, 1.0, np.where(pred == 0, -1.0, 0.0))
    dmask = dirn != 0
    rows["sim"], rows["delay"] = [], []
    for q in (0.0, 0.5, 0.75, 0.9, 0.95, 0.99):
        tau = np.quantile(conf[dmask], q) if dmask.any() else 1.0
        m = dmask & (conf >= tau)
        if m.sum() < 20:
            continue
        gross = dirn[m] * Db[m, 0]
        rows["sim"].append({"market": name, "kept_top": f"{100 * (1 - q):.0f}% of directional", "share_of_ticks": m.mean(),
                            "calls": int(m.sum()), "hit_rate": (pred[m] == yb[m]).mean(), "gross_edge_ticks": gross.mean(),
                            "spread_ticks": Sb[m].mean(), "net_after_spread_ticks": (gross - Sb[m]).mean(),
                            "net_after_half_spread_ticks": (gross - 0.5 * Sb[m]).mean()})
    m = dmask & (conf >= np.quantile(conf[dmask], 0.9))
    for j, L in enumerate(DELAYS):
        ok = m & np.isfinite(Db[:, j])
        yL = lab(Db[ok, j])
        rows["delay"].append({"market": name, "delay_events": L, "gross_edge_ticks": float(np.mean(dirn[ok] * Db[ok, j])),
                              "accuracy": (pred[ok] == yL).mean(), "calls": int(ok.sum())})
    print(f"{name:45s} {time.time() - t0:6.0f}s  student F1 {f1_s:.3f} | distilled {f1_k:.3f} | teacher {f1_t:.3f}")
    return rows


def plots(T):
    names = list(dict.fromkeys(T["pareto"].market))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c_, nm in zip(COLORS, names):
        d = T["pareto"][T["pareto"].market == nm].sort_values("bytes")
        ax.plot(d.bytes, d.macro_f1, "o", color=c_, ms=6, alpha=0.5)
        fr = d[d.macro_f1 >= d.macro_f1.cummax()]          # Pareto frontier: no smaller model is better
        ax.plot(fr.bytes, fr.macro_f1, "-o", color=c_, ms=7, label=nm.split(" (")[0], markeredgecolor="white")
        t = T["teacher"][(T["teacher"].market == nm) & T["teacher"].model.str.startswith("teacher")]
        ax.plot(t.bytes, t.macro_f1, "*", color=c_, ms=16, markeredgecolor="white")
    ax.set_xscale("log")
    ax.set(xlabel="deployed model size, bytes (log)", ylabel="macro-F1, k=10, test days",
           title="Accuracy vs size: tiny int8 students (lines) and the deep teacher (stars)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(OUT / "pareto.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for c_, nm in zip(COLORS, names):
        d = T["delay"][T["delay"].market == nm]
        base = d[d.delay_events == 0].gross_edge_ticks.iloc[0]
        ax.plot(d.delay_events, 100 * d.gross_edge_ticks / base, "-o", color=c_, label=nm.split(" (")[0], markeredgecolor="white")
    ax.axhline(0, color="#888", lw=1)
    ax.set(xlabel="prediction arrives this many book events late", ylabel="edge kept (% of on-time edge)",
           title="Why latency matters: edge of the top-10% confident calls vs delay")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "latency_decay.png", dpi=120); plt.close()

    imp = T["importance"].pivot_table(index="feature", columns="market", values="f1_drop").reindex(tm.FEATURES)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(imp.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_yticks(range(len(imp)), imp.index); ax.set_xticks(range(imp.shape[1]), [c.split(" (")[0] for c in imp.columns], rotation=15, ha="right")
    for (i, j), v in np.ndenumerate(imp.values):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="#111" if v < imp.values.max() * 0.6 else "white")
    ax.set_title("Permutation importance: macro-F1 lost when a feature is shuffled")
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout(); plt.savefig(OUT / "importance.png", dpi=120); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--markets", nargs="*", help="substring filter on market names")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    M = mkt.load_markets(include_sim=False)
    if a.markets:
        M = {k: v for k, v in M.items() if any(s.lower() in k.lower() for s in a.markets)}
    allrows = {}
    for name, spec in M.items():
        for k, v in run_market(name, spec, a.quick).items():
            allrows.setdefault(k, []).extend(v)
        T = {k: pd.DataFrame(v) for k, v in allrows.items()}
        for k, d in T.items():
            d.to_csv(OUT / f"{k}.csv", index=False)              # written after every market, so partial runs keep results
        plots(T)
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))
