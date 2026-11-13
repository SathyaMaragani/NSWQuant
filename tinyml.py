"""TinyML order-book core shared by the notebook, the C export and the live NIFTY script.

* streaming microstructure features (one implementation for training and serving)
* future-only labels, the order-book simulator
* integer-only int8 MLP: int8 inputs/activations, int32 accumulators, fixed-point requantisation
  (no floating point after the feature step), allocation-free numba kernels
* export of the deployed model as a dependency-free C header
"""
import numpy as np
import numba as nb

LEVELS = 5
BOOK_COLS = ([f"bp{i}" for i in range(1, 6)] + [f"bq{i}" for i in range(1, 6)]
             + [f"ap{i}" for i in range(1, 6)] + [f"aq{i}" for i in range(1, 6)] + ["volume", "tbq", "tsq"])
FEATURES = ["spread", "micro", "imb1", "imb3", "imb5", "ofi1", "ofi5", "ofi20",
            "ret1", "ret5", "ret20", "rvol20", "vold", "tbq_imb", "since_move",
            "ofi_deep", "ofi_deep5", "svol5"]
NF = len(FEATURES)
W, MW, W5 = 20, 21, 5          # OFI ring, mid ring, short rings
# state layout: [prev bp1,bq1,ap1,aq1,volume, since, t, prev mid | OFI ring | mid ring | prev levels 2-5 | deep-OFI ring | signed-volume ring]
OR = 8
MR = OR + W
PL = MR + MW                   # prev bp2..5, bq2..5, ap2..5, aq2..5 (16)
DR = PL + 16                   # deep (levels 2-5) OFI ring
VR = DR + W5                   # tick-rule signed volume ring
ST_SIZE = VR + W5


# ----------------------------------------------------------------------------- simulator
@nb.njit(cache=True)
def _simulate(n, seed, tick):
    np.random.seed(seed)
    L, LOT = 5, 75.0
    out = np.empty((n, 23))
    qb = np.empty(L); qa = np.empty(L)
    for i in range(L):
        qb[i] = LOT * np.random.randint(5, 40); qa[i] = LOT * np.random.randint(5, 40)
    bid = 480000; ask = bid + 1            # prices in ticks
    alpha = 0.0; vol = 0.0
    for t in range(n):
        for _ in range(np.random.randint(1, 6)):
            alpha = 0.97 * alpha + 0.25 * np.random.randn()
            buy = np.random.rand() < 0.5 + 0.35 * np.tanh(alpha)   # hidden buy pressure
            size = LOT * np.random.geometric(0.3)
            u = np.random.rand()
            if ask - bid > 1 and np.random.rand() < 0.6:           # gap after a touch emptied: refilled fast,
                if buy:                                              # by the side the pressure favours
                    for i in range(L - 1, 0, -1): qb[i] = qb[i - 1]
                    bid += 1; qb[0] = size
                else:
                    for i in range(L - 1, 0, -1): qa[i] = qa[i - 1]
                    ask -= 1; qa[0] = size
            elif u < 0.35:                                           # market order
                vol += size
                if buy: qa[0] -= size
                else: qb[0] -= size
            elif u < 0.75:                                           # limit add
                lvl = min(np.random.geometric(0.5) - 1, L - 1)
                if lvl == 0 and ask - bid > 1 and np.random.rand() < 0.5:
                    if buy:
                        for i in range(L - 1, 0, -1): qb[i] = qb[i - 1]
                        bid += 1; qb[0] = size
                    else:
                        for i in range(L - 1, 0, -1): qa[i] = qa[i - 1]
                        ask -= 1; qa[0] = size
                elif buy: qb[lvl] += size
                else: qa[lvl] += size
            else:                                                    # cancel on the side pressure is against
                lvl = min(np.random.geometric(0.5) - 1, L - 1)
                if buy: qa[lvl] = max(qa[lvl] - size, 0.0)
                else: qb[lvl] = max(qb[lvl] - size, 0.0)
            while qa[0] <= 0:                                        # ask touch emptied -> walk up
                left = -qa[0]
                for i in range(L - 1): qa[i] = qa[i + 1]
                qa[L - 1] = LOT * np.random.randint(5, 40)
                ask += 1; qa[0] -= left
            while qb[0] <= 0:                                        # bid touch emptied -> walk down
                left = -qb[0]
                for i in range(L - 1): qb[i] = qb[i + 1]
                qb[L - 1] = LOT * np.random.randint(5, 40)
                bid -= 1; qb[0] -= left
        for i in range(L):
            out[t, i] = (bid - i) * tick; out[t, 5 + i] = qb[i]
            out[t, 10 + i] = (ask + i) * tick; out[t, 15 + i] = qa[i]
        out[t, 20] = vol
        out[t, 21] = 3.0 * qb.sum() + LOT * np.random.randint(0, 400)   # total pending buy qty (whole book)
        out[t, 22] = 3.0 * qa.sum() + LOT * np.random.randint(0, 400)
    return out


def simulate_lob(sessions=4, n=60_000, tick=0.05, seed=0):
    import pandas as pd
    parts = [pd.DataFrame(_simulate(n, seed + s, tick), columns=BOOK_COLS).assign(session=s) for s in range(sessions)]
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- features and labels
@nb.njit(cache=True)
def feat_update(st, r, tick, out):
    """r = bp1..5, bq1..5, ap1..5, aq1..5, volume, tbq, tsq.  Writes NF features into out. O(1) per tick."""
    bp1, bq1, ap1, aq1 = r[0], r[5], r[10], r[15]
    t = int(st[6])
    mid = 0.5 * (bp1 + ap1)
    out[0] = (ap1 - bp1) / tick
    out[1] = ((ap1 * bq1 + bp1 * aq1) / (bq1 + aq1) - mid) / tick
    out[2] = (bq1 - aq1) / (bq1 + aq1)
    sb = 0.0; sa = 0.0
    for i in range(3):
        sb += r[5 + i]; sa += r[15 + i]
    out[3] = (sb - sa) / (sb + sa)
    for i in range(3, 5):
        sb += r[5 + i]; sa += r[15 + i]
    out[4] = (sb - sa) / (sb + sa)
    e = 0.0; dv = 0.0
    if t > 0:
        pbp, pbq, pap, paq = st[0], st[1], st[2], st[3]
        if bp1 >= pbp: e += bq1
        if bp1 <= pbp: e -= pbq
        if ap1 <= pap: e -= aq1
        if ap1 >= pap: e += paq
        dv = r[20] - st[4]
    st[OR + t % W] = e
    s5 = 0.0; s20 = 0.0
    for k in range(min(t + 1, W)):
        v = st[OR + (t - k) % W]
        s20 += v
        if k < 5: s5 += v
    out[5] = e; out[6] = s5; out[7] = s20
    st[MR + t % MW] = mid
    out[8] = (mid - st[MR + (t - 1) % MW]) / tick if t >= 1 else 0.0
    out[9] = (mid - st[MR + (t - 5) % MW]) / tick if t >= 5 else 0.0
    out[10] = (mid - st[MR + (t - 20) % MW]) / tick if t >= 20 else 0.0
    c = min(t + 1, W); s = 0.0; ss = 0.0
    for k in range(c):
        u = t - k
        d = 0.0 if u == 0 else (st[MR + u % MW] - st[MR + (u - 1) % MW]) / tick
        s += d; ss += d * d
    out[11] = np.sqrt(max((ss - s * s / c) / (c - 1), 0.0)) if c > 1 else 0.0
    out[12] = np.log1p(dv) if dv > 0 else 0.0
    tq = r[21] + r[22]
    out[13] = (r[21] - r[22]) / tq if tq > 0 else 0.0
    since = 0.0 if (t == 0 or mid != st[7]) else st[5] + 1.0
    out[14] = np.log1p(min(since, 100.0))
    # multi-level OFI: the same flow measure at levels 2-5, now and summed over the last 5 events
    ed = 0.0
    if t > 0:
        for i in range(1, 5):
            if r[i] >= st[PL + i - 1]: ed += r[5 + i]
            if r[i] <= st[PL + i - 1]: ed -= st[PL + 4 + i - 1]
            if r[10 + i] <= st[PL + 8 + i - 1]: ed -= r[15 + i]
            if r[10 + i] >= st[PL + 8 + i - 1]: ed += st[PL + 12 + i - 1]
    st[DR + t % W5] = ed
    sd5 = 0.0
    for k in range(min(t + 1, W5)):
        sd5 += st[DR + (t - k) % W5]
    out[15] = ed; out[16] = sd5
    # tick-rule signed volume: traded volume signed by the direction the mid moved, last 5 events
    sv = 0.0
    if t > 0 and dv > 0:
        if mid > st[7]: sv = dv
        elif mid < st[7]: sv = -dv
    st[VR + t % W5] = sv
    vs = 0.0
    for k in range(min(t + 1, W5)):
        vs += st[VR + (t - k) % W5]
    out[17] = np.log1p(vs) if vs > 0 else (-np.log1p(-vs) if vs < 0 else 0.0)
    for i in range(1, 5):
        st[PL + i - 1] = r[i]; st[PL + 4 + i - 1] = r[5 + i]; st[PL + 8 + i - 1] = r[10 + i]; st[PL + 12 + i - 1] = r[15 + i]
    st[0] = bp1; st[1] = bq1; st[2] = ap1; st[3] = aq1; st[4] = r[20]
    st[5] = since; st[6] = t + 1; st[7] = mid


@nb.njit(cache=True)
def run_features(R, tick):
    st = np.zeros(ST_SIZE)
    X = np.empty((R.shape[0], NF))
    for t in range(R.shape[0]):
        feat_update(st, R[t], tick, X[t])
    return X


def future_move(m, h):
    """Mean mid over the next h updates minus the current mid (ticks). NaN where the future is unknown."""
    c = np.concatenate([[0.0], np.cumsum(m)])
    d = np.full(len(m), np.nan)
    n = len(m) - h
    if n > 0:
        d[:n] = (c[h + 1:h + 1 + n] - c[1:1 + n]) / h - m[:n]
    return d


def clean_book(df):
    """Drop invalid books (zero, one-sided, crossed or locked) and snapshots identical to the previous one."""
    ok = (df.bp1 > 0) & (df.ap1 > 0) & (df.ap1 > df.bp1) & (df.bq1 > 0) & (df.aq1 > 0)
    df = df[ok]
    same = df[BOOK_COLS].diff().abs().sum(axis=1) == 0
    if "session" in df:
        same &= df.session == df.session.shift()
    return df[~same].reset_index(drop=True)


def train_mlp(Xtr, ytr, Xva, yva, hidden=(8, 8), epochs=40, patience=4, bs=1024, seed=0, lr=3e-3):
    """Small ReLU MLP with Adam and early stopping on validation loss -> [W1, b1, W2, b2, W3, b3] (float32, W as (out, in))."""
    import copy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    dims = [Xtr.shape[1], *hidden]
    layers = []
    for a_, b_ in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a_, b_), nn.ReLU()]
    net = nn.Sequential(*layers, nn.Linear(dims[-1], 3))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(np.asarray(Xtr, np.float32)), torch.from_numpy(np.asarray(ytr)).long()
    Xv, yv = torch.from_numpy(np.asarray(Xva, np.float32)), torch.from_numpy(np.asarray(yva)).long()
    best, bad, state = np.inf, 0, None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); F.cross_entropy(net(Xt[idx]), yt[idx]).backward(); opt.step()
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
    return [p.detach().numpy().astype(np.float32).copy() for p in net.parameters()]


# ----------------------------------------------------------------------------- integer-only int8 MLP
def _fixed_point(real):
    """real > 0 -> (q, rshift) with real ~= q / 2**rshift, q a Q31 int (gemmlowp / TFLite-Micro style)."""
    m, e = np.frexp(np.asarray(real, dtype=np.float64))          # real = m * 2**e, m in [0.5, 1)
    q = np.round(m * (1 << 31)).astype(np.int64)
    over = q == (1 << 31)
    q[over] //= 2; e[over] += 1
    rshift = (31 - e).astype(np.int64)
    tiny = rshift > 62                    # multiplier < 2^-31 (a near-dead neuron): its output rounds to 0 anyway
    q[tiny], rshift[tiny] = 0, 1
    assert (rshift > 0).all(), "requantisation multiplier >= 2^31"
    return q, rshift


def _round_half_up(x):
    return np.floor(x + 0.5)


def quantize_mlp(ws, mu, sd, X_cal, wbits=8):
    """Float MLP (W1,b1,W2,b2,W3,b3 with W as (out, in)) + raw-feature normalisation -> integer-only model.

    Inputs:  xq = clip(round((x - mu) * a), -127, 127), a = 1 / (sd * s_in)       (one float step)
    Hidden:  acc = b_q + sum(W_q * x_q) in int32; ReLU; h_q = clip((acc * q + 2^(r-1)) >> r, 0, 127)
    Output:  int32 logits with one scale for all classes, so argmax needs no rescaling.
    Weights: int8, per-output-channel scales in hidden layers. Activation ranges from calibration data."""
    Z = (X_cal - mu) / sd
    s_in = float(np.percentile(np.abs(Z), 99.9) / 127.0)
    # input scaling is stored as float32 (C header) but computed in double on exactly those values
    mu = np.asarray(mu, np.float32).astype(np.float64)
    a = (1.0 / (sd * s_in)).astype(np.float32).astype(np.float64)
    xq = np.clip(_round_half_up((X_cal - mu) * a), -127, 127).astype(np.int64)
    s_x, P = s_in, []
    qmax = 2 ** (wbits - 1) - 1                       # 127 for int8 weights, 7 for int4 (stored in int8 here)
    pairs = list(zip(ws[0::2], ws[1::2]))
    for li, (Wf, bf) in enumerate(pairs):
        Wf, bf = np.asarray(Wf, np.float64), np.asarray(bf, np.float64)
        last = li == len(pairs) - 1
        s_w = (np.full(Wf.shape[0], np.abs(Wf).max()) if last else np.abs(Wf).max(1)) / qmax
        s_w = np.where(s_w > 0, s_w, 1e-12)
        Wq = np.clip(_round_half_up(Wf / s_w[:, None]), -qmax, qmax).astype(np.int8)
        bq64 = _round_half_up(bf / (s_w * s_x))
        assert np.abs(bq64).max() < 2 ** 30, "bias too large for int32 accumulation"
        bq = bq64.astype(np.int32)
        acc = xq @ Wq.T.astype(np.int64) + bq
        if last:
            P += [Wq, bq]
            out_scale = float(s_w[0] * s_x)
            break
        y = np.maximum(acc, 0) * (s_w * s_x)
        s_out = float(np.percentile(y, 99.99) / 127.0) or 1e-12
        q, r = _fixed_point(s_w * s_x / s_out)
        P += [Wq, bq, q, r]
        xq = np.clip((np.maximum(acc, 0) * q + (1 << (r - 1))) >> r, 0, 127)
        s_x = s_out
    return {"mu": np.asarray(mu, np.float64), "a": a, "s_in": s_in, "out_scale": out_scale, "P": tuple(P), "wbits": wbits}


def qmlp_forward(QM, X_raw):
    """Vectorised reference of the integer pipeline -> (int32 logits, class). Same arithmetic as the kernels."""
    W1, b1, m1, r1, W2, b2, m2, r2, W3, b3 = QM["P"]
    x = np.clip(_round_half_up((X_raw - QM["mu"]) * QM["a"]), -127, 127).astype(np.int64)
    for Wq, bq, q, r in ((W1, b1, m1, r1), (W2, b2, m2, r2)):
        acc = np.maximum(x @ Wq.T.astype(np.int64) + bq, 0)
        x = np.clip((acc * q + (1 << (r - 1))) >> r, 0, 127)
    logits = x @ W3.T.astype(np.int64) + b3
    return logits, logits.argmax(1)


@nb.njit(cache=True)
def _dense_relu_q(x, Wq, bq, q, r, out):
    for i in range(Wq.shape[0]):
        acc = np.int64(bq[i])
        for j in range(Wq.shape[1]):
            acc += np.int64(Wq[i, j]) * np.int64(x[j])
        if acc < 0:
            acc = 0
        v = (acc * q[i] + (np.int64(1) << (r[i] - 1))) >> r[i]
        out[i] = 127 if v > 127 else v


@nb.njit(cache=True)
def qmlp_predict(xq, W1, b1, m1, r1, W2, b2, m2, r2, W3, b3, h1, h2, logits):
    """Integer-only forward pass into preallocated buffers (no allocation per call). Returns the class."""
    _dense_relu_q(xq, W1, b1, m1, r1, h1)
    _dense_relu_q(h1, W2, b2, m2, r2, h2)
    best = 0
    for i in range(W3.shape[0]):
        acc = np.int64(b3[i])
        for j in range(W3.shape[1]):
            acc += np.int64(W3[i, j]) * np.int64(h2[j])
        logits[i] = acc
        if acc > logits[best]:
            best = i
    return best


@nb.njit(cache=True)
def quantize_input(x, mu, a, xq):
    for j in range(x.shape[0]):
        v = np.floor((x[j] - mu[j]) * a[j] + 0.5)
        xq[j] = -127 if v < -127 else (127 if v > 127 else v)


@nb.njit(cache=True)
def tick_predict(st, r, tick, mu, a, xb, xq, W1, b1, m1, r1, W2, b2, m2, r2, W3, b3, h1, h2, logits):
    """Full deployed path for one book update: features -> int8 input -> integer MLP -> class."""
    feat_update(st, r, tick, xb)
    quantize_input(xb, mu, a, xq)
    return qmlp_predict(xq, W1, b1, m1, r1, W2, b2, m2, r2, W3, b3, h1, h2, logits)


@nb.njit(cache=True)
def run_pipeline(R, tick, mu, a, W1, b1, m1, r1, W2, b2, m2, r2, W3, b3):
    """Stream rows through tick_predict, returning every class (for parity checks and in-kernel timing)."""
    st = np.zeros(ST_SIZE); xb = np.empty(mu.shape[0])
    xq = np.zeros(mu.shape[0], np.int64); h1 = np.zeros(W1.shape[0], np.int64)
    h2 = np.zeros(W2.shape[0], np.int64); lg = np.zeros(W3.shape[0], np.int64)
    out = np.empty(R.shape[0], np.int8)
    for t in range(R.shape[0]):
        out[t] = tick_predict(st, R[t], tick, mu, a, xb, xq, W1, b1, m1, r1, W2, b2, m2, r2, W3, b3, h1, h2, lg)
    return out


def buffers(QM):
    """Preallocated per-stream buffers for tick_predict."""
    W1, _, _, _, W2, _, _, _, W3, _ = QM["P"]
    n = len(QM["mu"])
    return (np.zeros(ST_SIZE), np.empty(n), np.zeros(n, np.int64), np.zeros(W1.shape[0], np.int64),
            np.zeros(W2.shape[0], np.int64), np.zeros(W3.shape[0], np.int64))


def model_bytes(QM):
    """Bytes the deployed model occupies as stored in the C header (weights int8, biases/multipliers int32, shifts int8)."""
    W1, b1, m1, r1, W2, b2, m2, r2, W3, b3 = QM["P"]
    w = W1.size + W2.size + W3.size
    return {"weights_int8": int(np.ceil(w * QM.get("wbits", 8) / 8)), "bias_int32": 4 * (b1.size + b2.size + b3.size),
            "requant_int32+shift": 5 * (m1.size + m2.size), "input_scaling_f32": 8 * len(QM["mu"]),
            "macs": w}


# ----------------------------------------------------------------------------- reliability
def softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    """Temperature T > 0 minimising validation log-loss of softmax(logits / T)."""
    from scipy.optimize import minimize_scalar
    logits = np.asarray(logits, np.float64)
    idx = np.arange(len(y))

    def nll(logT):
        p = softmax(logits / np.exp(logT))
        return -np.log(np.clip(p[idx, y], 1e-12, None)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))


def ece(p, y, bins=15):
    """Expected calibration error of the top-class probability (equal-width bins)."""
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(e)


def conformal_qhat(p_cal, y_cal, alpha=0.1):
    """Split-conformal (LAC) threshold: sets {k : p_k >= 1 - qhat} cover the true class with prob >= 1 - alpha."""
    s = np.sort(1.0 - p_cal[np.arange(len(y_cal)), y_cal])
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])


def guaranteed_threshold(conf, correct, eps=0.30, delta=0.05, grid=None, min_n=200):
    """Smallest confidence threshold tau such that directional calls with confidence >= tau have error <= eps,
    certified on calibration data by Clopper-Pearson bounds tested in a fixed sequence from high to low tau
    (Learn-then-Test style: the whole procedure holds with probability >= 1 - delta). None if no tau passes."""
    from scipy.stats import beta
    grid = np.unique(np.quantile(conf, np.linspace(0.99, 0.0, 100))) if grid is None else np.asarray(grid)
    best = None
    for tau in grid[::-1]:                                 # highest confidence first
        acc = conf >= tau
        n = int(acc.sum())
        if n < min_n:                                      # too few calls to test: start lower (no test spent)
            continue
        k = int((~correct[acc]).sum())
        ucb = 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))
        if ucb <= eps:
            best = float(tau)
        else:
            break
    return best


def psi(ref, cur, bins=10):
    """Population stability index of one feature: ref (training) vs cur (live/test). >0.25 = large shift."""
    q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return 0.0
    a = np.histogram(np.clip(ref, q[0], q[-1]), q)[0] / len(ref) + 1e-6
    b = np.histogram(np.clip(cur, q[0], q[-1]), q)[0] / len(cur) + 1e-6
    return float(((b - a) * np.log(b / a)).sum())


# ----------------------------------------------------------------------------- C export
def _carr(name, arr, ctype):
    flat = np.asarray(arr).ravel()
    if ctype in ("double", "float"):
        body = ", ".join(repr(float(v)) for v in flat)   # float32 values print exactly; the literal rounds back to them
    else:
        body = ", ".join(str(int(v)) for v in flat)
    return f"static const {ctype} {name}[{flat.size}] = {{{body}}};\n"


def export_c(QM, tick, path, note=""):
    """Write the deployed model as a single dependency-free C header (int8 weights, integer-only inference)."""
    W1, b1, m1, r1, W2, b2, m2, r2, W3, b3 = QM["P"]
    nf, h1, h2, nc = W1.shape[1], W1.shape[0], W2.shape[0], W3.shape[0]
    s = [f"/* tinyml_model.h — generated by tinyml.export_c. {note}\n"
         f"   Integer-only MLP {nf}->{h1}->{h2}->{nc}: int8 weights/activations, int32 accumulators,\n"
         f"   fixed-point requantisation. Classes: 0 = down, 1 = flat, 2 = up. */\n",
         "#pragma once\n#include <stdint.h>\n#include <math.h>\n\n",
         f"#define TM_NF {nf}\n#define TM_H1 {h1}\n#define TM_H2 {h2}\n#define TM_NC {nc}\n",
         f"#define TM_TICK {float(tick)!r}\n\n",
         _carr("TM_MU", QM["mu"], "float"), _carr("TM_A", QM["a"], "float"),
         _carr("TM_W1", W1, "int8_t"), _carr("TM_B1", b1, "int32_t"), _carr("TM_M1", m1, "int32_t"), _carr("TM_R1", r1, "int8_t"),
         _carr("TM_W2", W2, "int8_t"), _carr("TM_B2", b2, "int32_t"), _carr("TM_M2", m2, "int32_t"), _carr("TM_R2", r2, "int8_t"),
         _carr("TM_W3", W3, "int8_t"), _carr("TM_B3", b3, "int32_t"),
         r"""
/* int8 x int8 products summed in int32 (|sum| <= n_in*127*127 + |b|, far below 2^31), so compilers can use
   SIMD multiply-add; requantisation multiply in int64. Bit-identical to the Python kernels. */
static inline void tm_dense_relu(const int8_t *x, int n_in, int n_out, const int8_t *W, const int32_t *b,
                                 const int32_t *m, const int8_t *r, int8_t *out) {
    for (int i = 0; i < n_out; i++) {
        int32_t acc = b[i];
        const int8_t *w = W + i * n_in;
        for (int j = 0; j < n_in; j++) acc += (int32_t)w[j] * (int32_t)x[j];
        if (acc < 0) acc = 0;
        int64_t v = ((int64_t)acc * (int64_t)m[i] + ((int64_t)1 << (r[i] - 1))) >> r[i];
        out[i] = (int8_t)(v > 127 ? 127 : v);
    }
}

/* raw features (double[TM_NF]) -> class; logits (int32[TM_NC]) optional */
static inline int tm_predict(const double *x, int32_t *logits) {
    int8_t xq[TM_NF], h1[TM_H1], h2[TM_H2];
    for (int j = 0; j < TM_NF; j++) {
        double v = floor((x[j] - (double)TM_MU[j]) * (double)TM_A[j] + 0.5);
        xq[j] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
    }
    tm_dense_relu(xq, TM_NF, TM_H1, TM_W1, TM_B1, TM_M1, TM_R1, h1);
    tm_dense_relu(h1, TM_H1, TM_H2, TM_W2, TM_B2, TM_M2, TM_R2, h2);
    int best = 0; int32_t bv = 0;
    for (int i = 0; i < TM_NC; i++) {
        int32_t acc = TM_B3[i];
        for (int j = 0; j < TM_H2; j++) acc += (int32_t)TM_W3[i * TM_H2 + j] * (int32_t)h2[j];
        if (logits) logits[i] = acc;
        if (i == 0 || acc > bv) { bv = acc; best = i; }
    }
    return best;
}
"""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(s))


if __name__ == "__main__":
    # self-check: integer kernel == vectorised reference, and quantised accuracy tracks the float model
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, NF)) * rng.uniform(0.5, 3, NF) + rng.normal(size=NF)
    ws = [rng.normal(size=(16, NF)).astype(np.float32) * 0.3, rng.normal(size=16).astype(np.float32) * 0.1,
          rng.normal(size=(16, 16)).astype(np.float32) * 0.3, rng.normal(size=16).astype(np.float32) * 0.1,
          rng.normal(size=(3, 16)).astype(np.float32) * 0.3, rng.normal(size=3).astype(np.float32) * 0.1]
    mu, sd = X.mean(0), X.std(0)
    QM = quantize_mlp(ws, mu, sd, X)
    _, ref = qmlp_forward(QM, X)
    xq = np.zeros(NF, np.int64); h1 = np.zeros(16, np.int64); h2 = np.zeros(16, np.int64); lg = np.zeros(3, np.int64)
    ker = np.array([qmlp_predict((quantize_input(x, QM["mu"], QM["a"], xq), xq)[1], *QM["P"], h1, h2, lg) for x in X])
    assert (ker == ref).all()
    Z = ((X - mu) / sd).astype(np.float32)
    h = np.maximum(Z @ ws[0].T + ws[1], 0); h = np.maximum(h @ ws[2].T + ws[3], 0); fl = (h @ ws[4].T + ws[5]).argmax(1)
    agree = (fl == ref).mean()
    assert agree > 0.97, agree
    print(f"selftest ok: kernel == reference, int8 vs float agreement {agree:.4f}")
