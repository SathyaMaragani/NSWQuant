/* tinyml_features.h — C port of tinyml.feat_update: 15 order-book features, O(1) per book update.
   Same operations in the same order as the Python/numba kernel; compile with -ffp-contract=off so
   results match bit for bit. Row layout: bp1..5, bq1..5, ap1..5, aq1..5, volume, tbq, tsq. */
#pragma once
#include <math.h>

#define TF_NF 18
#define TF_W 20
#define TF_MW 21
#define TF_W5 5
#define TF_OR 8
#define TF_MR 28
#define TF_PL 49
#define TF_DR 65
#define TF_VR 70
#define TF_ST 75   /* state doubles; must start zeroed */

static inline void tf_update(double *st, const double *r, double tick, double *out) {
    double bp1 = r[0], bq1 = r[5], ap1 = r[10], aq1 = r[15];
    long t = (long)st[6];
    double mid = 0.5 * (bp1 + ap1);
    out[0] = (ap1 - bp1) / tick;
    out[1] = ((ap1 * bq1 + bp1 * aq1) / (bq1 + aq1) - mid) / tick;
    out[2] = (bq1 - aq1) / (bq1 + aq1);
    double sb = 0.0, sa = 0.0;
    for (int i = 0; i < 3; i++) { sb += r[5 + i]; sa += r[15 + i]; }
    out[3] = (sb - sa) / (sb + sa);
    for (int i = 3; i < 5; i++) { sb += r[5 + i]; sa += r[15 + i]; }
    out[4] = (sb - sa) / (sb + sa);
    double e = 0.0, dv = 0.0;
    if (t > 0) {
        double pbp = st[0], pbq = st[1], pap = st[2], paq = st[3];
        if (bp1 >= pbp) e += bq1;
        if (bp1 <= pbp) e -= pbq;
        if (ap1 <= pap) e -= aq1;
        if (ap1 >= pap) e += paq;
        dv = r[20] - st[4];
    }
    st[TF_OR + t % TF_W] = e;
    double s5 = 0.0, s20 = 0.0;
    long c = t + 1 < TF_W ? t + 1 : TF_W;
    for (long k = 0; k < c; k++) {
        double v = st[TF_OR + (t - k) % TF_W];
        s20 += v;
        if (k < 5) s5 += v;
    }
    out[5] = e; out[6] = s5; out[7] = s20;
    st[TF_MR + t % TF_MW] = mid;
    out[8] = t >= 1 ? (mid - st[TF_MR + (t - 1) % TF_MW]) / tick : 0.0;
    out[9] = t >= 5 ? (mid - st[TF_MR + (t - 5) % TF_MW]) / tick : 0.0;
    out[10] = t >= 20 ? (mid - st[TF_MR + (t - 20) % TF_MW]) / tick : 0.0;
    double s = 0.0, ss = 0.0;
    for (long k = 0; k < c; k++) {
        long u = t - k;
        double d = u == 0 ? 0.0 : (st[TF_MR + u % TF_MW] - st[TF_MR + (u - 1) % TF_MW]) / tick;
        s += d; ss += d * d;
    }
    if (c > 1) {
        double var = (ss - s * s / (double)c) / (double)(c - 1);
        out[11] = sqrt(var > 0.0 ? var : 0.0);
    } else {
        out[11] = 0.0;
    }
    out[12] = dv > 0 ? log1p(dv) : 0.0;
    double tq = r[21] + r[22];
    out[13] = tq > 0 ? (r[21] - r[22]) / tq : 0.0;
    double since = (t == 0 || mid != st[7]) ? 0.0 : st[5] + 1.0;
    out[14] = log1p(since < 100.0 ? since : 100.0);
    double ed = 0.0;
    if (t > 0) {
        for (int i = 1; i < 5; i++) {
            if (r[i] >= st[TF_PL + i - 1]) ed += r[5 + i];
            if (r[i] <= st[TF_PL + i - 1]) ed -= st[TF_PL + 4 + i - 1];
            if (r[10 + i] <= st[TF_PL + 8 + i - 1]) ed -= r[15 + i];
            if (r[10 + i] >= st[TF_PL + 8 + i - 1]) ed += st[TF_PL + 12 + i - 1];
        }
    }
    st[TF_DR + t % TF_W5] = ed;
    long c5 = t + 1 < TF_W5 ? t + 1 : TF_W5;
    double sd5 = 0.0;
    for (long k = 0; k < c5; k++) sd5 += st[TF_DR + (t - k) % TF_W5];
    out[15] = ed; out[16] = sd5;
    double sv = 0.0;
    if (t > 0 && dv > 0) {
        if (mid > st[7]) sv = dv;
        else if (mid < st[7]) sv = -dv;
    }
    st[TF_VR + t % TF_W5] = sv;
    double vs = 0.0;
    for (long k = 0; k < c5; k++) vs += st[TF_VR + (t - k) % TF_W5];
    out[17] = vs > 0 ? log1p(vs) : (vs < 0 ? -log1p(-vs) : 0.0);
    for (int i = 1; i < 5; i++) {
        st[TF_PL + i - 1] = r[i]; st[TF_PL + 4 + i - 1] = r[5 + i];
        st[TF_PL + 8 + i - 1] = r[10 + i]; st[TF_PL + 12 + i - 1] = r[15 + i];
    }
    st[0] = bp1; st[1] = bq1; st[2] = ap1; st[3] = aq1; st[4] = r[20];
    st[5] = since; st[6] = (double)(t + 1); st[7] = mid;
}
