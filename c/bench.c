/* bench.c — parity and latency of the deployed C model on a recorded book stream.
   usage: bench rows.bin expected.bin n
     rows.bin     n x 23 doubles (bp1..5, bq1..5, ap1..5, aq1..5, volume, tbq, tsq)
     expected.bin n int8 classes from the Python/numba pipeline on the same rows
   Prints mismatches, throughput (ns per update) and per-call latency percentiles from the CPU cycle counter. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include <intrin.h>
#include "tinyml_features.h"
#include "tinyml_model.h"

static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : (x > y);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: bench rows.bin expected.bin n\n"); return 2; }
    long n = atol(argv[3]);
    double *R = malloc(sizeof(double) * 23 * n);
    int8_t *E = malloc(n);
    double *X = malloc(sizeof(double) * TF_NF * n);
    uint64_t *cyc = malloc(sizeof(uint64_t) * n);
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(R, sizeof(double), 23 * n, f) != (size_t)(23 * n)) { fprintf(stderr, "bad rows\n"); return 1; }
    fclose(f);
    f = fopen(argv[2], "rb");
    if (!f || fread(E, 1, n, f) != (size_t)n) { fprintf(stderr, "bad expected\n"); return 1; }
    fclose(f);

    double st[TF_ST], x[TF_NF];
    long mism = 0, sink = 0;

    /* 1. parity + full-tick throughput (features + model) */
    memset(st, 0, sizeof st);
    double t0 = now_s();
    for (long i = 0; i < n; i++) {
        tf_update(st, R + 23 * i, TM_TICK, x);
        int c = tm_predict(x, NULL);
        mism += (c != E[i]);
    }
    double t1 = now_s();

    /* 2. model-only throughput on precomputed features */
    memset(st, 0, sizeof st);
    for (long i = 0; i < n; i++) tf_update(st, R + 23 * i, TM_TICK, X + TF_NF * i);
    double t2 = now_s();
    for (long i = 0; i < n; i++) sink += tm_predict(X + TF_NF * i, NULL);
    double t3 = now_s();

    /* 3. per-call cycles for the full tick */
    memset(st, 0, sizeof st);
    for (long i = 0; i < n; i++) {
        uint64_t a = __rdtsc();
        tf_update(st, R + 23 * i, TM_TICK, x);
        sink += tm_predict(x, NULL);
        cyc[i] = __rdtsc() - a;
    }
    double q0 = now_s(); uint64_t c0 = __rdtsc();
    while (now_s() - q0 < 0.25) {}
    uint64_t c1 = __rdtsc(); double q1 = now_s();
    double ghz = (double)(c1 - c0) / (q1 - q0) / 1e9;
    qsort(cyc, n, sizeof(uint64_t), cmp_u64);

    printf("rows %ld\nmismatches %ld\n", n, mism);
    printf("tick_ns %.2f\nmodel_ns %.2f\n", (t1 - t0) / n * 1e9, (t3 - t2) / n * 1e9);
    printf("tsc_ghz %.3f\n", ghz);
    printf("p50_ns %.1f\np99_ns %.1f\np999_ns %.1f\n", cyc[n / 2] / ghz, cyc[(long)(n * 0.99)] / ghz, cyc[(long)(n * 0.999)] / ghz);
    printf("sink %ld\n", sink);
    return mism != 0;
}
