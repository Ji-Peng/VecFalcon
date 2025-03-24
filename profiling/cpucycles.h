#ifndef CPUCYCLES_H
#define CPUCYCLES_H

#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#define DECL_SSR(NAME)   \
    void start_##NAME(); \
    void stop_##NAME();  \
    uint64_t read_##NAME();

int init_perf_counters();
DECL_SSR(cycles)
DECL_SSR(instructions)
// DECL_SSR(stalled_cycles_frontend)
// DECL_SSR(stalled_cycles_backend)

#define get_cpuinstret(FUNC, VALUE)        \
    do {                                   \
        start_instructions();              \
        for (int ii = 0; ii < 100; ii++) { \
            FUNC;                          \
        }                                  \
        stop_instructions();               \
        VALUE = read_instructions() / 100; \
    } while (0)

#define get_cpucycles(FUNC, INIT, VALUE, WARMUP_N, CYCLES_N) \
    do {                                                     \
        for (int ii = 0; ii < WARMUP_N; ii++) {              \
            FUNC;                                            \
        }                                                    \
        INIT;                                                \
        start_cycles();                                      \
        for (int ii = 0; ii < CYCLES_N; ii++) {              \
            FUNC;                                            \
        }                                                    \
        stop_cycles();                                       \
        uint64_t cycles_sum = read_cycles();                 \
        VALUE = cycles_sum / CYCLES_N;                       \
    } while (0)

#define PERF(FUNC, LABEL, INIT, WARMUP_N, CYCLES_N)       \
    do {                                                  \
        INIT;                                             \
        for (int ii = 0; ii < WARMUP_N; ii++) {           \
            FUNC;                                         \
        }                                                 \
        INIT;                                             \
        start_cycles();                                   \
        start_instructions();                             \
        for (int ii = 0; ii < CYCLES_N; ii++) {           \
            FUNC;                                         \
        }                                                 \
        stop_cycles();                                    \
        stop_instructions();                              \
        uint64_t cycles_sum = read_cycles();              \
        uint64_t inst_sum = read_instructions();          \
        printf("%-30s ", #LABEL);                         \
        printf("cycles/insts/CPI=%llu/%llu/%.2f\n",       \
               (unsigned long long)cycles_sum / CYCLES_N, \
               (unsigned long long)inst_sum / CYCLES_N,   \
               (float)cycles_sum / inst_sum);             \
    } while (0)

#define PERF_N(FUNC, LABEL, INIT, WARMUP_N, CYCLES_N, N_WAY)         \
    do {                                                             \
        INIT;                                                        \
        for (int ii = 0; ii < WARMUP_N; ii++) {                      \
            FUNC;                                                    \
        }                                                            \
        INIT;                                                        \
        start_cycles();                                              \
        start_instructions();                                        \
        for (int ii = 0; ii < CYCLES_N; ii++) {                      \
            FUNC;                                                    \
        }                                                            \
        stop_cycles();                                               \
        stop_instructions();                                         \
        uint64_t cycles_sum = read_cycles();                         \
        uint64_t inst_sum = read_instructions();                     \
        printf("%-30s ", #LABEL);                                    \
        printf("cycles/insts/CPI/1-wayCC=%llu/%llu/%.2f/%llu\n",     \
               (unsigned long long)cycles_sum / CYCLES_N,            \
               (unsigned long long)inst_sum / CYCLES_N,              \
               (float)cycles_sum / inst_sum,                         \
               (unsigned long long)(cycles_sum / CYCLES_N / N_WAY)); \
    } while (0)

// these two events are not supported on my machine with WSL ubuntu
#define PERF_STALLED(FUNC, LABEL, INIT, WARMUP_N, CYCLES_N)             \
    do {                                                                \
        INIT;                                                           \
        for (int ii = 0; ii < WARMUP_N; ii++) {                         \
            FUNC;                                                       \
        }                                                               \
        INIT;                                                           \
        start_stalled_cycles_frontend();                                \
        start_stalled_cycles_backend();                                 \
        for (int ii = 0; ii < CYCLES_N; ii++) {                         \
            FUNC;                                                       \
        }                                                               \
        stop_stalled_cycles_frontend();                                 \
        stop_stalled_cycles_backend();                                  \
        uint64_t stalled_frontend_sum = read_stalled_cycles_frontend(); \
        uint64_t stalled_backend_sum = read_stalled_cycles_backend();   \
        printf("%-30s ", #LABEL);                                       \
        printf("stalled_frontend/stalled_backend=%llu/%llu\n",          \
               (unsigned long long)stalled_frontend_sum / CYCLES_N,     \
               (unsigned long long)stalled_backend_sum / CYCLES_N);     \
    } while (0)

#undef DECL_SSR
#endif