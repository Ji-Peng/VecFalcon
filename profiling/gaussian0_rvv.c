#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "cpucycles.h"
#include "inner.h"

typedef struct {
#if FNDSA_SHAKE256X4
    shake256x4_context pc;
#else
    shake_context pc;
#endif
    unsigned logn;
} sampler_state;

/* We access the PRNG through macros so that they can be overridden by some
   compatiblity tests with the original Falcon implementation. */
#ifndef prng_init
#    if FNDSA_SHAKE256X4
#        define prng_init shake256x4_init
#        define prng_next_u8 shake256x4_next_u8
#        define prng_next_u16 shake256x4_next_u16
#        define prng_next_u24 shake256x4_next_u24
#        define prng_next_u64 shake256x4_next_u64
#    else
#        define prng_init(pc, seed, seed_len)     \
            do {                                  \
                shake_init(pc, 256);              \
                shake_inject(pc, seed, seed_len); \
                shake_flip(pc);                   \
            } while (0)
#        define prng_next_u8 shake_next_u8
#        define prng_next_u16 shake_next_u16
#        define prng_next_u24 shake_next_u24
#        define prng_next_u64 shake_next_u64
#    endif
#endif

void print_poly(int32_t *data, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        printf("%d", data[i]);

        if ((i + 1) % 32 == 0) {
            printf("\n");
        } else if (i != n - 1) {
            printf(" ");
        }
    }
    if (n % 32 != 0) {
        printf("\n");
    }
}

/* see sign_inner.h */
void sampler_init(sampler_state *ss, unsigned logn, const void *seed,
                  size_t seed_len)
{
    prng_init(&ss->pc, seed, seed_len);
    ss->logn = logn;
}

static const uint32_t GAUSS0[][3] = {{10745844, 3068844, 3741698},
                                     {5559083, 1580863, 8248194},
                                     {2260429, 13669192, 2736639},
                                     {708981, 4421575, 10046180},
                                     {169348, 7122675, 4136815},
                                     {30538, 13063405, 7650655},
                                     {4132, 14505003, 7826148},
                                     {417, 16768101, 11363290},
                                     {31, 8444042, 8086568},
                                     {1, 12844466, 265321},
                                     {0, 1232676, 13644283},
                                     {0, 38047, 9111839},
                                     {0, 870, 6138264},
                                     {0, 14, 12545723},
                                     {0, 0, 3104126},
                                     {0, 0, 28824},
                                     {0, 0, 198},
                                     {0, 0, 1}};

/**
 * Obtained from https://github.com/pornin/c-fn-dsa.
 *
 * The original version returns z directly after the for loop ends. We do
 * some additional operations, namely "z*z" and "get a random bit b to turn
 * the sampling into a bimodal distribution". In the original version of
 * the implementation, these additional operations are done in the
 * sampler_next subroutine. The reason we do this is to facilitate
 * vectorization of these additional operations.
 *
 * Let z after the loop be z0.
 * @param *z_bimodal = b + ((b << 1) - 1) * z0, where b is a random bit
 * @param *z_square = z0*z0
 */
void gaussian0_ref(sampler_state *ss, int32_t *z_bimodal,
                   int32_t *z_square)
{
    /* Get a random 72-bit value, into three 24-bit limbs (v0..v2). */
    uint64_t lo = prng_next_u64(&ss->pc);
    uint32_t hi = prng_next_u8(&ss->pc);
    uint32_t v0 = (uint32_t)lo & 0xFFFFFF;
    uint32_t v1 = (uint32_t)(lo >> 24) & 0xFFFFFF;
    uint32_t v2 = (uint32_t)(lo >> 48) | (hi << 16);

    /* Sampled value is z such that v0..v2 is lower than the first
    z elements of the table. */
    int32_t z = 0;
    for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
        uint32_t cc;
        cc = (v0 - GAUSS0[i][2]) >> 31;
        cc = (v1 - GAUSS0[i][1] - cc) >> 31;
        cc = (v2 - GAUSS0[i][0] - cc) >> 31;
        z += (int32_t)cc;
    }
    // Get a random bit b to turn the sampling into a bimodal distribution.
    int32_t b = prng_next_u8(&ss->pc) & 1;
    *z_bimodal = b + ((b << 1) - 1) * z;
    *z_square = z * z;
}

// used for testing correctness
void gaussian0_ref_16w(sampler_state *ss, int32_t *z_bimodal,
                       int32_t *z_square)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;
    int32_t z[16] = {0};

    for (size_t j = 0; j < 16; j++) {
        /* Get a random 72-bit value, into three 24-bit limbs (v0..v2). */
        uint64_t lo = prng_next_u64(&ss->pc);
        uint32_t hi = prng_next_u8(&ss->pc);
        uint32_t v0 = (uint32_t)lo & 0xFFFFFF;
        uint32_t v1 = (uint32_t)(lo >> 24) & 0xFFFFFF;
        uint32_t v2 = (uint32_t)(lo >> 48) | (hi << 16);
        for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
            uint32_t cc;
            cc = (v0 - GAUSS0[i][2]) >> 31;
            cc = (v1 - GAUSS0[i][1] - cc) >> 31;
            cc = (v2 - GAUSS0[i][0] - cc) >> 31;
            z[j] += (int32_t)cc;
        }
    }
    uint16_t b_16b = prng_next_u16(&ss->pc);
    for (size_t j = 0; j < 16; j++) {
        int32_t b = (b_16b >> j) & 1;
        *(_z_bi + j) = b + ((b << 1) - 1) * z[j];
        *(_z_sq + j) = z[j] * z[j];
    }
}

// used for testing correctness
void gaussian0_ref_16w_u24(sampler_state *ss, int32_t *z_bimodal,
                           int32_t *z_square)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;
    int32_t z[16] = {0};

    for (size_t j = 0; j < 16; j++) {
        /* Get a random 72-bit value, into three 24-bit limbs (v0..v2). */
        uint32_t v0 = prng_next_u24(&ss->pc);
        uint32_t v1 = prng_next_u24(&ss->pc);
        uint32_t v2 = prng_next_u24(&ss->pc);
        for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
            uint32_t cc;
            cc = (v0 - GAUSS0[i][2]) >> 31;
            cc = (v1 - GAUSS0[i][1] - cc) >> 31;
            cc = (v2 - GAUSS0[i][0] - cc) >> 31;
            z[j] += (int32_t)cc;
        }
    }
    uint16_t b_16b = prng_next_u16(&ss->pc);
    for (size_t j = 0; j < 16; j++) {
        int32_t b = (b_16b >> j) & 1;
        *(_z_bi + j) = b + ((b << 1) - 1) * z[j];
        *(_z_sq + j) = z[j] * z[j];
    }
}

// used for testing correctness
void gaussian0_ref_64w(sampler_state *ss, int32_t *z_bimodal,
                       int32_t *z_square)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;
    int32_t z[64] = {0};

    for (size_t j = 0; j < 64; j++) {
        /* Get a random 72-bit value, into three 24-bit limbs (v0..v2). */
        uint64_t lo = prng_next_u64(&ss->pc);
        uint32_t hi = prng_next_u8(&ss->pc);
        uint32_t v0 = (uint32_t)lo & 0xFFFFFF;
        uint32_t v1 = (uint32_t)(lo >> 24) & 0xFFFFFF;
        uint32_t v2 = (uint32_t)(lo >> 48) | (hi << 16);
        for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
            uint32_t cc;
            cc = (v0 - GAUSS0[i][2]) >> 31;
            cc = (v1 - GAUSS0[i][1] - cc) >> 31;
            cc = (v2 - GAUSS0[i][0] - cc) >> 31;
            z[j] += (int32_t)cc;
        }
    }
    uint64_t b_64b = prng_next_u64(&ss->pc);
    for (size_t j = 0; j < 64; j++) {
        int32_t b = (b_64b >> j) & 1;
        *(_z_bi + j) = b + ((b << 1) - 1) * z[j];
        *(_z_sq + j) = z[j] * z[j];
    }
}

/**
 * Only used to test the performance of core calculations, excluding the
 * overhead caused by obtaining random numbers.
 */
void gaussian0_ref_core(int32_t *z_bimodal, int32_t *z_square)
{
    /* Get a random 72-bit value, into three 24-bit limbs (v0..v2). */
    uint64_t lo = *z_bimodal << 32 + *z_bimodal;
    uint32_t hi = *z_square;
    uint32_t v0 = (uint32_t)lo & 0xFFFFFF;
    uint32_t v1 = (uint32_t)(lo >> 24) & 0xFFFFFF;
    uint32_t v2 = (uint32_t)(lo >> 48) | (hi << 16);

    /* Sampled value is z such that v0..v2 is lower than the first
    z elements of the table. */
    int32_t z = 0;
    for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
        uint32_t cc;
        cc = (v0 - GAUSS0[i][2]) >> 31;
        cc = (v1 - GAUSS0[i][1] - cc) >> 31;
        cc = (v2 - GAUSS0[i][0] - cc) >> 31;
        z += (int32_t)cc;
    }
    unsigned b = *z_bimodal & 1;
    *z_bimodal = b + ((b << 1) - 1) * z;
    *z_square = z * z;
}

typedef union {
    uint32_t u32[3][8] __attribute__((aligned(32)));
} prn_24x3_8w;

#define ALIGNED_INT32(N)   \
    union {                \
        int32_t coeffs[N]; \
    }

extern void gaussian0_rvv(int32_t *out, uint32_t *prn, size_t n);

void gaussian0_rvv_16w(sampler_state *ss, int32_t *z_bimodal,
                       int32_t *z_square)
{
    prn_24x3_8w prn[2];
    int32_t z[16] __attribute__((aligned(32)));
    int32_t *_z_bi = (int32_t *)z_bimodal;
    int32_t *_z_sq = (int32_t *)z_square;

    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 8; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    gaussian0_rvv(z, &prn[0].u32[0][0], 2);
    uint16_t b_16b = prng_next_u16(&ss->pc);
    for (size_t j = 0; j < 16; j += 2) {
        int32_t b0 = (b_16b >> j) & 1;
        int32_t b1 = (b_16b >> (j + 1)) & 1;
        int32_t m0 = (b0 << 1);
        int32_t m1 = (b1 << 1);
        m0 = m0 - 1;
        m1 = m1 - 1;
        m0 = m0 * z[j];
        m1 = m1 * z[j + 1];
        _z_bi[j] = b0 + m0;
        _z_bi[j + 1] = b1 + m1;
        _z_sq[j] = z[j] * z[j];
        _z_sq[j + 1] = z[j + 1] * z[j + 1];
    }
}

extern void gaussian0_rvv_bisq(int32_t *z_bimodal, int32_t *z_square,
                               uint32_t *prn, uint32_t *b_16bs, size_t n);
// including the bimodal tunning and square calculation
void gaussian0_bisq_rvv_16w(sampler_state *ss, int32_t *z_bimodal,
                            int32_t *z_square)
{
    prn_24x3_8w prn[2];
    uint32_t b_16bs[16];
    int32_t *_z_bi = (int32_t *)z_bimodal;
    int32_t *_z_sq = (int32_t *)z_square;

    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 8; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    uint16_t b_16b = prng_next_u16(&ss->pc);
    for (int i = 0; i < 16; i += 2) {
        b_16bs[i] = (b_16b >> i) & 1;
        b_16bs[i + 1] = (b_16b >> (i + 1)) & 1;
    }
    gaussian0_rvv_bisq(_z_bi, _z_sq, &prn[0].u32[0][0], b_16bs, 2);
}

// including the bimodal tunning and square calculation
void gaussian0_bisq_rvv_64w(sampler_state *ss, int32_t *z_bimodal,
                            int32_t *z_square)
{
    ALIGNED_INT32(64) b_64bs;
    prn_24x3_8w prn[8];
    int32_t *_z_bi = (int32_t *)z_bimodal;
    int32_t *_z_sq = (int32_t *)z_square;

    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    uint64_t b_64b = prng_next_u64(&ss->pc);
    for (int i = 0; i < 64; i += 4) {
        b_64bs.coeffs[i] = (b_64b >> i) & 1;
        b_64bs.coeffs[i + 1] = (b_64b >> (i + 1)) & 1;
        b_64bs.coeffs[i + 2] = (b_64b >> (i + 2)) & 1;
        b_64bs.coeffs[i + 3] = (b_64b >> (i + 3)) & 1;
    }
    gaussian0_rvv_bisq(_z_bi, _z_sq, &prn[0].u32[0][0], b_64bs.coeffs, 8);
}

/**
 * Only used to test the performance of core calculations, excluding the
 * overhead caused by obtaining random numbers.
 */
void gaussian0_rvv_nw_core(int32_t *z_bimodal, int32_t *z_square, size_t n)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;
    uint32_t t[128] __attribute__((aligned(32)));

    gaussian0_rvv_bisq(_z_bi, _z_sq, t, t, n >> 3);
}

#define WARMUP_N 1000
// 136*4-(136*4%3)=543
// 543*3 % 9 = 0
#define TESTS_N ((543 * 3) * 100)
#define SAMPLES_N (8 * (1 << 5))

void test_gaussian0()
{
    ALIGNED_INT32(SAMPLES_N) z0_bi, z1_bi, z0_sq, z1_sq;
    uint8_t seed[32] = {0};
    sampler_state ss0, ss1;
    size_t i;

    memset(&z0_bi, 0, sizeof(z0_bi));
    memset(&z0_sq, 0, sizeof(z0_sq));
    sampler_init(&ss0, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 16) {
        gaussian0_ref_16w_u24(&ss0, &z0_bi.coeffs[i], &z0_sq.coeffs[i]);
    }
    memset(&z1_bi, 0, sizeof(z1_bi));
    memset(&z1_sq, 0, sizeof(z1_sq));
    sampler_init(&ss1, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 16) {
        gaussian0_rvv_16w(&ss1, &z1_bi.coeffs[i], &z1_sq.coeffs[i]);
    }
    if (memcmp(z0_bi.coeffs, z1_bi.coeffs, sizeof(int32_t) * SAMPLES_N) !=
            0 ||
        memcmp(z0_sq.coeffs, z1_sq.coeffs, sizeof(int32_t) * SAMPLES_N) !=
            0) {
        printf("gaussian0_ref_16w_u24 != gaussian0_rvv_16w\n");
    }
    memset(&z1_bi, 0, sizeof(z1_bi));
    memset(&z1_sq, 0, sizeof(z1_sq));
    sampler_init(&ss1, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 16) {
        gaussian0_bisq_rvv_16w(&ss1, &z1_bi.coeffs[i], &z1_sq.coeffs[i]);
    }
    if (memcmp(z0_bi.coeffs, z1_bi.coeffs, sizeof(int32_t) * SAMPLES_N) !=
            0 ||
        memcmp(z0_sq.coeffs, z1_sq.coeffs, sizeof(int32_t) * SAMPLES_N) !=
            0) {
        printf("gaussian0_ref_16w_u24 != gaussian0_bisq_rvv_16w\n");
    }
}
extern void init_vector_e16();
extern void init_vector_e32();
extern void init_vector_e64();
extern void gaussian0_rvv_24b_p0_x2();
extern void gaussian0_rvv_24b_p0_x3();
extern void gaussian0_rvv_24b_p1_x3();
extern void gaussian0_rvv_24b_p1_x3_v2();
extern void gaussian0_rvv_64b_p0_x2();
extern void gaussian0_rvv_64b_p0_x3();
extern void gaussian0_rvv_64b_p1_x3();
extern void gaussian0_rvv_64b_p1_x3_v2();

void speed_gaussian0()
{
    sampler_state ss0;
    uint8_t seed[32] = {0};
    ALIGNED_INT32(128) z0_bi, z0_sq;

    init_perf_counters();

    printf("Including prng_next_* subroutines\n");
    PERF(gaussian0_ref(&ss0, &z0_bi.coeffs[0], &z0_sq.coeffs[0]),
         gaussian0_ref, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
         TESTS_N);
    PERF_N(
        gaussian0_bisq_rvv_16w(&ss0, &z0_bi.coeffs[0], &z0_sq.coeffs[0]),
        gaussian0_bisq_rvv_16w, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
        TESTS_N, 16);
    PERF_N(
        gaussian0_bisq_rvv_64w(&ss0, &z0_bi.coeffs[0], &z0_sq.coeffs[0]),
        gaussian0_bisq_rvv_64w, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
        TESTS_N, 64);
    /**
     * Test the performance of core calculations, excluding
     * the overhead caused by obtaining random numbers.
     */
    printf(
        "\nExcluding prng_next_* subroutines; including bimodal/square "
        "calculations\n");
    PERF(gaussian0_ref_core(&z0_bi.coeffs[0], &z0_sq.coeffs[0]),
         gaussian0_ref_core, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
         TESTS_N);
    PERF_N(gaussian0_rvv_nw_core(&z0_bi.coeffs[0], &z0_sq.coeffs[0], 16),
           gaussian0_rvv_16w_core, sampler_init(&ss0, 9, seed, 32),
           WARMUP_N, TESTS_N, 16);
    PERF_N(gaussian0_rvv_nw_core(&z0_bi.coeffs[0], &z0_sq.coeffs[0], 64),
           gaussian0_rvv_64w_core, sampler_init(&ss0, 9, seed, 32),
           WARMUP_N, TESTS_N, 64);

#if REF_SHA3 == 0
    printf("\nSome optimization ideas:\n");
    init_vector_e64();
    PERF_N(gaussian0_rvv_64b_p0_x2(), gaussian0_rvv_64b_p0_x2, , WARMUP_N,
           TESTS_N, 2);
    PERF_N(gaussian0_rvv_64b_p0_x3(), gaussian0_rvv_64b_p0_x3, , WARMUP_N,
           TESTS_N, 3);
    PERF_N(gaussian0_rvv_64b_p1_x3(), gaussian0_rvv_64b_p1_x3, , WARMUP_N,
           TESTS_N, 3);
    PERF_N(gaussian0_rvv_64b_p1_x3_v2(), gaussian0_rvv_64b_p1_x3_v2, ,
           WARMUP_N, TESTS_N, 3);

    init_vector_e32();
    PERF_N(gaussian0_rvv_24b_p0_x2(), gaussian0_rvv_24b_p0_x2, , WARMUP_N,
           TESTS_N, 2);
    PERF_N(gaussian0_rvv_24b_p0_x3(), gaussian0_rvv_24b_p0_x3, , WARMUP_N,
           TESTS_N, 3);
    PERF_N(gaussian0_rvv_24b_p1_x3(), gaussian0_rvv_24b_p1_x3, , WARMUP_N,
           TESTS_N, 3);
    PERF_N(gaussian0_rvv_24b_p1_x3_v2(), gaussian0_rvv_24b_p1_x3_v2, ,
           WARMUP_N, TESTS_N, 3);
#endif
}

int main(void)
{
    test_gaussian0();
    speed_gaussian0();
    return 0;
}