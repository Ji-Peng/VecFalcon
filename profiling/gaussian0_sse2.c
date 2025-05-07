#include <immintrin.h>
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
#        define prng_next_u24 shake_next_u24
#        define prng_next_u64 shake_next_u64
#    endif
#endif

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

#define U32X4(W)       \
    {                  \
        {              \
            W, W, W, W \
        }              \
    }

typedef union {
    uint32_t u32[4];
    __m128i xmm;
} gauss0_32x4;

static const gauss0_32x4 GAUSS0_SSE2[][3] = {
    {U32X4(10745844), U32X4(3068844), U32X4(3741698)},
    {U32X4(5559083), U32X4(1580863), U32X4(8248194)},
    {U32X4(2260429), U32X4(13669192), U32X4(2736639)},
    {U32X4(708981), U32X4(4421575), U32X4(10046180)},
    {U32X4(169348), U32X4(7122675), U32X4(4136815)},
    {U32X4(30538), U32X4(13063405), U32X4(7650655)},
    {U32X4(4132), U32X4(14505003), U32X4(7826148)},
    {U32X4(417), U32X4(16768101), U32X4(11363290)},
    {U32X4(31), U32X4(8444042), U32X4(8086568)},
    {U32X4(1), U32X4(12844466), U32X4(265321)},
    {U32X4(0), U32X4(1232676), U32X4(13644283)},
    {U32X4(0), U32X4(38047), U32X4(9111839)},
    {U32X4(0), U32X4(870), U32X4(6138264)},
    {U32X4(0), U32X4(14), U32X4(12545723)},
    {U32X4(0), U32X4(0), U32X4(3104126)},
    {U32X4(0), U32X4(0), U32X4(28824)},
    {U32X4(0), U32X4(0), U32X4(198)},
    {U32X4(0), U32X4(0), U32X4(1)}};

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

/**
 * The way to get 72-bit in gaussian0_ref is u64+u8. When using SHAKE256X4,
 * the test vectors of u64+u8 and u24x3 will be different. The critical
 * point is 136*4-60*9=4 bytes. The former will directly trigger refill,
 * while the latter can still get 24 bits.
 */
void gaussian0_ref_u24(sampler_state *ss, void *z_bimodal, void *z_square)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;
    /* Get a random 72-bit value, into three 24-bit limbs (v0..v2). */
    uint32_t v0 = prng_next_u24(&ss->pc);
    uint32_t v1 = prng_next_u24(&ss->pc);
    uint32_t v2 = prng_next_u24(&ss->pc);

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
    *_z_bi = b + ((b << 1) - 1) * z;
    *_z_sq = z * z;
}

// used for test correctness of gaussian0_sse2_4w
void gaussian0_ref_u24_4w(sampler_state *ss, void *z_bimodal,
                          void *z_square)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;

    int32_t z[4] = {0};
    for (size_t j = 0; j < 4; j++) {
        /* Get a random 72-bit value, into three 24-bit limbs (v0..v2). */
        uint32_t v0 = prng_next_u24(&ss->pc);
        uint32_t v1 = prng_next_u24(&ss->pc);
        uint32_t v2 = prng_next_u24(&ss->pc);
        /* Sampled value is z such that v0..v2 is lower than the first
        z elements of the table. */
        for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
            uint32_t cc;
            cc = (v0 - GAUSS0[i][2]) >> 31;
            cc = (v1 - GAUSS0[i][1] - cc) >> 31;
            cc = (v2 - GAUSS0[i][0] - cc) >> 31;
            z[j] += (int32_t)cc;
        }
    }

    for (size_t j = 0; j < 4; j++) {
        // Get a random bit b to turn the sampling into a bimodal
        // distribution.
        int32_t b = prng_next_u8(&ss->pc) & 1;
        *(_z_bi + j) = b + ((b << 1) - 1) * z[j];
        *(_z_sq + j) = z[j] * z[j];
    }
}

// used for test correctness of gaussian0_sse2_8w
void gaussian0_ref_u24_8w(sampler_state *ss, void *z_bimodal,
                          void *z_square)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;

    int32_t z[8] = {0};
    for (size_t j = 0; j < 8; j++) {
        /* Get a random 72-bit value, into three 24-bit limbs (v0..v2). */
        uint32_t v0 = prng_next_u24(&ss->pc);
        uint32_t v1 = prng_next_u24(&ss->pc);
        uint32_t v2 = prng_next_u24(&ss->pc);
        /* Sampled value is z such that v0..v2 is lower than the first
        z elements of the table. */
        for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
            uint32_t cc;
            cc = (v0 - GAUSS0[i][2]) >> 31;
            cc = (v1 - GAUSS0[i][1] - cc) >> 31;
            cc = (v2 - GAUSS0[i][0] - cc) >> 31;
            z[j] += (int32_t)cc;
        }
    }

    for (size_t j = 0; j < 8; j++) {
        // Get a random bit b to turn the sampling into a bimodal
        // distribution.
        int32_t b = prng_next_u8(&ss->pc) & 1;
        *(_z_bi + j) = b + ((b << 1) - 1) * z[j];
        *(_z_sq + j) = z[j] * z[j];
    }
}

// used for test correctness of gaussian0_sse2_16w
void gaussian0_ref_u24_16w(sampler_state *ss, void *z_bimodal,
                           void *z_square)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;

    int32_t z[16] = {0};
    for (size_t j = 0; j < 16; j++) {
        /* Get a random 72-bit value, into three 24-bit limbs (v0..v2). */
        uint32_t v0 = prng_next_u24(&ss->pc);
        uint32_t v1 = prng_next_u24(&ss->pc);
        uint32_t v2 = prng_next_u24(&ss->pc);
        /* Sampled value is z such that v0..v2 is lower than the first
        z elements of the table. */
        for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
            uint32_t cc;
            cc = (v0 - GAUSS0[i][2]) >> 31;
            cc = (v1 - GAUSS0[i][1] - cc) >> 31;
            cc = (v2 - GAUSS0[i][0] - cc) >> 31;
            z[j] += (int32_t)cc;
        }
    }

    for (size_t j = 0; j < 16; j++) {
        // Get a random bit b to turn the sampling into a bimodal
        // distribution.
        int32_t b = prng_next_u8(&ss->pc) & 1;
        *(_z_bi + j) = b + ((b << 1) - 1) * z[j];
        *(_z_sq + j) = z[j] * z[j];
    }
}

typedef union {
    uint32_t u32[3][4];
    __m128i xmm[3];
} prn_24x3_4w;

#define ALIGNED_INT32(N)          \
    union {                       \
        int32_t coeffs[N];        \
        __m128i vec[(N + 3) / 4]; \
    }

void gaussian0_sse2_4w(sampler_state *ss, void *z_bimodal, void *z_square)
{
    prn_24x3_4w prn[1];
    __m128i *_z_bi = (__m128i *)z_bimodal;
    __m128i *_z_sq = (__m128i *)z_square;

    /* Get random 72-bit values, with 3x24-bit form. */
    for (int i = 0; i < 4; i++) {
        prn[0].u32[0][i] = prng_next_u24(&ss->pc);
        prn[0].u32[1][i] = prng_next_u24(&ss->pc);
        prn[0].u32[2][i] = prng_next_u24(&ss->pc);
    }
    __m128i z0 = _mm_setzero_si128();
    __m128i cc0;
    __m128i t0, t1, t2;
    for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
        // load pre-computed table
        t2 = _mm_loadu_si128(&GAUSS0_SSE2[i][2].xmm);
        t1 = _mm_loadu_si128(&GAUSS0_SSE2[i][1].xmm);
        t0 = _mm_loadu_si128(&GAUSS0_SSE2[i][0].xmm);
        // cc = (v0 - GAUSS0[i][2]) >> 31;
        cc0 = _mm_sub_epi32(prn[0].xmm[0], t2);
        cc0 = _mm_srli_epi32(cc0, 31);
        // cc = (v1 - GAUSS0[i][1] - cc) >> 31;
        cc0 = _mm_sub_epi32(prn[0].xmm[1], cc0);
        cc0 = _mm_sub_epi32(cc0, t1);
        cc0 = _mm_srli_epi32(cc0, 31);
        // cc = (v2 - GAUSS0[i][0] - cc) >> 31;
        cc0 = _mm_sub_epi32(prn[0].xmm[2], cc0);
        cc0 = _mm_sub_epi32(cc0, t0);
        cc0 = _mm_srli_epi32(cc0, 31);
        z0 = _mm_add_epi32(z0, cc0);
    }
    ALIGNED_INT32(4) b;
    for (size_t i = 0; i < 4; i++) {
        b.coeffs[i] = prng_next_u8(&ss->pc) & 1;
    }
    /**
     * Each sample is in the range [0,18], so we can use the 16-bit
     * multiplication instruction.
     *
     * But when taking z_bimodal, please note that we only take the lower
     * 16 bits of each sample and then convert it to a signed 32-bit
     * number.
     */
    t0 = _mm_load_si128(&b.vec[0]);
    t1 = _mm_add_epi32(t0, t0);
    t1 = _mm_sub_epi32(t1, _mm_set1_epi32(1));
    t2 = _mm_mullo_epi16(t1, z0);
    t2 = _mm_add_epi32(t2, t0);
    _mm_store_si128(_z_bi, t2);
    _mm_store_si128(_z_sq, _mm_mullo_epi16(z0, z0));
}

void gaussian0_sse2_8w(sampler_state *ss, void *z_bimodal, void *z_square)
{
    prn_24x3_4w prn[2];
    __m128i *_z_bi = (__m128i *)z_bimodal;
    __m128i *_z_sq = (__m128i *)z_square;

    /* Get random 72-bit values, with 3x24-bit form. */
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 4; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    __m128i z0 = _mm_setzero_si128(), z1 = _mm_setzero_si128();
    __m128i cc0, cc1;
    __m128i t0, t1, t2, t3, t4, t5;
    for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
        // load pre-computed table
        t2 = _mm_loadu_si128(&GAUSS0_SSE2[i][2].xmm);
        t1 = _mm_loadu_si128(&GAUSS0_SSE2[i][1].xmm);
        t0 = _mm_loadu_si128(&GAUSS0_SSE2[i][0].xmm);
        // cc = (v0 - GAUSS0[i][2]) >> 31;
        cc0 = _mm_sub_epi32(prn[0].xmm[0], t2);
        cc1 = _mm_sub_epi32(prn[1].xmm[0], t2);
        cc0 = _mm_srli_epi32(cc0, 31);
        cc1 = _mm_srli_epi32(cc1, 31);
        // cc = (v1 - GAUSS0[i][1] - cc) >> 31;
        cc0 = _mm_sub_epi32(prn[0].xmm[1], cc0);
        cc1 = _mm_sub_epi32(prn[1].xmm[1], cc1);
        cc0 = _mm_sub_epi32(cc0, t1);
        cc1 = _mm_sub_epi32(cc1, t1);
        cc0 = _mm_srli_epi32(cc0, 31);
        cc1 = _mm_srli_epi32(cc1, 31);
        // cc = (v2 - GAUSS0[i][0] - cc) >> 31;
        cc0 = _mm_sub_epi32(prn[0].xmm[2], cc0);
        cc1 = _mm_sub_epi32(prn[1].xmm[2], cc1);
        cc0 = _mm_sub_epi32(cc0, t0);
        cc1 = _mm_sub_epi32(cc1, t0);
        cc0 = _mm_srli_epi32(cc0, 31);
        cc1 = _mm_srli_epi32(cc1, 31);
        z0 = _mm_add_epi32(z0, cc0);
        z1 = _mm_add_epi32(z1, cc1);
    }
    ALIGNED_INT32(8) b;
    for (size_t i = 0; i < 8; i++) {
        b.coeffs[i] = prng_next_u8(&ss->pc) & 1;
    }
    /**
     * Each sample is in the range [0,18], so we can use the 16-bit
     * multiplication instruction.
     *
     * But when taking z_bimodal, please note that we only take the lower
     * 16 bits of each sample and then convert it to a signed 32-bit
     * number.
     */
    t0 = _mm_load_si128(&b.vec[0]);
    t3 = _mm_load_si128(&b.vec[1]);
    t1 = _mm_add_epi32(t0, t0);
    t4 = _mm_add_epi32(t3, t3);
    t1 = _mm_sub_epi32(t1, _mm_set1_epi32(1));
    t4 = _mm_sub_epi32(t4, _mm_set1_epi32(1));
    t2 = _mm_mullo_epi16(t1, z0);
    t5 = _mm_mullo_epi16(t4, z1);
    t2 = _mm_add_epi32(t2, t0);
    t5 = _mm_add_epi32(t5, t3);
    _mm_store_si128(_z_bi, t2);
    _mm_store_si128(_z_bi + 1, t5);
    _mm_store_si128(_z_sq, _mm_mullo_epi16(z0, z0));
    _mm_store_si128(_z_sq + 1, _mm_mullo_epi16(z1, z1));
}

void gaussian0_sse2_16w(sampler_state *ss, void *z_bimodal, void *z_square)
{
    prn_24x3_4w prn[2];
    __m128i *_z_bi = (__m128i *)z_bimodal;
    __m128i *_z_sq = (__m128i *)z_square;
    __m128i z0[2], z1[2];
    __m128i cc0, cc1;
    __m128i t0, t1, t2, t3, t4, t5;

    for (size_t j = 0; j < 2; j++) {
        /* Get random 72-bit values, with 3x24-bit form. */
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < 4; i++) {
                prn[k].u32[0][i] = prng_next_u24(&ss->pc);
                prn[k].u32[1][i] = prng_next_u24(&ss->pc);
                prn[k].u32[2][i] = prng_next_u24(&ss->pc);
            }
        z0[j] = z1[j] = _mm_setzero_si128();
        for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
            // load pre-computed table
            t2 = _mm_loadu_si128(&GAUSS0_SSE2[i][2].xmm);
            t1 = _mm_loadu_si128(&GAUSS0_SSE2[i][1].xmm);
            t0 = _mm_loadu_si128(&GAUSS0_SSE2[i][0].xmm);
            // cc = (v0 - GAUSS0[i][2]) >> 31;
            cc0 = _mm_sub_epi32(prn[0].xmm[0], t2);
            cc1 = _mm_sub_epi32(prn[1].xmm[0], t2);
            cc0 = _mm_srli_epi32(cc0, 31);
            cc1 = _mm_srli_epi32(cc1, 31);
            // cc = (v1 - GAUSS0[i][1] - cc) >> 31;
            cc0 = _mm_sub_epi32(prn[0].xmm[1], cc0);
            cc1 = _mm_sub_epi32(prn[1].xmm[1], cc1);
            cc0 = _mm_sub_epi32(cc0, t1);
            cc1 = _mm_sub_epi32(cc1, t1);
            cc0 = _mm_srli_epi32(cc0, 31);
            cc1 = _mm_srli_epi32(cc1, 31);
            // cc = (v2 - GAUSS0[i][0] - cc) >> 31;
            cc0 = _mm_sub_epi32(prn[0].xmm[2], cc0);
            cc1 = _mm_sub_epi32(prn[1].xmm[2], cc1);
            cc0 = _mm_sub_epi32(cc0, t0);
            cc1 = _mm_sub_epi32(cc1, t0);
            cc0 = _mm_srli_epi32(cc0, 31);
            cc1 = _mm_srli_epi32(cc1, 31);
            z0[j] = _mm_add_epi32(z0[j], cc0);
            z1[j] = _mm_add_epi32(z1[j], cc1);
        }
    }
    ALIGNED_INT32(8) b;
    for (size_t j = 0; j < 2; j++) {
        for (size_t i = 0; i < 8; i++) {
            b.coeffs[i] = prng_next_u8(&ss->pc) & 1;
        }
        /**
         * Each sample is in the range [0,18], so we can use the 16-bit
         * multiplication instruction.
         *
         * But when taking z_bimodal, please note that we only take the
         * lower 16 bits of each sample and then convert it to a signed
         * 32-bit number.
         */
        t0 = _mm_load_si128(&b.vec[0]);
        t3 = _mm_load_si128(&b.vec[1]);
        t1 = _mm_add_epi32(t0, t0);
        t4 = _mm_add_epi32(t3, t3);
        t1 = _mm_sub_epi32(t1, _mm_set1_epi32(1));
        t4 = _mm_sub_epi32(t4, _mm_set1_epi32(1));
        t2 = _mm_mullo_epi16(t1, z0[j]);
        t5 = _mm_mullo_epi16(t4, z1[j]);
        t2 = _mm_add_epi32(t2, t0);
        t5 = _mm_add_epi32(t5, t3);
        _mm_store_si128(_z_bi++, t2);
        _mm_store_si128(_z_bi++, t5);
        _mm_store_si128(_z_sq++, _mm_mullo_epi16(z0[j], z0[j]));
        _mm_store_si128(_z_sq++, _mm_mullo_epi16(z1[j], z1[j]));
    }
}

#define SAMPLES_N (8 * (1 << 5))

void test_gaussian0()
{
    ALIGNED_INT32(SAMPLES_N) z0_bimodal, z1_bimodal, z0_square, z1_square;
    uint8_t seed[32] = {0};
    sampler_state ss0, ss1;
    size_t i;

    memset(&ss0, 0, sizeof(ss0));
    memset(&z0_bimodal, 0, sizeof(z0_bimodal));
    memset(&z0_square, 0, sizeof(z0_square));
    sampler_init(&ss0, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 4) {
        gaussian0_ref_u24_4w(&ss0, &z0_bimodal.coeffs[i],
                             &z0_square.coeffs[i]);
    }
    memset(&ss1, 0, sizeof(ss1));
    memset(&z1_bimodal, 0, sizeof(z1_bimodal));
    memset(&z1_square, 0, sizeof(z1_square));
    sampler_init(&ss1, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 4) {
        gaussian0_sse2_4w(&ss1, &z1_bimodal.coeffs[i],
                          &z1_square.coeffs[i]);
    }
    for (i = 0; i < SAMPLES_N; i++) {
        z1_bimodal.coeffs[i] =
            (int32_t)(int16_t)(z1_bimodal.coeffs[i]);
    }
    if (memcmp(z0_bimodal.coeffs, z1_bimodal.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0 ||
        memcmp(z0_square.coeffs, z1_square.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0) {
        printf("gaussian0_ref_u24_4w != gaussian0_sse2_4w\n");
    }

    memset(&ss0, 0, sizeof(ss0));
    memset(&z0_bimodal, 0, sizeof(z0_bimodal));
    memset(&z0_square, 0, sizeof(z0_square));
    sampler_init(&ss0, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 8) {
        gaussian0_ref_u24_8w(&ss0, &z0_bimodal.coeffs[i],
                             &z0_square.coeffs[i]);
    }
    memset(&ss1, 0, sizeof(ss1));
    memset(&z1_bimodal, 0, sizeof(z1_bimodal));
    memset(&z1_square, 0, sizeof(z1_square));
    sampler_init(&ss1, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 8) {
        gaussian0_sse2_8w(&ss1, &z1_bimodal.coeffs[i],
                          &z1_square.coeffs[i]);
    }
    for (i = 0; i < SAMPLES_N; i++) {
        z1_bimodal.coeffs[i] =
            (int32_t)(int16_t)(z1_bimodal.coeffs[i]);
    }
    if (memcmp(z0_bimodal.coeffs, z1_bimodal.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0 ||
        memcmp(z0_square.coeffs, z1_square.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0) {
        printf("gaussian0_ref_u24_8w != gaussian0_sse2_8w\n");
    }

    memset(&ss0, 0, sizeof(ss0));
    memset(&z0_bimodal, 0, sizeof(z0_bimodal));
    memset(&z0_square, 0, sizeof(z0_square));
    sampler_init(&ss0, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 16) {
        gaussian0_ref_u24_16w(&ss0, &z0_bimodal.coeffs[i],
                              &z0_square.coeffs[i]);
    }
    memset(&ss1, 0, sizeof(ss1));
    memset(&z1_bimodal, 0, sizeof(z1_bimodal));
    memset(&z1_square, 0, sizeof(z1_square));
    sampler_init(&ss1, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 16) {
        gaussian0_sse2_16w(&ss1, &z1_bimodal.coeffs[i],
                           &z1_square.coeffs[i]);
    }
    for (i = 0; i < SAMPLES_N; i++) {
        z1_bimodal.coeffs[i] =
            (int32_t)(int16_t)(z1_bimodal.coeffs[i] & 0xffff);
    }
    if (memcmp(z0_bimodal.coeffs, z1_bimodal.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0 ||
        memcmp(z0_square.coeffs, z1_square.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0) {
        printf("gaussian0_ref_u24_16w != gaussian0_sse2_16w\n");
    }
}

#define WARMUP_N 1000
// 136*4-(136*4%3)=543
// 543*3 % 9 = 0
#define TESTS_N ((543 * 3) * 100)

void speed_gaussian0()
{
    sampler_state ss0;
    uint8_t seed[32] = {0};
    ALIGNED_INT32(64) z0_bimodal, z0_square;

    init_perf_counters();

    PERF(gaussian0_ref(&ss0, &z0_bimodal.coeffs[0], &z0_square.coeffs[0]),
         gaussian0_ref, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
         TESTS_N);
    PERF(gaussian0_ref_u24(&ss0, &z0_bimodal.coeffs[0],
                           &z0_square.coeffs[0]),
         gaussian0_ref_u24, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
         TESTS_N);

    PERF_N(gaussian0_sse2_4w(&ss0, &z0_bimodal.coeffs[0],
                             &z0_square.coeffs[0]),
           gaussian0_sse2_4w, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
           TESTS_N, 4);
    PERF_N(gaussian0_sse2_8w(&ss0, &z0_bimodal.coeffs[0],
                             &z0_square.coeffs[0]),
           gaussian0_sse2_8w, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
           TESTS_N, 8);
    PERF_N(gaussian0_sse2_16w(&ss0, &z0_bimodal.coeffs[0],
                              &z0_square.coeffs[0]),
           gaussian0_sse2_16w, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
           TESTS_N, 16);
}

int main(void)
{
    test_gaussian0();
    speed_gaussian0();

    return 0;
}