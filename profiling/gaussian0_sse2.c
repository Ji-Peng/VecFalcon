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
 * Obtained from https://github.com/pornin/c-fn-dsa
 */
int32_t gaussian0_ref(sampler_state *ss)
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
    return z;
}

/**
 * The way to get 72-bit in gaussian0_ref is u64+u8. When using SHAKE256X4,
 * the test vectors of u64+u8 and u24x3 will be different. The critical
 * point is 136*4-60*9=4 bytes. The former will directly trigger refill,
 * while the latter can still get 24 bits.
 */
int32_t gaussian0_ref_u24(sampler_state *ss)
{
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
    return z;
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

void gaussian0_sse2_4w(sampler_state *ss, __m128i *samples)
{
    prn_24x3_4w prn[1];
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
    _mm_store_si128(samples, z0);
}

void gaussian0_sse2_8w(sampler_state *ss, __m128i *samples)
{
    prn_24x3_4w prn[2];
    /* Get random 72-bit values, with 3x24-bit form. */
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 4; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    __m128i z0 = _mm_setzero_si128(), z1 = _mm_setzero_si128();
    __m128i cc0, cc1;
    __m128i t0, t1, t2;
    /**
     * Through decompilation, we found that the following loop only uses
     * at least 12 ymm registers.
     * The number of AVX2 instructions in the loop is 23.
     */
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
    _mm_store_si128(samples, z0);
    _mm_store_si128(samples + 1, z1);
}

#define SAMPLES_N (8 * (1 << 5))

void test_gaussian0()
{
    ALIGNED_INT32(SAMPLES_N) samples0, samples1;
    uint8_t seed[32] = {0};
    sampler_state ss0, ss1;
    __m128i *vecp;
    size_t i;

    memset(&ss0, 0, sizeof(ss0));
    memset(&samples0, 0, sizeof(samples0));
    sampler_init(&ss0, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i++) {
        samples0.coeffs[i] = gaussian0_ref_u24(&ss0);
    }
    memset(&ss1, 0, sizeof(ss1));
    memset(&samples1, 0, sizeof(samples1));
    sampler_init(&ss1, 9, seed, 32);
    for (i = 0, vecp = &samples1.vec[0]; i < SAMPLES_N; i += 4, vecp++) {
        gaussian0_sse2_4w(&ss1, vecp);
    }
    if (memcmp(samples0.coeffs, samples1.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0) {
        printf("gaussian0_ref_u24 != gaussian0_sse2_4w\n");
        // exit(0);
    }

    memset(&ss1, 0, sizeof(ss1));
    memset(&samples1, 0, sizeof(samples1));
    sampler_init(&ss1, 9, seed, 32);
    for (i = 0, vecp = &samples1.vec[0]; i < SAMPLES_N;
         i += 8, vecp += 2) {
        gaussian0_sse2_8w(&ss1, vecp);
    }
    if (memcmp(samples0.coeffs, samples1.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0) {
        printf("gaussian0_ref_u24 != gaussian0_sse2_8w\n");
        // exit(0);
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
    ALIGNED_INT32(64) samples;

    init_perf_counters();

    PERF(gaussian0_ref(&ss0), gaussian0_ref,
         sampler_init(&ss0, 9, seed, 32), WARMUP_N, TESTS_N);
    PERF(gaussian0_ref_u24(&ss0), gaussian0_ref_u24,
         sampler_init(&ss0, 9, seed, 32), WARMUP_N, TESTS_N);

    PERF_N(gaussian0_sse2_4w(&ss0, &samples.vec[0]), gaussian0_sse2_4w,
           sampler_init(&ss0, 9, seed, 32), WARMUP_N, TESTS_N, 4);
    PERF_N(gaussian0_sse2_8w(&ss0, &samples.vec[0]), gaussian0_sse2_8w,
           sampler_init(&ss0, 9, seed, 32), WARMUP_N, TESTS_N, 8);
}

int main(void)
{
    test_gaussian0();
    speed_gaussian0();

    return 0;
}