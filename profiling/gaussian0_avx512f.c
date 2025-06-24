#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "cpucycles.h"
#include "inner.h"

typedef struct {
#if FNDSA_SHAKE256X8
    shake256x8_context pc;
#elif FNDSA_SHAKE256X4
    shake256x4_context pc;
#else
    shake_context pc;
#endif
    unsigned logn;
} sampler_state;

/* We access the PRNG through macros so that they can be overridden by some
   compatiblity tests with the original Falcon implementation. */
#ifndef prng_init
#    if FNDSA_SHAKE256X8
#        define prng_init shake256x8_init
#        define prng_next_u8 shake256x8_next_u8
#        define prng_next_u16 shake256x8_next_u16
#        define prng_next_u24 shake256x8_next_u24
#        define prng_next_u32 shake256x8_next_u32
#        define prng_next_u64 shake256x8_next_u64
#    elif FNDSA_SHAKE256X4
#        define prng_init shake256x4_init
#        define prng_next_u8 shake256x4_next_u8
#        define prng_next_u16 shake256x4_next_u16
#        define prng_next_u24 shake256x4_next_u24
#        define prng_next_u32 shake256x4_next_u32
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
#        define prng_next_u32 shake_next_u32
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

/*
 * Sample an integer value along a half-gaussian distribution centered
 * on zero and standard deviation 1.8205, with a precision of 72 bits.
 *
 * Obtained from https://falcon-sign.info/Falcon-impl-20211101.zip.
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
void gaussian0_avx2_ref(sampler_state *ss, int32_t *z_bimodal,
                        int32_t *z_square)
{
    /*
     * High words.
     */
    const union {
        uint16_t u16[16];
        __m256i ymm[1];
    } rhi15 = {{0x51FB, 0x2A69, 0x113E, 0x0568, 0x014A, 0x003B, 0x0008,
                0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                0x0000, 0x0000}};
    const union {
        uint64_t u64[20];
        __m256i ymm[5];
    } rlo57 = {{0x1F42ED3AC391802, 0x12B181F3F7DDB82, 0x1CDD0934829C1FF,
                0x1754377C7994AE4, 0x1846CAEF33F1F6F, 0x14AC754ED74BD5F,
                0x024DD542B776AE4, 0x1A1FFDC65AD63DA, 0x01F80D88A7B6428,
                0x001C3FDB2040C69, 0x00012CF24D031FB, 0x00000949F8B091F,
                0x0000003665DA998, 0x00000000EBF6EBB, 0x0000000002F5D7E,
                0x000000000007098, 0x0000000000000C6, 0x000000000000001,
                0x000000000000000, 0x000000000000000}};
    uint64_t lo;
    unsigned hi;
    __m256i xhi, rhi, gthi, eqhi, eqm;
    __m256i xlo, gtlo0, gtlo1, gtlo2, gtlo3, gtlo4;
    __m128i t, zt;
    int r;
    /*
     * Get a 72-bit random value and split it into a low part
     * (57 bits) and a high part (15 bits)
     */
    lo = prng_next_u64(&ss->pc);
    hi = prng_next_u8(&ss->pc);

    hi = (hi << 7) | (unsigned)(lo >> 57);
    lo &= 0x1FFFFFFFFFFFFFF;
    /*
     * Broadcast the high part and compare it with the relevant
     * values. We need both a "greater than" and an "equal"
     * comparisons.
     */
    xhi = _mm256_broadcastw_epi16(_mm_cvtsi32_si128(hi));
    rhi = _mm256_loadu_si256(&rhi15.ymm[0]);
    gthi = _mm256_cmpgt_epi16(rhi, xhi);
    eqhi = _mm256_cmpeq_epi16(rhi, xhi);
    /*
     * The result is the number of 72-bit values (among the list of 19)
     * which are greater than the 72-bit random value. We first count
     * all non-zero 16-bit elements in the first eight of gthi. Such
     * elements have value -1 or 0, so we first negate them.
     */
    t = _mm_srli_epi16(_mm256_castsi256_si128(gthi), 15);
    zt = _mm_setzero_si128();
    t = _mm_hadd_epi16(t, zt);
    t = _mm_hadd_epi16(t, zt);
    t = _mm_hadd_epi16(t, zt);
    r = _mm_cvtsi128_si32(t);
    /*
     * We must look at the low bits for all values for which the
     * high bits are an "equal" match; values 8-18 all have the
     * same high bits (0).
     * On 32-bit systems, 'lo' really is two registers, requiring
     * some extra code.
     */
    xlo = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(*(int64_t *)&lo));
    gtlo0 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[0]), xlo);
    gtlo1 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[1]), xlo);
    gtlo2 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[2]), xlo);
    gtlo3 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[3]), xlo);
    gtlo4 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[4]), xlo);
    /*
     * Keep only comparison results that correspond to the non-zero
     * elements in eqhi.
     */
    gtlo0 = _mm256_and_si256(
        gtlo0, _mm256_cvtepi16_epi64(_mm256_castsi256_si128(eqhi)));
    gtlo1 = _mm256_and_si256(
        gtlo1, _mm256_cvtepi16_epi64(
                   _mm256_castsi256_si128(_mm256_bsrli_epi128(eqhi, 8))));
    eqm = _mm256_permute4x64_epi64(eqhi, 0xFF);
    gtlo2 = _mm256_and_si256(gtlo2, eqm);
    gtlo3 = _mm256_and_si256(gtlo3, eqm);
    gtlo4 = _mm256_and_si256(gtlo4, eqm);
    /*
     * Add all values to count the total number of "-1" elements.
     * Since the first eight "high" words are all different, only
     * one element (at most) in gtlo0:gtlo1 can be non-zero; however,
     * if the high word of the random value is zero, then many
     * elements of gtlo2:gtlo3:gtlo4 can be non-zero.
     */
    gtlo0 = _mm256_or_si256(gtlo0, gtlo1);
    gtlo0 = _mm256_add_epi64(_mm256_add_epi64(gtlo0, gtlo2),
                             _mm256_add_epi64(gtlo3, gtlo4));
    t = _mm_add_epi64(_mm256_castsi256_si128(gtlo0),
                      _mm256_extracti128_si256(gtlo0, 1));
    t = _mm_add_epi64(t, _mm_srli_si128(t, 8));
    r -= _mm_cvtsi128_si32(t);
    // Get a random bit b to turn the sampling into a bimodal distribution.
    int32_t b = prng_next_u8(&ss->pc) & 1;
    *z_bimodal = b + ((b << 1) - 1) * r;
    *z_square = r * r;
}

void gaussian0_avx2_ref_core(int32_t *z_bimodal, int32_t *z_square)
{
    /*
     * High words.
     */
    const union {
        uint16_t u16[16];
        __m256i ymm[1];
    } rhi15 = {{0x51FB, 0x2A69, 0x113E, 0x0568, 0x014A, 0x003B, 0x0008,
                0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                0x0000, 0x0000}};
    const union {
        uint64_t u64[20];
        __m256i ymm[5];
    } rlo57 = {{0x1F42ED3AC391802, 0x12B181F3F7DDB82, 0x1CDD0934829C1FF,
                0x1754377C7994AE4, 0x1846CAEF33F1F6F, 0x14AC754ED74BD5F,
                0x024DD542B776AE4, 0x1A1FFDC65AD63DA, 0x01F80D88A7B6428,
                0x001C3FDB2040C69, 0x00012CF24D031FB, 0x00000949F8B091F,
                0x0000003665DA998, 0x00000000EBF6EBB, 0x0000000002F5D7E,
                0x000000000007098, 0x0000000000000C6, 0x000000000000001,
                0x000000000000000, 0x000000000000000}};
    uint64_t lo;
    unsigned hi;
    __m256i xhi, rhi, gthi, eqhi, eqm;
    __m256i xlo, gtlo0, gtlo1, gtlo2, gtlo3, gtlo4;
    __m128i t, zt;
    int r;
    /*
     * Get a 72-bit random value and split it into a low part
     * (57 bits) and a high part (15 bits)
     */
    lo = *z_bimodal << 32 + *z_bimodal;
    hi = *z_square;

    hi = (hi << 7) | (unsigned)(lo >> 57);
    lo &= 0x1FFFFFFFFFFFFFF;
    /*
     * Broadcast the high part and compare it with the relevant
     * values. We need both a "greater than" and an "equal"
     * comparisons.
     */
    xhi = _mm256_broadcastw_epi16(_mm_cvtsi32_si128(hi));
    rhi = _mm256_loadu_si256(&rhi15.ymm[0]);
    gthi = _mm256_cmpgt_epi16(rhi, xhi);
    eqhi = _mm256_cmpeq_epi16(rhi, xhi);
    /*
     * The result is the number of 72-bit values (among the list of 19)
     * which are greater than the 72-bit random value. We first count
     * all non-zero 16-bit elements in the first eight of gthi. Such
     * elements have value -1 or 0, so we first negate them.
     */
    t = _mm_srli_epi16(_mm256_castsi256_si128(gthi), 15);
    zt = _mm_setzero_si128();
    t = _mm_hadd_epi16(t, zt);
    t = _mm_hadd_epi16(t, zt);
    t = _mm_hadd_epi16(t, zt);
    r = _mm_cvtsi128_si32(t);
    /*
     * We must look at the low bits for all values for which the
     * high bits are an "equal" match; values 8-18 all have the
     * same high bits (0).
     * On 32-bit systems, 'lo' really is two registers, requiring
     * some extra code.
     */
    xlo = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(*(int64_t *)&lo));
    gtlo0 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[0]), xlo);
    gtlo1 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[1]), xlo);
    gtlo2 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[2]), xlo);
    gtlo3 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[3]), xlo);
    gtlo4 = _mm256_cmpgt_epi64(_mm256_loadu_si256(&rlo57.ymm[4]), xlo);
    /*
     * Keep only comparison results that correspond to the non-zero
     * elements in eqhi.
     */
    gtlo0 = _mm256_and_si256(
        gtlo0, _mm256_cvtepi16_epi64(_mm256_castsi256_si128(eqhi)));
    gtlo1 = _mm256_and_si256(
        gtlo1, _mm256_cvtepi16_epi64(
                   _mm256_castsi256_si128(_mm256_bsrli_epi128(eqhi, 8))));
    eqm = _mm256_permute4x64_epi64(eqhi, 0xFF);
    gtlo2 = _mm256_and_si256(gtlo2, eqm);
    gtlo3 = _mm256_and_si256(gtlo3, eqm);
    gtlo4 = _mm256_and_si256(gtlo4, eqm);
    /*
     * Add all values to count the total number of "-1" elements.
     * Since the first eight "high" words are all different, only
     * one element (at most) in gtlo0:gtlo1 can be non-zero; however,
     * if the high word of the random value is zero, then many
     * elements of gtlo2:gtlo3:gtlo4 can be non-zero.
     */
    gtlo0 = _mm256_or_si256(gtlo0, gtlo1);
    gtlo0 = _mm256_add_epi64(_mm256_add_epi64(gtlo0, gtlo2),
                             _mm256_add_epi64(gtlo3, gtlo4));
    t = _mm_add_epi64(_mm256_castsi256_si128(gtlo0),
                      _mm256_extracti128_si256(gtlo0, 1));
    t = _mm_add_epi64(t, _mm_srli_si128(t, 8));
    r -= _mm_cvtsi128_si32(t);
    // Get a random bit b to turn the sampling into a bimodal distribution.
    unsigned b = *z_bimodal & 1;
    *z_bimodal = b + ((b << 1) - 1) * r;
    *z_square = r * r;
}

const uint32_t GAUSS0[][3] = {{10745844, 3068844, 3741698},
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

#define U32X16(W)                                          \
    {                                                      \
        {                                                  \
            W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W \
        }                                                  \
    }

typedef union {
    uint32_t u32[16];
    __m512i zmm;
} gauss0_32x16;

const gauss0_32x16 GAUSS0_AVX512[][3] = {
    {U32X16(10745844), U32X16(3068844), U32X16(3741698)},
    {U32X16(5559083), U32X16(1580863), U32X16(8248194)},
    {U32X16(2260429), U32X16(13669192), U32X16(2736639)},
    {U32X16(708981), U32X16(4421575), U32X16(10046180)},
    {U32X16(169348), U32X16(7122675), U32X16(4136815)},
    {U32X16(30538), U32X16(13063405), U32X16(7650655)},
    {U32X16(4132), U32X16(14505003), U32X16(7826148)},
    {U32X16(417), U32X16(16768101), U32X16(11363290)},
    {U32X16(31), U32X16(8444042), U32X16(8086568)},
    {U32X16(1), U32X16(12844466), U32X16(265321)},
    {U32X16(0), U32X16(1232676), U32X16(13644283)},
    {U32X16(0), U32X16(38047), U32X16(9111839)},
    {U32X16(0), U32X16(870), U32X16(6138264)},
    {U32X16(0), U32X16(14), U32X16(12545723)},
    {U32X16(0), U32X16(0), U32X16(3104126)},
    {U32X16(0), U32X16(0), U32X16(28824)},
    {U32X16(0), U32X16(0), U32X16(198)},
    {U32X16(0), U32X16(0), U32X16(1)}};

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

// used for test correctness of gaussian0_avx512f_32w
void gaussian0_ref_u24_32w(sampler_state *ss, void *z_bimodal,
                           void *z_square)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;

    int32_t z[32] = {0};
    for (size_t j = 0; j < 32; j++) {
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

    uint32_t b_32b = prng_next_u32(&ss->pc);
    for (size_t j = 0; j < 32; j++) {
        int32_t b = (b_32b >> j) & 1;
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
    uint32_t u32[3][16];
    __m512i zmm[3];
} prn_24x3_16w;

#define ALIGNED_INT32(N)            \
    union {                         \
        int32_t coeffs[N];          \
        __m512i vec[(N + 15) / 16]; \
    }

void gaussian0_avx512f_32w(sampler_state *ss, void *z_bimodal,
                           void *z_square)
{
    prn_24x3_16w prn[2];
    ALIGNED_INT32(32) b;
    __m512i *_z_bi = (__m512i *)z_bimodal;
    __m512i *_z_sq = (__m512i *)z_square;

    /* Get random 72-bit values, with 3x24-bit form. */
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 16; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    uint32_t b_32b = prng_next_u32(&ss->pc);
    for (size_t i = 0; i < 32; i += 4) {
        b.coeffs[i] = (b_32b >> i) & 1;
        b.coeffs[i + 1] = (b_32b >> (i + 1)) & 1;
        b.coeffs[i + 2] = (b_32b >> (i + 2)) & 1;
        b.coeffs[i + 3] = (b_32b >> (i + 3)) & 1;
    }
    __m512i z0 = _mm512_setzero_si512(), z1 = _mm512_setzero_si512();
    __m512i cc0, cc1;
    __m512i t0, t1, t2, t3, t4, t5;
    for (size_t i = 0;
         i < (sizeof GAUSS0_AVX512) / sizeof(GAUSS0_AVX512[0]); i++) {
        // load pre-computed table
        t2 = _mm512_loadu_si512(&GAUSS0_AVX512[i][2].zmm);
        t1 = _mm512_loadu_si512(&GAUSS0_AVX512[i][1].zmm);
        t0 = _mm512_loadu_si512(&GAUSS0_AVX512[i][0].zmm);
        // cc = (v0 - GAUSS0[i][2]) >> 31;
        cc0 = _mm512_sub_epi32(prn[0].zmm[0], t2);
        cc1 = _mm512_sub_epi32(prn[1].zmm[0], t2);
        cc0 = _mm512_srli_epi32(cc0, 31);
        cc1 = _mm512_srli_epi32(cc1, 31);
        // cc = (v1 - GAUSS0[i][1] - cc) >> 31;
        cc0 = _mm512_sub_epi32(prn[0].zmm[1], cc0);
        cc1 = _mm512_sub_epi32(prn[1].zmm[1], cc1);
        cc0 = _mm512_sub_epi32(cc0, t1);
        cc1 = _mm512_sub_epi32(cc1, t1);
        cc0 = _mm512_srli_epi32(cc0, 31);
        cc1 = _mm512_srli_epi32(cc1, 31);
        // cc = (v2 - GAUSS0[i][0] - cc) >> 31;
        cc0 = _mm512_sub_epi32(prn[0].zmm[2], cc0);
        cc1 = _mm512_sub_epi32(prn[1].zmm[2], cc1);
        cc0 = _mm512_sub_epi32(cc0, t0);
        cc1 = _mm512_sub_epi32(cc1, t0);
        cc0 = _mm512_srli_epi32(cc0, 31);
        cc1 = _mm512_srli_epi32(cc1, 31);
        z0 = _mm512_add_epi32(z0, cc0);
        z1 = _mm512_add_epi32(z1, cc1);
    }
    t0 = _mm512_load_si512(&b.vec[0]);
    t3 = _mm512_load_si512(&b.vec[1]);
    t1 = _mm512_add_epi32(t0, t0);
    t4 = _mm512_add_epi32(t3, t3);
    t1 = _mm512_sub_epi32(t1, _mm512_set1_epi32(1));
    t4 = _mm512_sub_epi32(t4, _mm512_set1_epi32(1));
    t2 = _mm512_mullo_epi32(t1, z0);
    t5 = _mm512_mullo_epi32(t4, z1);
    t2 = _mm512_add_epi32(t2, t0);
    t5 = _mm512_add_epi32(t5, t3);
    _mm512_store_si512(_z_bi, t2);
    _mm512_store_si512(_z_bi + 1, t5);
    /**
     * Each sample is in the range [0,18], so we can use the 16-bit
     * multiplication instruction. However, _mm512_mullo_epi16 relies on
     * the BW extension, so we still use 32-bit multiplication.
     */
    _mm512_store_si512(_z_sq, _mm512_mullo_epi32(z0, z0));
    _mm512_store_si512(_z_sq + 1, _mm512_mullo_epi32(z1, z1));
}

void gaussian0_avx512f_32w_core(void *z_bimodal, void *z_square)
{
    prn_24x3_16w prn[2];
    ALIGNED_INT32(32) b;
    __m512i *_z_bi = (__m512i *)z_bimodal;
    __m512i *_z_sq = (__m512i *)z_square;
    uint32_t *_z_bi_u32 = (uint32_t *)z_bimodal;
    uint32_t *_z_sq_u32 = (uint32_t *)z_square;

    /* Get random 72-bit values, with 3x24-bit form. */
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 16; i++) {
            prn[j].u32[0][i] = *(_z_bi_u32 + i + 48 * j);
            prn[j].u32[1][i] = *(_z_bi_u32 + 16 + i + 48 * j);
            prn[j].u32[2][i] = *(_z_bi_u32 + 32 + i + 48 * j);
        }
    uint32_t b_32b = *(_z_bi_u32 + 96);
    for (size_t i = 0; i < 32; i += 4) {
        b.coeffs[i] = (b_32b >> i) & 1;
        b.coeffs[i + 1] = (b_32b >> (i + 1)) & 1;
        b.coeffs[i + 2] = (b_32b >> (i + 2)) & 1;
        b.coeffs[i + 3] = (b_32b >> (i + 3)) & 1;
    }
    __m512i z0 = _mm512_setzero_si512(), z1 = _mm512_setzero_si512();
    __m512i cc0, cc1;
    __m512i t0, t1, t2, t3, t4, t5;
    for (size_t i = 0;
         i < (sizeof GAUSS0_AVX512) / sizeof(GAUSS0_AVX512[0]); i++) {
        // load pre-computed table
        t2 = _mm512_loadu_si512(&GAUSS0_AVX512[i][2].zmm);
        t1 = _mm512_loadu_si512(&GAUSS0_AVX512[i][1].zmm);
        t0 = _mm512_loadu_si512(&GAUSS0_AVX512[i][0].zmm);
        // cc = (v0 - GAUSS0[i][2]) >> 31;
        cc0 = _mm512_sub_epi32(prn[0].zmm[0], t2);
        cc1 = _mm512_sub_epi32(prn[1].zmm[0], t2);
        cc0 = _mm512_srli_epi32(cc0, 31);
        cc1 = _mm512_srli_epi32(cc1, 31);
        // cc = (v1 - GAUSS0[i][1] - cc) >> 31;
        cc0 = _mm512_sub_epi32(prn[0].zmm[1], cc0);
        cc1 = _mm512_sub_epi32(prn[1].zmm[1], cc1);
        cc0 = _mm512_sub_epi32(cc0, t1);
        cc1 = _mm512_sub_epi32(cc1, t1);
        cc0 = _mm512_srli_epi32(cc0, 31);
        cc1 = _mm512_srli_epi32(cc1, 31);
        // cc = (v2 - GAUSS0[i][0] - cc) >> 31;
        cc0 = _mm512_sub_epi32(prn[0].zmm[2], cc0);
        cc1 = _mm512_sub_epi32(prn[1].zmm[2], cc1);
        cc0 = _mm512_sub_epi32(cc0, t0);
        cc1 = _mm512_sub_epi32(cc1, t0);
        cc0 = _mm512_srli_epi32(cc0, 31);
        cc1 = _mm512_srli_epi32(cc1, 31);
        z0 = _mm512_add_epi32(z0, cc0);
        z1 = _mm512_add_epi32(z1, cc1);
    }
    t0 = _mm512_load_si512(&b.vec[0]);
    t3 = _mm512_load_si512(&b.vec[1]);
    t1 = _mm512_add_epi32(t0, t0);
    t4 = _mm512_add_epi32(t3, t3);
    t1 = _mm512_sub_epi32(t1, _mm512_set1_epi32(1));
    t4 = _mm512_sub_epi32(t4, _mm512_set1_epi32(1));
    t2 = _mm512_mullo_epi32(t1, z0);
    t5 = _mm512_mullo_epi32(t4, z1);
    t2 = _mm512_add_epi32(t2, t0);
    t5 = _mm512_add_epi32(t5, t3);
    _mm512_store_si512(_z_bi, t2);
    _mm512_store_si512(_z_bi + 1, t5);
    /**
     * Each sample is in the range [0,18], so we can use the 16-bit
     * multiplication instruction. However, _mm512_mullo_epi16 relies on
     * the BW extension, so we still use 32-bit multiplication.
     */
    _mm512_store_si512(_z_sq, _mm512_mullo_epi32(z0, z0));
    _mm512_store_si512(_z_sq + 1, _mm512_mullo_epi32(z1, z1));
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
    for (i = 0; i < SAMPLES_N; i += 32) {
        gaussian0_ref_u24_32w(&ss0, &z0_bimodal.coeffs[i],
                              &z0_square.coeffs[i]);
    }
    memset(&ss1, 0, sizeof(ss1));
    memset(&z1_bimodal, 0, sizeof(z1_bimodal));
    memset(&z1_square, 0, sizeof(z1_square));
    sampler_init(&ss1, 9, seed, 32);
    for (i = 0; i < SAMPLES_N; i += 32) {
        gaussian0_avx512f_32w(&ss1, &z1_bimodal.coeffs[i],
                              &z1_square.coeffs[i]);
    }
    if (memcmp(z0_bimodal.coeffs, z1_bimodal.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0 ||
        memcmp(z0_square.coeffs, z1_square.coeffs,
               sizeof(int32_t) * SAMPLES_N) != 0) {
        printf("gaussian0_ref_u24_32w != gaussian0_avx512f_32w\n");
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

    printf("Including prng_next_* subroutines\n");
    PERF(gaussian0_ref(&ss0, &z0_bimodal.coeffs[0], &z0_square.coeffs[0]),
         gaussian0_ref, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
         TESTS_N);
    PERF(gaussian0_ref_u24(&ss0, &z0_bimodal.coeffs[0],
                           &z0_square.coeffs[0]),
         gaussian0_ref_u24, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
         TESTS_N);
    PERF(gaussian0_avx2_ref(&ss0, &z0_bimodal.coeffs[0],
                            &z0_square.coeffs[0]),
         gaussian0_avx2_ref, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
         TESTS_N);
    PERF_N(gaussian0_avx512f_32w(&ss0, &z0_bimodal.coeffs[0],
                                 &z0_square.coeffs[0]),
           gaussian0_avx512f_32w, sampler_init(&ss0, 9, seed, 32),
           WARMUP_N, TESTS_N, 32);

    printf(
        "\nExcluding prng_next_* subroutines; including bimodal/square "
        "calculations\n");
    PERF(gaussian0_ref_core(&z0_bimodal.coeffs[0], &z0_square.coeffs[0]),
         gaussian0_ref_core, sampler_init(&ss0, 9, seed, 32), WARMUP_N,
         TESTS_N);
    PERF(gaussian0_avx2_ref_core(&z0_bimodal.coeffs[0],
                                 &z0_square.coeffs[0]),
         gaussian0_avx2_ref_core, sampler_init(&ss0, 9, seed, 32),
         WARMUP_N, TESTS_N);
    PERF_N(gaussian0_avx512f_32w_core(&z0_bimodal.coeffs[0],
                                      &z0_square.coeffs[0]),
           gaussian0_avx512f_32w_core, sampler_init(&ss0, 9, seed, 32),
           WARMUP_N, TESTS_N, 32);
}

int main(void)
{
    test_gaussian0();
    speed_gaussian0();

    return 0;
}
