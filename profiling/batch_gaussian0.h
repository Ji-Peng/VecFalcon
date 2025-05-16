#ifndef BATCH_GAUSSIAN0_H
#define BATCH_GAUSSIAN0_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>  // for posix_memalign

#include "inner.h"

// from sign_inner.h
typedef struct sampler_state {
#if FNDSA_SHAKE256X8
    shake256x8_context pc;
#elif FNDSA_SHAKE256X4
    shake256x4_context pc;
#else
    shake_context pc;
#endif
    unsigned logn;
    struct gaussian0_store *gauss_store;
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

/* see sign_inner.h */
void sampler_init(sampler_state *ss, unsigned logn, const void *seed,
                  size_t seed_len)
{
    prng_init(&ss->pc, seed, seed_len);
    ss->logn = logn;
}

#if GAUSSIAN0_REF == 1
// REF implementation is used for testing correctness
#    define ALIGNED_INT32(N)   \
        union {                \
            int32_t coeffs[N]; \
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

/**
 * Returns the number of samples.
 */
#    if (RV64 == 1 && RVV_VLEN256 == 1) || (FNDSA_NEON == 1)
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
{
    int32_t *_z_bi = z_bimodal;
    int32_t *_z_sq = z_square;

    int32_t z[64] = {0};
    for (size_t j = 0; j < 64; j++) {
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
    uint64_t b_64b = prng_next_u64(&ss->pc);
    for (size_t j = 0; j < 64; j++) {
        int32_t b = (b_64b >> j) & 1;
        *_z_bi++ = b + ((b << 1) - 1) * z[j];
        *_z_sq++ = z[j] * z[j];
    }
    return 64;
}
#    elif (FNDSA_RV64D == 1)
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
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
        *_z_bi++ = b + ((b << 1) - 1) * z[j];
        *_z_sq++ = z[j] * z[j];
    }
    return 64;
}
#    elif FNDSA_AVX2 == 1 || FNDSA_SSE2 == 1
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
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
        for (size_t i = 0; i < (sizeof GAUSS0) / sizeof(GAUSS0[0]); i++) {
            uint32_t cc;
            cc = (v0 - GAUSS0[i][2]) >> 31;
            cc = (v1 - GAUSS0[i][1] - cc) >> 31;
            cc = (v2 - GAUSS0[i][0] - cc) >> 31;
            z[j] += (int32_t)cc;
        }
    }
    unsigned b_16b = prng_next_u16(&ss->pc);
    for (size_t j = 0; j < 16; j++) {
        int32_t b = (b_16b >> j) & 1;
        *_z_bi++ = b + ((b << 1) - 1) * z[j];
        *_z_sq++ = z[j] * z[j];
    }
    return 16;
}
#    elif FNDSA_AVX512F == 1
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
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
        *_z_bi++ = b + ((b << 1) - 1) * z[j];
        *_z_sq++ = z[j] * z[j];
    }
    return 32;
}
#    endif

#else

#    if FNDSA_AVX512F == 1
#        define U32X16(W)                                          \
            {                                                      \
                {                                                  \
                    W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W \
                }                                                  \
            }

typedef union {
    uint32_t u32[16];
    __m512i zmm;
} gauss0_32x16;

typedef union {
    uint32_t u32[3][16];
    __m512i zmm[3];
} prn_24x3_16w;

#        define ALIGNED_INT32(N)            \
            union {                         \
                int32_t coeffs[N];          \
                __m512i vec[(N + 15) / 16]; \
            }

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

static inline int gaussian0(sampler_state *ss, void *z_bimodal,
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
    return 32;
}

#    elif FNDSA_AVX2 == 1
#        define U32X8(W)                   \
            {                              \
                {                          \
                    W, W, W, W, W, W, W, W \
                }                          \
            }

typedef union {
    uint32_t u32[8];
    __m256i ymm;
} gauss0_32x8;

typedef union {
    uint32_t u32[3][8];
    __m256i ymm[3];
} prn_24x3_8w;

const gauss0_32x8 GAUSS0_AVX2[][3] = {
    {U32X8(10745844), U32X8(3068844), U32X8(3741698)},
    {U32X8(5559083), U32X8(1580863), U32X8(8248194)},
    {U32X8(2260429), U32X8(13669192), U32X8(2736639)},
    {U32X8(708981), U32X8(4421575), U32X8(10046180)},
    {U32X8(169348), U32X8(7122675), U32X8(4136815)},
    {U32X8(30538), U32X8(13063405), U32X8(7650655)},
    {U32X8(4132), U32X8(14505003), U32X8(7826148)},
    {U32X8(417), U32X8(16768101), U32X8(11363290)},
    {U32X8(31), U32X8(8444042), U32X8(8086568)},
    {U32X8(1), U32X8(12844466), U32X8(265321)},
    {U32X8(0), U32X8(1232676), U32X8(13644283)},
    {U32X8(0), U32X8(38047), U32X8(9111839)},
    {U32X8(0), U32X8(870), U32X8(6138264)},
    {U32X8(0), U32X8(14), U32X8(12545723)},
    {U32X8(0), U32X8(0), U32X8(3104126)},
    {U32X8(0), U32X8(0), U32X8(28824)},
    {U32X8(0), U32X8(0), U32X8(198)},
    {U32X8(0), U32X8(0), U32X8(1)}};

#        define ALIGNED_INT32(N)          \
            union {                       \
                int32_t coeffs[N];        \
                __m256i vec[(N + 7) / 8]; \
            }

/**
 * @param samples must be aligned on a 32-byte boundary.
 *
 * Returns the number of samples.
 */
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
{
    prn_24x3_8w prn[2];
    ALIGNED_INT32(16) b;
    __m256i *_z_bi = (__m256i *)z_bimodal;
    __m256i *_z_sq = (__m256i *)z_square;

    /* Get random 72-bit values, with 3x24-bit form. */
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 8; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    unsigned b_16b = prng_next_u16(&ss->pc);
    for (size_t i = 0; i < 16; i += 4) {
        b.coeffs[i] = (b_16b >> i) & 1;
        b.coeffs[i + 1] = (b_16b >> (i + 1)) & 1;
        b.coeffs[i + 2] = (b_16b >> (i + 2)) & 1;
        b.coeffs[i + 3] = (b_16b >> (i + 3)) & 1;
    }
    __m256i z0 = _mm256_setzero_si256(), z1 = _mm256_setzero_si256();
    __m256i cc0, cc1;
    __m256i t0, t1, t2, t3, t4, t5;
    for (size_t i = 0; i < (sizeof GAUSS0_AVX2) / sizeof(GAUSS0_AVX2[0]);
         i++) {
        // load pre-computed table
        t2 = _mm256_loadu_si256(&GAUSS0_AVX2[i][2].ymm);
        t1 = _mm256_loadu_si256(&GAUSS0_AVX2[i][1].ymm);
        t0 = _mm256_loadu_si256(&GAUSS0_AVX2[i][0].ymm);
        // if v0 < GAUSS0[i][2] then -1 else 0
        cc0 = _mm256_cmpgt_epi32(t2, prn[0].ymm[0]);
        cc1 = _mm256_cmpgt_epi32(t2, prn[1].ymm[0]);
        // cc = (v1 - GAUSS0[i][1] - cc) >> 31;
        cc0 = _mm256_add_epi32(cc0, prn[0].ymm[1]);
        cc1 = _mm256_add_epi32(cc1, prn[1].ymm[1]);
        cc0 = _mm256_sub_epi32(cc0, t1);
        cc1 = _mm256_sub_epi32(cc1, t1);
        cc0 = _mm256_srli_epi32(cc0, 31);
        cc1 = _mm256_srli_epi32(cc1, 31);
        // cc = (v2 - GAUSS0[i][0] - cc) >> 31;
        cc0 = _mm256_sub_epi32(prn[0].ymm[2], cc0);
        cc1 = _mm256_sub_epi32(prn[1].ymm[2], cc1);
        cc0 = _mm256_sub_epi32(cc0, t0);
        cc1 = _mm256_sub_epi32(cc1, t0);
        cc0 = _mm256_srli_epi32(cc0, 31);
        cc1 = _mm256_srli_epi32(cc1, 31);
        z0 = _mm256_add_epi32(z0, cc0);
        z1 = _mm256_add_epi32(z1, cc1);
    }
    t0 = _mm256_load_si256(&b.vec[0]);
    t3 = _mm256_load_si256(&b.vec[1]);
    t1 = _mm256_add_epi32(t0, t0);
    t4 = _mm256_add_epi32(t3, t3);
    t1 = _mm256_sub_epi32(t1, _mm256_set1_epi32(1));
    t4 = _mm256_sub_epi32(t4, _mm256_set1_epi32(1));
    t2 = _mm256_mullo_epi32(t1, z0);
    t5 = _mm256_mullo_epi32(t4, z1);
    t2 = _mm256_add_epi32(t2, t0);
    t5 = _mm256_add_epi32(t5, t3);
    _mm256_store_si256(_z_bi, t2);
    _mm256_store_si256(_z_bi + 1, t5);
    /**
     * Each sample is in the range [0,18], so we can use the 16-bit
     * multiplication instruction.
     */
    _mm256_store_si256(_z_sq, _mm256_mullo_epi16(z0, z0));
    _mm256_store_si256(_z_sq + 1, _mm256_mullo_epi16(z1, z1));
    return 16;
}
#    elif FNDSA_SSE2 == 1
#        define U32X4(W)       \
            {                  \
                {              \
                    W, W, W, W \
                }              \
            }

typedef union {
    uint32_t u32[4];
    __m128i xmm;
} gauss0_32x4;

typedef union {
    uint32_t u32[3][4];
    __m128i xmm[3];
} prn_24x3_4w;

const gauss0_32x4 GAUSS0_SSE2[][3] = {
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

#        define ALIGNED_INT32(N)          \
            union {                       \
                int32_t coeffs[N];        \
                __m128i vec[(N + 3) / 4]; \
            }

/**
 * @param samples must be aligned on a 16-byte boundary.
 * This function generates 16 samples at a time to keep consistent with the
 * AVX2 version.
 * Returns the number of samples.
 */
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
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
        for (size_t i = 0;
             i < (sizeof GAUSS0_SSE2) / sizeof(GAUSS0_SSE2[0]); i++) {
            // load pre-computed table
            t2 = _mm_loadu_si128(&GAUSS0_SSE2[i][2].xmm);
            t1 = _mm_loadu_si128(&GAUSS0_SSE2[i][1].xmm);
            t0 = _mm_loadu_si128(&GAUSS0_SSE2[i][0].xmm);
            // if v0 < GAUSS0[i][2] then -1 else 0
            cc0 = _mm_cmpgt_epi32(t2, prn[0].xmm[0]);
            cc1 = _mm_cmpgt_epi32(t2, prn[1].xmm[0]);
            // cc = (v1 - GAUSS0[i][1] - cc) >> 31;
            cc0 = _mm_add_epi32(cc0, prn[0].xmm[1]);
            cc1 = _mm_add_epi32(cc1, prn[1].xmm[1]);
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
    unsigned b_16b = prng_next_u16(&ss->pc);
    for (size_t j = 0; j < 2; j++, b_16b >>= 8) {
        for (size_t i = 0; i < 8; i += 4) {
            b.coeffs[i] = (b_16b >> (i)) & 1;
            b.coeffs[i + 1] = (b_16b >> (i + 1)) & 1;
            b.coeffs[i + 2] = (b_16b >> (i + 2)) & 1;
            b.coeffs[i + 3] = (b_16b >> (i + 3)) & 1;
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
    return 16;
}
#    elif RV64 == 1 && RVV_VLEN256 == 1
typedef union {
    uint32_t u32[3][8] __attribute__((aligned(32)));
} prn_24x3_8w;

#        define ALIGNED_INT32(N)                                \
            union {                                             \
                int32_t coeffs[N] __attribute__((aligned(32))); \
            }
extern void gaussian0_rvv_bisq(int32_t *z_bimodal, int32_t *z_square,
                               uint32_t *prn, uint32_t *bs, size_t n);
#        if BATCH_GAUSSIAN0_SIZE % 64 == 0
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
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
    return 64;
}
#        else
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
{
    ALIGNED_INT32(16) b_16bs;
    prn_24x3_8w prn[2];
    int32_t *_z_bi = (int32_t *)z_bimodal;
    int32_t *_z_sq = (int32_t *)z_square;

    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 8; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    uint16_t b_16b = prng_next_u16(&ss->pc);
    for (int i = 0; i < 16; i += 4) {
        b_16bs.coeffs[i] = (b_16b >> i) & 1;
        b_16bs.coeffs[i + 1] = (b_16b >> (i + 1)) & 1;
        b_16bs.coeffs[i + 2] = (b_16b >> (i + 2)) & 1;
        b_16bs.coeffs[i + 3] = (b_16b >> (i + 3)) & 1;
    }
    gaussian0_rvv_bisq(_z_bi, _z_sq, &prn[0].u32[0][0], b_16bs.coeffs, 2);
    return 16;
}
#        endif
#    elif FNDSA_RV64D == 1
#        define ALIGNED_INT32(N)   \
            union {                \
                int32_t coeffs[N]; \
            }
extern void gaussian0_rv64im_nw(int32_t *r, uint64_t *prn, size_t n_way);
#        if BATCH_GAUSSIAN0_SIZE % 64 == 0
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
{
    uint64_t prn[64][2];
    int32_t *_z_bi = (int32_t *)z_bimodal;
    int32_t *_z_sq = (int32_t *)z_square;
    int32_t z[64];

    for (int j = 0; j < 64; j++) {
        prn[j][0] = prng_next_u64(&ss->pc);
        prn[j][1] = prng_next_u8(&ss->pc);
    }
    gaussian0_rv64im_nw(z, &prn[0][0], 64);
    uint64_t b_64b = prng_next_u64(&ss->pc);
    for (size_t j = 0; j < 64; j += 4) {
        int32_t b0 = (b_64b >> j) & 1;
        int32_t b1 = (b_64b >> (j + 1)) & 1;
        int32_t b2 = (b_64b >> (j + 2)) & 1;
        int32_t b3 = (b_64b >> (j + 3)) & 1;
        int32_t m0 = (b0 << 1);
        int32_t m1 = (b1 << 1);
        int32_t m2 = (b2 << 1);
        int32_t m3 = (b3 << 1);
        m0 = m0 - 1;
        m1 = m1 - 1;
        m2 = m2 - 1;
        m3 = m3 - 1;
        m0 = m0 * z[j];
        m1 = m1 * z[j + 1];
        m2 = m2 * z[j + 2];
        m3 = m3 * z[j + 3];
        _z_bi[j] = b0 + m0;
        _z_bi[j + 1] = b1 + m1;
        _z_bi[j + 2] = b2 + m2;
        _z_bi[j + 3] = b3 + m3;
        _z_sq[j] = z[j] * z[j];
        _z_sq[j + 1] = z[j + 1] * z[j + 1];
        _z_sq[j + 2] = z[j + 2] * z[j + 2];
        _z_sq[j + 3] = z[j + 3] * z[j + 3];
    }
    return 64;
}
#        else
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
{
    uint64_t prn[16][2];
    int32_t *_z_bi = (int32_t *)z_bimodal;
    int32_t *_z_sq = (int32_t *)z_square;
    int32_t z[16];

    for (int j = 0; j < 16; j++) {
        prn[j][0] = prng_next_u64(&ss->pc);
        prn[j][1] = prng_next_u8(&ss->pc);
    }
    gaussian0_rv64im_nw(z, &prn[0][0], 16);
    uint16_t b_16b = prng_next_u16(&ss->pc);
    for (size_t j = 0; j < 16; j += 4) {
        int32_t b0 = (b_64b >> j) & 1;
        int32_t b1 = (b_64b >> (j + 1)) & 1;
        int32_t b2 = (b_64b >> (j + 2)) & 1;
        int32_t b3 = (b_64b >> (j + 3)) & 1;
        int32_t m0 = (b0 << 1);
        int32_t m1 = (b1 << 1);
        int32_t m2 = (b2 << 1);
        int32_t m3 = (b3 << 1);
        m0 = m0 - 1;
        m1 = m1 - 1;
        m2 = m2 - 1;
        m3 = m3 - 1;
        m0 = m0 * z[j];
        m1 = m1 * z[j + 1];
        m2 = m2 * z[j + 2];
        m3 = m3 * z[j + 3];
        _z_bi[j] = b0 + m0;
        _z_bi[j + 1] = b1 + m1;
        _z_bi[j + 2] = b2 + m2;
        _z_bi[j + 3] = b3 + m3;
        _z_sq[j] = z[j] * z[j];
        _z_sq[j + 1] = z[j + 1] * z[j + 1];
        _z_sq[j + 2] = z[j + 2] * z[j + 2];
        _z_sq[j + 3] = z[j + 3] * z[j + 3];
    }
    return 16;
}
#        endif
#    elif FNDSA_NEON == 1
typedef union {
    uint32_t u32[3][4] __attribute__((aligned(16)));
} prn_24x3_4w;

#        define ALIGNED_INT32(N)                                \
            union {                                             \
                int32_t coeffs[N] __attribute__((aligned(16))); \
            }
extern void gaussian0_neon_bisq(int32_t *z_bi, int32_t *z_sq,
                                uint32_t *prn, uint32_t *prn_bisq,
                                size_t n_way);
#        if BATCH_GAUSSIAN0_SIZE % 64 == 0
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
{
    ALIGNED_INT32(64) b_64bs;
    prn_24x3_4w prn[16];
    int32_t *_z_bi = (int32_t *)z_bimodal;
    int32_t *_z_sq = (int32_t *)z_square;

    for (int j = 0; j < 16; j++)
        for (int i = 0; i < 4; i++) {
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
    gaussian0_neon_bisq(_z_bi, _z_sq, &prn[0].u32[0][0],
                        (uint32_t *)b_64bs.coeffs, 16);
    return 64;
}
#        else
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
{
    ALIGNED_INT32(16) b_16bs;
    prn_24x3_4w prn[4];
    int32_t *_z_bi = (int32_t *)z_bimodal;
    int32_t *_z_sq = (int32_t *)z_square;

    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    uint16_t b_16b = prng_next_u16(&ss->pc);
    for (int i = 0; i < 16; i += 4) {
        b_16bs.coeffs[i] = (b_16b >> i) & 1;
        b_16bs.coeffs[i + 1] = (b_16b >> (i + 1)) & 1;
        b_16bs.coeffs[i + 2] = (b_16b >> (i + 2)) & 1;
        b_16bs.coeffs[i + 3] = (b_16b >> (i + 3)) & 1;
    }
    gaussian0_neon_bisq(_z_bi, _z_sq, &prn[0].u32[0][0], b_16bs.coeffs, 4);
    return 16;
}

#        endif
#    endif
#endif

/**
 * 8*16*sizeof(int32_t) = 0.5 KB
 * This number must be a multiple of 16, because for the AVX2
 * implementation of gaussian0 we generate 16 samples at a time
 */
#ifndef BATCH_GAUSSIAN0_SIZE
#    define BATCH_GAUSSIAN0_SIZE (8 * 16)
#endif

#if BATCH_GAUSSIAN0_SIZE % 16 != 0
#    error "BATCH_GAUSSIAN0_SIZE must be a multiple of 16"
#endif

struct sampler_state;
struct gaussian0_store;

typedef struct gaussian0_store {
    ALIGNED_INT32(BATCH_GAUSSIAN0_SIZE) _z_bi;
    ALIGNED_INT32(BATCH_GAUSSIAN0_SIZE) _z_sq;
    size_t batch_size;
    /** current_pos==batch_size means the store is empty */
    size_t current_pos;
    struct sampler_state *ss;
} GAUSSIAN0_STORE;

/**
 * Fill the store with a call of the gaussian0 function.
 * This function is called when 1) initializing the store, or
 * 2) when the current store is empty.
 */
static inline void GAUSSIAN0_STORE_fill(GAUSSIAN0_STORE *store)
{
    size_t i;
    size_t num;

    for (i = 0; i < store->batch_size; i += num) {
        num = gaussian0(store->ss, &store->_z_bi.coeffs[i],
                        &store->_z_sq.coeffs[i]);
    }
    /** The store is full */
    store->current_pos = 0;
};

static inline GAUSSIAN0_STORE *GAUSSIAN0_STORE_new(sampler_state *ss)
{
    GAUSSIAN0_STORE *store = NULL;

    // posix_memalign is used to allocate aligned memory space
    int ret = posix_memalign((void **)&store, 32, sizeof(*store));
    if (ret != 0) {
        fprintf(stderr,
                "GAUSSIAN0_STORE_new Error: Failed to allocate "
                "memory for store\n");
        return NULL;
    }
    store->batch_size = BATCH_GAUSSIAN0_SIZE;
    store->current_pos = 0;
    store->ss = ss;
    GAUSSIAN0_STORE_fill(store);

    return store;
}

static inline void GAUSSIAN0_STORE_free(GAUSSIAN0_STORE *store)
{
    if (store == NULL) {
        return;
    }
    free(store);
}

static inline void GAUSSIAN0_STORE_get_next(GAUSSIAN0_STORE *store,
                                            int32_t *z_bimodal,
                                            int32_t *z_square)
{
    if (store->current_pos >= store->batch_size) {
        GAUSSIAN0_STORE_fill(store);
    }
#if FNDSA_AVX2 == 0 && FNDSA_SSE2 == 1
    /**
     * For SSE2, we must use _mm_mullo_epi16 to turn into bimodal
     * distribution, so the following type conversion is necessary.
     */
    *z_bimodal = (int32_t)(int16_t)store->_z_bi.coeffs[store->current_pos];
#else
    *z_bimodal = store->_z_bi.coeffs[store->current_pos];
#endif
    *z_square = store->_z_sq.coeffs[store->current_pos];
    store->current_pos++;
}

#endif  // BATCH_GAUSSIAN0_H