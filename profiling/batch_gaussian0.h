#ifndef BATCH_GAUSSIAN0_H
#define BATCH_GAUSSIAN0_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>  // for posix_memalign

#include "inner.h"

// from sign_inner.h
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

/* see sign_inner.h */
void sampler_init(sampler_state *ss, unsigned logn, const void *seed,
                  size_t seed_len)
{
    prng_init(&ss->pc, seed, seed_len);
    ss->logn = logn;
}

#if FNDSA_AVX2 == 1 && BATCH_GAUSSIAN0 == 1
#    define BATCH_GAUSSIAN0_AVX2 1
#elif FNDSA_SSE2 == 1 && BATCH_GAUSSIAN0 == 1
#    define BATCH_GAUSSIAN0_SSE2 1
#elif BATCH_GAUSSIAN0 == 1
#    define BATCH_GAUSSIAN0_REF 1
#else

#endif

#if BATCH_GAUSSIAN0_AVX2 == 1
#    define U32X8(W)                   \
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

static const gauss0_32x8 GAUSS0_AVX2[][3] = {
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

#    undef U32X8

#    define ALIGNED_INT32(N)          \
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
    __m256i *_z_bimodal = (__m256i *)z_bimodal;
    __m256i *_z_square = (__m256i *)z_square;

    /* Get random 72-bit values, with 3x24-bit form. */
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 8; i++) {
            prn[j].u32[0][i] = prng_next_u24(&ss->pc);
            prn[j].u32[1][i] = prng_next_u24(&ss->pc);
            prn[j].u32[2][i] = prng_next_u24(&ss->pc);
        }
    __m256i z0 = _mm256_setzero_si256(), z1 = _mm256_setzero_si256();
    __m256i cc0, cc1;
    __m256i t0, t1, t2, t3, t4, t5;
    /**
     * Through decompilation, we found that the following loop only uses
     * at least 12 ymm registers.
     * The number of AVX2 instructions in the loop is 23.
     */
    for (size_t i = 0; i < (sizeof GAUSS0_AVX2) / sizeof(GAUSS0_AVX2[0]);
         i++) {
        // load pre-computed table
        t2 = _mm256_loadu_si256(&GAUSS0_AVX2[i][2].ymm);
        t1 = _mm256_loadu_si256(&GAUSS0_AVX2[i][1].ymm);
        t0 = _mm256_loadu_si256(&GAUSS0_AVX2[i][0].ymm);
        // cc = (v0 - GAUSS0[i][2]) >> 31;
        cc0 = _mm256_sub_epi32(prn[0].ymm[0], t2);
        cc1 = _mm256_sub_epi32(prn[1].ymm[0], t2);
        cc0 = _mm256_srli_epi32(cc0, 31);
        cc1 = _mm256_srli_epi32(cc1, 31);
        // cc = (v1 - GAUSS0[i][1] - cc) >> 31;
        cc0 = _mm256_sub_epi32(prn[0].ymm[1], cc0);
        cc1 = _mm256_sub_epi32(prn[1].ymm[1], cc1);
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
    ALIGNED_INT32(16) b;
    unsigned b_16b = prng_next_u16(&ss->pc);
    for (size_t i = 0; i < 16; i++) {
        b.coeffs[i] = (b_16b >> i) & 1;
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
    _mm256_store_si256(_z_bimodal, t2);
    _mm256_store_si256(_z_bimodal + 1, t5);
    /**
     * Each sample is in the range [0,18], so we can use the 16-bit
     * multiplication instruction.
     */
    _mm256_store_si256(_z_square, _mm256_mullo_epi16(z0, z0));
    _mm256_store_si256(_z_square + 1, _mm256_mullo_epi16(z1, z1));
    return 16;
}
#elif BATCH_GAUSSIAN0_SSE2 == 1
#    define U32X4(W)       \
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

#    undef U32X4

#    define ALIGNED_INT32(N)          \
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
    __m128i *_z_bimodal = (__m128i *)z_bimodal;
    __m128i *_z_square = (__m128i *)z_square;
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
    unsigned b_16b = prng_next_u16(&ss->pc);
    for (size_t j = 0; j < 2; j++, b_16b >>= 8) {
        for (size_t i = 0; i < 8; i++) {
            b.coeffs[i] = (b_16b >> i) & 1;
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
        _mm_store_si128(_z_bimodal++, t2);
        _mm_store_si128(_z_bimodal++, t5);
        _mm_store_si128(_z_square++, _mm_mullo_epi16(z0[j], z0[j]));
        _mm_store_si128(_z_square++, _mm_mullo_epi16(z1[j], z1[j]));
    }
    return 16;
}
#else
#    define ALIGNED_INT32(N)   \
        union {                \
            int32_t coeffs[N]; \
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
 * Returns the number of samples.
 */
static inline int gaussian0(sampler_state *ss, void *z_bimodal,
                            void *z_square)
{
    int32_t *_z_bimodal = z_bimodal;
    int32_t *_z_square = z_square;

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
    unsigned b_16b = prng_next_u16(&ss->pc);
    for (size_t j = 0; j < 16; j++) {
        // Get a random bit b to turn the sampling into a bimodal
        // distribution.
        int32_t b = (b_16b >> j) & 1;
        *_z_bimodal++ = b + ((b << 1) - 1) * z[j];
        *_z_square++ = z[j] * z[j];
    }
    return 16;
}
#endif

/**
 * 8*16*sizeof(int32_t) = 0.5 KB
 * This number must be a multiple of 16, because for the AVX2
 * implementation of gaussian0 we generate 16 samples at a time
 */
#ifndef BATCH_GAUSSIAN0_SIZE
#    define BATCH_GAUSSIAN0_SIZE (8 * 16)
#endif

typedef struct gaussian0_store {
    size_t batch_size;
    /** current_pos==batch_size means the store is empty */
    size_t current_pos;
    sampler_state *ss;
    ALIGNED_INT32(BATCH_GAUSSIAN0_SIZE) _z_bimodal;
    ALIGNED_INT32(BATCH_GAUSSIAN0_SIZE) _z_square;
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
        num = gaussian0(store->ss, &store->_z_bimodal.coeffs[i],
                        &store->_z_square.coeffs[i]);
    }
    /** The store is full */
    store->current_pos = 0;
};

static inline GAUSSIAN0_STORE *GAUSSIAN0_STORE_new(
    sampler_state *ss)
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

static inline void GAUSSIAN0_STORE_get_next(
    GAUSSIAN0_STORE *store, int32_t *z_bimodal, int32_t *z_square)
{
    if (store->current_pos >= store->batch_size) {
        GAUSSIAN0_STORE_fill(store);
    }
#if BATCH_GAUSSIAN0_SSE2 == 1
    /**
     * For SSE2, we must use _mm_mullo_epi16 to turn into bimodal
     * distribution, so the following type conversion is necessary.
     */
    *z_bimodal =
        (int32_t)(int16_t)store->_z_bimodal.coeffs[store->current_pos];
#else
    *z_bimodal = store->_z_bimodal.coeffs[store->current_pos];
#endif
    *z_square = store->_z_square.coeffs[store->current_pos];
    store->current_pos++;
}

#endif  // BATCH_GAUSSIAN0_H