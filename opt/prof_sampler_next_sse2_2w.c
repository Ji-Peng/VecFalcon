/*
 * Gaussian sampling.
 */

#include <stdio.h>
#include <stdlib.h>  // for posix_memalign

#include "../profiling/cpucycles.h"
#include "sign_inner.h"

/* Union type to get easier access to values with SIMD intrinsics. */
typedef union {
    fpr f;
#if FNDSA_NEON
    float64x1_t v;
#endif
#if FNDSA_RV64D
    f64 v;
#endif
} fpr_u;

/* 1/(2*(1.8205^2)) */
#define INV_2SQRSIGMA0 FPR(5435486223186882, -55)

/* For logn = 1 to 10, n = 2^logn:
      q = 12289
      gs_norm = (117/100)*sqrt(q)
      bitsec = max(2, n/4)
      eps = 1/sqrt(bitsec*2^64)
      smoothz2n = sqrt(log(4*n*(1 + 1/eps))/pi)/sqrt(2*pi)
      sigma = smoothz2n*gs_norm
      sigma_min = sigma/gs_norm = smoothz2n
   We store precomputed values for 1/sigma and for sigma_min, indexed
   by logn. */
static const fpr_u INV_SIGMA[] = {
    {FPR_ZERO},                   /* unused */
    {FPR(7961475618707097, -60)}, /* 0.0069054793295940881528 */
    {FPR(7851656902127320, -60)}, /* 0.0068102267767177965681 */
    {FPR(7746260754658859, -60)}, /* 0.0067188101910722700565 */
    {FPR(7595833604889141, -60)}, /* 0.0065883354370073655600 */
    {FPR(7453842886538220, -60)}, /* 0.0064651781207602890978 */
    {FPR(7319528409832599, -60)}, /* 0.0063486788828078985744 */
    {FPR(7192222552237877, -60)}, /* 0.0062382586529084365056 */
    {FPR(7071336252758509, -60)}, /* 0.0061334065020930252290 */
    {FPR(6956347512113097, -60)}, /* 0.0060336696681577231923 */
    {FPR(6846791885593314, -60)}  /* 0.0059386453095331150985 */
};
static const fpr_u SIGMA_MIN[] = {
    {FPR_ZERO},                   /* unused */
    {FPR(5028307297130123, -52)}, /* 1.1165085072329102589 */
    {FPR(5098636688852518, -52)}, /* 1.1321247692325272406 */
    {FPR(5168009084304506, -52)}, /* 1.1475285353733668685 */
    {FPR(5270355833453349, -52)}, /* 1.1702540788534828940 */
    {FPR(5370752584786614, -52)}, /* 1.1925466358390344011 */
    {FPR(5469306724145091, -52)}, /* 1.2144300507766139921 */
    {FPR(5566116128735780, -52)}, /* 1.2359260567719808790 */
    {FPR(5661270305715104, -52)}, /* 1.2570545284063214163 */
    {FPR(5754851361258101, -52)}, /* 1.2778336969128335860 */
    {FPR(5846934829975396, -52)}  /* 1.2982803343442918540 */
};
static const union {
    fpr f[2];
    __m128d x;
} SIGMA_MINx2[] = {
    {FPR_ZERO, FPR_ZERO}, /* unused */
    {FPR(5028307297130123, -52), FPR(5028307297130123, -52)},
    {FPR(5098636688852518, -52), FPR(5098636688852518, -52)},
    {FPR(5168009084304506, -52), FPR(5168009084304506, -52)},
    {FPR(5270355833453349, -52), FPR(5270355833453349, -52)},
    {FPR(5370752584786614, -52), FPR(5370752584786614, -52)},
    {FPR(5469306724145091, -52), FPR(5469306724145091, -52)},
    {FPR(5566116128735780, -52), FPR(5566116128735780, -52)},
    {FPR(5661270305715104, -52), FPR(5661270305715104, -52)},
    {FPR(5754851361258101, -52), FPR(5754851361258101, -52)},
    {FPR(5846934829975396, -52), FPR(5846934829975396, -52)},
};

/* log(2) */
#define LOG2 FPR(6243314768165359, -53)

/* 1/log(2) */
#define INV_LOG2 FPR(6497320848556798, -52)

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

#if FNDSA_AVX2 == 1
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
#elif FNDSA_SSE2 == 1
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
        // printf(
        //     "GAUSSIAN0_STORE_fill will be called in "
        //     "GAUSSIAN0_STORE_get_next\n");
        GAUSSIAN0_STORE_fill(store);
    }
#if FNDSA_AVX2 == 0 && FNDSA_SSE2 == 1
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

/* see sign_inner.h */
void sampler_init(sampler_state *ss, unsigned logn, const void *seed,
                  size_t seed_len)
{
    prng_init(&ss->pc, seed, seed_len);
    ss->logn = logn;
    ss->gauss_store = GAUSSIAN0_STORE_new(ss);
}

void sampler_free(sampler_state *ss)
{
    GAUSSIAN0_STORE_free(ss->gauss_store);
}

/* ========================= SSE2 IMPLEMENTATION =========================
 */

/* Input: 0 <= x < log(2)
   Output: trunc(x*2^63) */
TARGET_SSE2
static inline int64_t mtwop63(__m128d x)
{
#if FNDSA_64
    static const union {
        fpr f[2];
        __m128d x;
    } twop63 = {{
        FPR(4503599627370496, 11),
        FPR(4503599627370496, 11),
    }};

    return _mm_cvttsd_si64(_mm_mul_sd(x, twop63.x));
#else
    /* 32-bit x86 does not have an SSE2 opcode to convert floating-point
       values to 64-bit integers, only 32-bit signed integers. We must
       do the conversion in three steps with factor 2^21. */
    static const union {
        fpr f[2];
        __m128d x;
    } twop21 = {{
        FPR(4503599627370496, -31),
        FPR(4503599627370496, -31),
    }};
    x = _mm_mul_sd(x, twop21.x);
    int32_t z2 = _mm_cvttsd_si32(x);
    x = _mm_sub_sd(x, _mm_cvtsi32_sd(_mm_setzero_pd(), z2));
    x = _mm_mul_sd(x, twop21.x);
    int32_t z1 = _mm_cvttsd_si32(x);
    x = _mm_sub_sd(x, _mm_cvtsi32_sd(_mm_setzero_pd(), z1));
    x = _mm_mul_sd(x, twop21.x);
    int32_t z0 = _mm_cvttsd_si32(x);
    return ((int64_t)z2 << 42) + ((int64_t)z1 << 21) + (int64_t)z0;
#endif
}

/* Compute ccs*exp(-x)*2^63, rounded to an integer. This function assumes
   that 0 <= x < log(2), and 0 <= ccs <= 1. It returns a value in [0,2^63].
 */
TARGET_SSE2
static inline uint64_t expm_p63(__m128d x, __m128d ccs)
{
    /* The polynomial approximation of exp(-x) is from FACCT:
          https://eprint.iacr.org/2018/1234
       Specifically, the values are extracted from the implementation
       referenced by the FACCT paper, available at:
          https://github.com/raykzhao/gaussian  */
    static const uint64_t EXPM_COEFFS[] = {
        0x00000004741183A3, 0x00000036548CFC06, 0x0000024FDCBF140A,
        0x0000171D939DE045, 0x0000D00CF58F6F84, 0x000680681CF796E3,
        0x002D82D8305B0FEA, 0x011111110E066FD0, 0x0555555555070F00,
        0x155555555581FF00, 0x400000000002B400, 0x7FFFFFFFFFFF4800,
        0x8000000000000000};

    uint64_t y = EXPM_COEFFS[0];
    uint64_t z = (uint64_t)mtwop63(x) << 1;
    uint64_t w = (uint64_t)mtwop63(ccs) << 1;
#if FNDSA_64
    /* On 64-bit x86, we have 64x64->128 multiplication, then we can use
       it, it's normally constant-time.
       MSVC uses a different syntax for this operation. */
    for (size_t i = 1; i < (sizeof EXPM_COEFFS) / sizeof(uint64_t); i++) {
#    if defined _MSC_VER
        y = EXPM_COEFFS[i] - __umulh(z, y);
#    else
        unsigned __int128 c = (unsigned __int128)z * (unsigned __int128)y;
        y = EXPM_COEFFS[i] - (uint64_t)(c >> 64);
#    endif
    }
#    if defined _MSC_VER
    y = __umulh(w, y);
#    else
    y = (uint64_t)(((unsigned __int128)w * (unsigned __int128)y) >> 64);
#    endif
#else
    /* On 32-bit x86, no 64x64->128 multiplication, we must use
       four 32x32->64 multiplications. */
    uint32_t z0 = (uint32_t)z, z1 = (uint32_t)(z >> 32);
    uint32_t w0 = (uint32_t)w, w1 = (uint32_t)(w >> 32);

    for (size_t i = 1; i < (sizeof EXPM_COEFFS) / sizeof(uint64_t); i++) {
        uint32_t y0 = (uint32_t)y, y1 = (uint32_t)(y >> 32);
        uint64_t f = (uint64_t)z0 * (uint64_t)y0;
        uint64_t a = (uint64_t)z0 * (uint64_t)y1 + (f >> 32);
        uint64_t b = (uint64_t)z1 * (uint64_t)y0;
        uint64_t c =
            (a >> 32) + (b >> 32) +
            (((uint64_t)(uint32_t)a + (uint64_t)(uint32_t)b) >> 32) +
            (uint64_t)z1 * (uint64_t)y1;
        y = EXPM_COEFFS[i] - c;
    }
    uint32_t y0 = (uint32_t)y, y1 = (uint32_t)(y >> 32);
    uint64_t f = (uint64_t)w0 * (uint64_t)y0;
    uint64_t a = (uint64_t)w0 * (uint64_t)y1 + (f >> 32);
    uint64_t b = (uint64_t)w1 * (uint64_t)y0;
    y = (a >> 32) + (b >> 32) +
        (((uint64_t)(uint32_t)a + (uint64_t)(uint32_t)b) >> 32) +
        (uint64_t)w1 * (uint64_t)y1;
#endif
    return y;
}

/* Sample a bit with probability ccs*exp(-x) (for x >= 0). */
TARGET_SSE2
static inline int ber_exp(sampler_state *ss, __m128d x, __m128d ccs)
{
    static union {
        fpr f[2];
        __m128d x;
    } LOG2_u = {{LOG2, LOG2}}, INV_LOG2_u = {{INV_LOG2, INV_LOG2}};

    int32_t si = _mm_cvttsd_si32(_mm_mul_sd(x, INV_LOG2_u.x));
    __m128d r = _mm_sub_sd(
        x, _mm_mul_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), si), LOG2_u.x));

    uint32_t s = (uint32_t)si;
    s |= (uint32_t)(63 - s) >> 26;

    uint64_t z = fpr_ursh((expm_p63(r, ccs) << 1) - 1, s);

    for (int i = 56; i >= 0; i -= 8) {
        unsigned w = prng_next_u8(&ss->pc);
        unsigned bz = (unsigned)(z >> i) & 0xFF;
        if (w != bz) {
            return w < bz;
        }
    }
    return 0;
}

TARGET_SSE2
static int32_t sampler_next_sse2(sampler_state *ss, __m128d mu,
                                 __m128d isigma)
{
    union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    int32_t s = _mm_cvttsd_si32(mu);
    s -= _mm_comilt_sd(mu, _mm_cvtsi32_sd(_mm_setzero_pd(), s));
    __m128d r = _mm_sub_sd(mu, _mm_cvtsi32_sd(_mm_setzero_pd(), s));

    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_sd(_mm_mul_sd(isigma, isigma), HALF_u.x);

    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_sd(
        isigma, _mm_load_sd((const double *)SIGMA_MIN + ss->logn));

    /* We sample on centre r. */
    for (;;) {
        // printf("1, ");
        int32_t z_bimodal, z_square;
        GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bimodal, &z_square);
        __m128d x =
            _mm_sub_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_bimodal), r);
        x = _mm_mul_sd(_mm_mul_sd(x, x), dss);
        x = _mm_sub_sd(
            x, _mm_mul_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_square),
                          INV_2SQRSIGMA0_u.x));
        if (ber_exp(ss, x, ccs)) {
            // printf("\n");
            return s + z_bimodal;
        }
    }
}

static int32_t sampler_next_sse2_part_v0(sampler_state *ss, __m128d mu,
                                         __m128d isigma)
{
    static union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}}, tt;
    static union {
        int32_t d[4];
        __m128i x;
    } t0;

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    int32_t s = _mm_cvttsd_si32(mu);
    s -= _mm_comilt_sd(mu, _mm_cvtsi32_sd(_mm_setzero_pd(), s));
    __m128d r = _mm_sub_sd(mu, _mm_cvtsi32_sd(_mm_setzero_pd(), s));
    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_sd(_mm_mul_sd(isigma, isigma), HALF_u.x);
    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_sd(
        isigma, _mm_load_sd((const double *)SIGMA_MIN + ss->logn));

    tt.x = _mm_sub_pd(r, ccs);
    tt.x = _mm_add_pd(tt.x, dss);
    int32_t tzbi, tzsq;
    GAUSSIAN0_STORE_get_next(ss->gauss_store, &tzbi, &tzsq);
    return (int32_t)tt.f[0] + tzbi + s;
}

static int32_t sampler_next_sse2_part_v1(sampler_state *ss, __m128d mu,
                                         __m128d isigma)
{
    static union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    int32_t s = _mm_cvttsd_si32(mu);
    s -= _mm_comilt_sd(mu, _mm_cvtsi32_sd(_mm_setzero_pd(), s));
    __m128d r = _mm_sub_sd(mu, _mm_cvtsi32_sd(_mm_setzero_pd(), s));

    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_sd(_mm_mul_sd(isigma, isigma), HALF_u.x);

    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_sd(
        isigma, _mm_load_sd((const double *)SIGMA_MIN + ss->logn));

    /* We sample on centre r. */
    for (;;) {
        // printf("1, ");
        int32_t z_bimodal, z_square;
        GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bimodal, &z_square);
        __m128d x =
            _mm_sub_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_bimodal), r);
        x = _mm_mul_sd(_mm_mul_sd(x, x), dss);
        x = _mm_sub_sd(
            x, _mm_mul_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_square),
                          INV_2SQRSIGMA0_u.x));
        return s + _mm_cvttsd_si32(x);
        // if (ber_exp(ss, x, ccs)) {
        //     // printf("\n");
        //     return s + z_bimodal;
        // }
        break;
    }
}

static int32_t sampler_next_sse2_part_v2(sampler_state *ss, __m128d mu,
                                         __m128d isigma)
{
    static union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    int32_t s = _mm_cvttsd_si32(mu);
    s -= _mm_comilt_sd(mu, _mm_cvtsi32_sd(_mm_setzero_pd(), s));
    __m128d r = _mm_sub_sd(mu, _mm_cvtsi32_sd(_mm_setzero_pd(), s));

    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_sd(_mm_mul_sd(isigma, isigma), HALF_u.x);

    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_sd(
        isigma, _mm_load_sd((const double *)SIGMA_MIN + ss->logn));

    /* We sample on centre r. */
    for (;;) {
        int32_t z_bimodal, z_square;
        GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bimodal, &z_square);
        __m128d x =
            _mm_sub_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_bimodal), r);
        x = _mm_mul_sd(_mm_mul_sd(x, x), dss);
        x = _mm_sub_sd(
            x, _mm_mul_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_square),
                          INV_2SQRSIGMA0_u.x));
        return s + z_bimodal + ber_exp(ss, x, ccs);
        break;
    }
}

/*
 * Input: 0 <= x < log(2) Output: trunc(x*2^63)
 */
static inline void mtwop63_2w(uint64_t *r0, uint64_t *r1, __m128d x)
{
    static const union {
        fpr f[2];
        __m128d x;
    } twop63 = {{
        FPR(4503599627370496, 11),
        FPR(4503599627370496, 11),
    }};
    __m128d t0, t1;

    t0 = _mm_mul_pd(x, twop63.x);
    t1 = _mm_shuffle_pd(t0, t0, 3);
    *r0 = (uint64_t)_mm_cvttsd_si64(t0);
    *r1 = (uint64_t)_mm_cvttsd_si64(t1);
}

#define MUL64HI(a, b) \
    ((uint64_t)(((unsigned __int128)(a) * (unsigned __int128)(b)) >> 64))

static inline void expm_p63_2w(uint64_t *r0, uint64_t *r1, __m128d x01,
                               __m128d ccs01)
{
    /* The polynomial approximation of exp(-x) is from FACCT:
    https://eprint.iacr.org/2018/1234
    Specifically, the values are extracted from the implementation
    referenced by the FACCT paper, available at:
    https://github.com/raykzhao/gaussian  */
    static const uint64_t EXPM_COEFFS[] = {
        0x00000004741183A3, 0x00000036548CFC06, 0x0000024FDCBF140A,
        0x0000171D939DE045, 0x0000D00CF58F6F84, 0x000680681CF796E3,
        0x002D82D8305B0FEA, 0x011111110E066FD0, 0x0555555555070F00,
        0x155555555581FF00, 0x400000000002B400, 0x7FFFFFFFFFFF4800,
        0x8000000000000000};

    uint64_t y0 = EXPM_COEFFS[0];
    uint64_t y1 = EXPM_COEFFS[0];
    uint64_t z0, z1, w0, w1;
    mtwop63_2w(&z0, &z1, x01);
    mtwop63_2w(&w0, &w1, ccs01);
    z0 <<= 1;
    z1 <<= 1;
    w0 <<= 1;
    w1 <<= 1;
    for (size_t i = 1; i < (sizeof EXPM_COEFFS) / sizeof(uint64_t); i++) {
        y0 = EXPM_COEFFS[i] - MUL64HI(z0, y0);
        y1 = EXPM_COEFFS[i] - MUL64HI(z1, y1);
    }
    *r0 = MUL64HI(w0, y0);
    *r1 = MUL64HI(w1, y1);
}

/* Sample a bit with probability ccs*exp(-x) (for x >= 0). */
TARGET_SSE2
static inline void ber_exp_2w(int *r0, int *r1, sampler_state *ss,
                              __m128d x, __m128d ccs)
{
    static const union {
        fpr f[2];
        __m128d x;
    } LOG2_u = {{LOG2, LOG2}}, INV_LOG2_u = {{INV_LOG2, INV_LOG2}};

    static const union {
        uint32_t d[4];
        __m128i x;
    } CONST_63 = {{63, 0, 63, 0}};

    static union {
        uint64_t d[2];
        __m128i x;
    } si;
    uint64_t p63_t0, p63_t1;

    /** _mm_cvttpd_epi32 does not affect the upper 64 bits */
    si.x = _mm_setzero_si128();
    si.x = _mm_cvttpd_epi32(_mm_mul_pd(x, INV_LOG2_u.x));
    __m128d r = _mm_sub_pd(x, _mm_mul_pd(_mm_cvtepi32_pd(si.x), LOG2_u.x));
    /** [0,0,si_1,si_0] -> [0,si_1,0,si_0], 0x98=0b10011000 */
    si.x = _mm_shuffle_epi32(si.x, 0x98);
    __m128i t1 = _mm_sub_epi32(CONST_63.x, si.x);
    t1 = _mm_srli_epi32(t1, 26);
    si.x = _mm_or_si128(si.x, t1);

    expm_p63_2w(&p63_t0, &p63_t1, r, ccs);
    p63_t0 = ((p63_t0 << 1) - 1) >> si.d[0];
    p63_t1 = ((p63_t1 << 1) - 1) >> si.d[1];

    *r0 = 0;
    *r1 = 0;
    for (int i = 56; i >= 0; i -= 8) {
        unsigned w = prng_next_u8(&ss->pc);
        unsigned bz = (unsigned)(p63_t0 >> i) & 0xFF;
        if (w != bz) {
            *r0 = (w < bz);
            break;
        }
    }
    for (int i = 56; i >= 0; i -= 8) {
        unsigned w = prng_next_u8(&ss->pc);
        unsigned bz = (unsigned)(p63_t1 >> i) & 0xFF;
        if (w != bz) {
            *r1 = (w < bz);
            break;
        }
    }
}

TARGET_SSE2
static void sampler_next_sse2_2w(int32_t *s0, int32_t *s1,
                                 sampler_state *ss, __m128d mu,
                                 __m128d isigma)
{
    const union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};

    union {
        int32_t d[4];
        __m128i x;
    } si, z_bi, z_sq;

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    __m128d trunc = _mm_cvtepi32_pd(_mm_cvttpd_epi32(mu));
    __m128d mask = _mm_cmplt_pd(mu, trunc);
    mask = _mm_and_pd(mask, _mm_set1_pd(1.0));
    __m128d floor = _mm_sub_pd(trunc, mask);
    __m128d r = _mm_sub_pd(mu, floor);
    si.x = _mm_cvttpd_epi32(floor);
    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_pd(_mm_mul_pd(isigma, isigma), HALF_u.x);
    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_pd(isigma, SIGMA_MINx2[ss->logn].x);

    int r0, r1;
    int32_t *s_no;
    int32_t si_t;
    for (;;) {
        GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[0], &z_sq.d[0]);
        GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[1], &z_sq.d[1]);
        __m128d x = _mm_sub_pd(_mm_cvtepi32_pd(z_bi.x), r);
        x = _mm_mul_pd(_mm_mul_pd(x, x), dss);
        x = _mm_sub_pd(
            x, _mm_mul_pd(_mm_cvtepi32_pd(z_sq.x), INV_2SQRSIGMA0_u.x));
        ber_exp_2w(&r0, &r1, ss, x, ccs);
        if (r0 == 1 || r1 == 1) {
            *s0 = si.d[0] + z_bi.d[0];
            *s1 = si.d[1] + z_bi.d[1];
            if (r0 == 1 && r1 == 1) {
                return;
            } else if (r0 == 1) {
                s_no = s1;
                si_t = si.d[1];
                r = _mm_shuffle_pd(r, r, 3);
            } else {
                s_no = s0;
                si_t = si.d[0];
                r = _mm_shuffle_pd(r, r, 0);
            }
            break;
        }
    }
    for (;;) {
        int32_t z_bimodal, z_square;
        GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bimodal, &z_square);
        __m128d x =
            _mm_sub_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_bimodal), r);
        x = _mm_mul_sd(_mm_mul_sd(x, x), dss);
        x = _mm_sub_sd(
            x, _mm_mul_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_square),
                          INV_2SQRSIGMA0_u.x));
        if (ber_exp(ss, x, ccs)) {
            *s_no = si_t + z_bimodal;
            return;
        }
    }
}

static void sampler_next_sse2_2w_part_v0(int32_t *s0, int32_t *s1,
                                         sampler_state *ss, __m128d mu,
                                         __m128d isigma)
{
    static const union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};

    static union {
        fpr f[2];
        __m128d x;
    } tt;

    static union {
        int32_t d[4];
        __m128i x;
    } t0, si, z_bi, z_sq;

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    __m128i mu_i32 = _mm_cvttpd_epi32(mu);
    __m128d trunc = _mm_cvtepi32_pd(mu_i32);
    __m128d mask = _mm_cmplt_pd(mu, trunc);
    mask = _mm_and_pd(mask, _mm_set1_pd(1.0));
    __m128d floor = _mm_sub_pd(trunc, mask);
    __m128d r = _mm_sub_pd(mu, floor);
    si.x = _mm_cvttpd_epi32(floor);
    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_pd(_mm_mul_pd(isigma, isigma), HALF_u.x);
    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_pd(isigma, SIGMA_MINx2[ss->logn].x);

    tt.x = _mm_sub_pd(r, ccs);
    tt.x = _mm_add_pd(tt.x, dss);
    int32_t tzbi0, tzsq0, tzbi1, tzsq1;
    GAUSSIAN0_STORE_get_next(ss->gauss_store, &tzbi0, &tzsq0);
    GAUSSIAN0_STORE_get_next(ss->gauss_store, &tzbi1, &tzsq1);
    *s0 = (int32_t)tt.f[0] + tzbi0 + si.d[0];
    *s1 = (int32_t)tt.f[1] + tzbi1 + si.d[1];
}

static void sampler_next_sse2_2w_part_v1(int32_t *s0, int32_t *s1,
                                         sampler_state *ss, __m128d mu,
                                         __m128d isigma)
{
    static const union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};

    union {
        int32_t d[4];
        __m128i x;
    } si, z_bi, z_sq, t0;

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    __m128d trunc = _mm_cvtepi32_pd(_mm_cvttpd_epi32(mu));
    GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[0], &z_sq.d[0]);
    __m128d mask = _mm_cmplt_pd(mu, trunc);
    GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[1], &z_sq.d[1]);
    mask = _mm_and_pd(mask, _mm_set1_pd(1.0));
    __m128d floor = _mm_sub_pd(trunc, mask);
    __m128i zbi = _mm_load_si128(&z_bi.x);
    __m128i zsq = _mm_load_si128(&z_sq.x);
    // __m128i zbi = _mm_setzero_si128();
    // __m128i zsq = _mm_setzero_si128();
    __m128d r = _mm_sub_pd(mu, floor);
    si.x = _mm_cvttpd_epi32(floor);
    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_pd(_mm_mul_pd(isigma, isigma), HALF_u.x);
    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_pd(isigma, SIGMA_MINx2[ss->logn].x);
    __m128d x = _mm_sub_pd(_mm_cvtepi32_pd(zbi), r);
    x = _mm_mul_pd(_mm_mul_pd(x, x), dss);
    x = _mm_sub_pd(x,
                   _mm_mul_pd(_mm_cvtepi32_pd(zsq), INV_2SQRSIGMA0_u.x));
    // t0.x = _mm_cvtpd_epi32(x);
    // *s0 = si.d[0] + t0.d[0];
    // *s1 = si.d[1] + t0.d[1];
    *s0 = si.d[0] + _mm_cvttsd_si32(x);
    *s1 = si.d[1] + _mm_cvttsd_si32(x);
    // return si.d[0] + _mm_cvttsd_si32(x);
    // *s0 = _mm_cvttsd_si32(x);
    // *s1 = _mm_cvttsd_si32(x);
    // return _mm_cvttsd_si32(x);

    // ber_exp_2w(&r0, &r1, ss, x, ccs);
    // if (r0 == 1 || r1 == 1) {
    //     *s0 = si.d[0] + z_bi.d[0];
    //     *s1 = si.d[1] + z_bi.d[1];
    //     if (r0 == 1 && r1 == 1) {
    //         return;
    //     } else if (r0 == 1) {
    //         s_no = s1;
    //         si_t = si.d[1];
    //         r = _mm_shuffle_pd(r, r, 3);
    //     } else {
    //         s_no = s0;
    //         si_t = si.d[0];
    //         r = _mm_shuffle_pd(r, r, 0);
    //     }
    //     break;
    // }
}

static void sampler_next_sse2_2w_part_v1_inorder(int32_t *s0, int32_t *s1,
                                                 sampler_state *ss,
                                                 __m128d mu,
                                                 __m128d isigma)
{
    static const union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};

    union {
        int32_t d[4];
        __m128i x;
    } si, z_bi, z_sq, t0;

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    __m128d trunc = _mm_cvtepi32_pd(_mm_cvttpd_epi32(mu));
    __m128d mask = _mm_cmplt_pd(mu, trunc);
    mask = _mm_and_pd(mask, _mm_set1_pd(1.0));
    __m128d floor = _mm_sub_pd(trunc, mask);
    __m128d r = _mm_sub_pd(mu, floor);
    si.x = _mm_cvttpd_epi32(floor);
    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_pd(_mm_mul_pd(isigma, isigma), HALF_u.x);
    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_pd(isigma, SIGMA_MINx2[ss->logn].x);

    GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[0], &z_sq.d[0]);
    GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[1], &z_sq.d[1]);
    __m128d x = _mm_sub_pd(_mm_cvtepi32_pd(z_bi.x), r);
    x = _mm_mul_pd(_mm_mul_pd(x, x), dss);
    x = _mm_sub_pd(
        x, _mm_mul_pd(_mm_cvtepi32_pd(z_sq.x), INV_2SQRSIGMA0_u.x));
    *s0 = si.d[0] + _mm_cvttsd_si32(x);
    *s1 = si.d[1] + _mm_cvttsd_si32(x);
    // ber_exp_2w(&r0, &r1, ss, x, ccs);
    // if (r0 == 1 || r1 == 1) {
    //     *s0 = si.d[0] + z_bi.d[0];
    //     *s1 = si.d[1] + z_bi.d[1];
    //     if (r0 == 1 && r1 == 1) {
    //         return;
    //     } else if (r0 == 1) {
    //         s_no = s1;
    //         si_t = si.d[1];
    //         r = _mm_shuffle_pd(r, r, 3);
    //     } else {
    //         s_no = s0;
    //         si_t = si.d[0];
    //         r = _mm_shuffle_pd(r, r, 0);
    //     }
    //     break;
    // }
}

static void sampler_next_sse2_2w_part_v2(int32_t *s0, int32_t *s1,
                                    sampler_state *ss, __m128d mu,
                                    __m128d isigma)
{
    const union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};

    union {
        int32_t d[4];
        __m128i x;
    } si, z_bi, z_sq;

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    __m128d trunc = _mm_cvtepi32_pd(_mm_cvttpd_epi32(mu));
    __m128d mask = _mm_cmplt_pd(mu, trunc);
    mask = _mm_and_pd(mask, _mm_set1_pd(1.0));
    __m128d floor = _mm_sub_pd(trunc, mask);
    __m128d r = _mm_sub_pd(mu, floor);
    si.x = _mm_cvttpd_epi32(floor);
    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_pd(_mm_mul_pd(isigma, isigma), HALF_u.x);
    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_pd(isigma, SIGMA_MINx2[ss->logn].x);

    int r0, r1;
    int32_t *s_no;
    int32_t si_t;
    for (;;) {
        GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[0], &z_sq.d[0]);
        GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[1], &z_sq.d[1]);
        __m128d x = _mm_sub_pd(_mm_cvtepi32_pd(z_bi.x), r);
        x = _mm_mul_pd(_mm_mul_pd(x, x), dss);
        x = _mm_sub_pd(
            x, _mm_mul_pd(_mm_cvtepi32_pd(z_sq.x), INV_2SQRSIGMA0_u.x));
        ber_exp_2w(&r0, &r1, ss, x, ccs);
        *s0 = si.d[0] + z_bi.d[0] + r0;
        *s1 = si.d[1] + z_bi.d[1] + r1;
        break;
    }
}

#ifndef WARMUP_N
#    define WARMUP_N 500
#endif

#ifndef TESTS_N
#    define TESTS_N 2000
#endif

int main()
{
    sampler_state ss;
    uint8_t seed[32] = {0};
    volatile __m128d mu, isigma;
    int32_t s0 = 0, s1 = 0, s3 = 0, s0acc = 0, s1acc = 0;
    volatile int32_t sum = 0;

    mu = _mm_set1_pd(3.0);
    isigma = _mm_set1_pd(0.5);

    sampler_init(&ss, 9, seed, 32);
    init_perf_counters();

    PERF(s3 += sampler_next_sse2_part_v0(&ss, mu, isigma),
         sampler_next_sse2_part_v0, sampler_init(&ss, 9, seed, 32),
         (8 * 16), (8 * 16));
    PERF_N(
        do {
            sampler_next_sse2_2w_part_v0(&s0, &s1, &ss, mu, isigma);
            s0acc += s0;
            s1acc += s1;
        } while (0),
        sampler_next_sse2_2w_part_v0, sampler_init(&ss, 9, seed, 32),
        (8 * 8), (8 * 8), 2);

    PERF(s3 = sampler_next_sse2_part_v1(&ss, mu, isigma),
         sampler_next_sse2_part_v1, sampler_init(&ss, 9, seed, 32),
         (8 * 16), (8 * 16));
    PERF_N(
        do {
            sampler_next_sse2_2w_part_v1(&s0, &s1, &ss, mu, isigma);
        } while (0),
        sampler_next_sse2_2w_part_v1, sampler_init(&ss, 9, seed, 32),
        (8 * 8), (8 * 8), 2);
    PERF_N(
        do {
            sampler_next_sse2_2w_part_v1_inorder(&s0, &s1, &ss, mu,
                                                 isigma);
        } while (0),
        sampler_next_sse2_2w_part_v1_inorder,
        sampler_init(&ss, 9, seed, 32), (8 * 8), (8 * 8), 2);

    PERF(s3 = sampler_next_sse2_part_v2(&ss, mu, isigma),
         sampler_next_sse2_part_v2, sampler_init(&ss, 9, seed, 32),
         (8 * 16), (8 * 16));
    PERF_N(
        do {
            sampler_next_sse2_2w_part_v2(&s0, &s1, &ss, mu, isigma);
        } while (0),
        sampler_next_sse2_2w_part_v2, sampler_init(&ss, 9, seed, 32),
        (8 * 8), (8 * 8), 2);

    PERF(s3 += sampler_next_sse2(&ss, mu, isigma), sampler_next_sse2,
         sampler_init(&ss, 9, seed, 32), (8 * 16), (8 * 16));
    PERF_N(
        do {
            sampler_next_sse2_2w(&s0, &s1, &ss, mu, isigma);
            s0acc += s0;
            s1acc += s1;
        } while (0),
        sampler_next_sse2_2w, sampler_init(&ss, 9, seed, 32), (8 * 8),
        (8 * 8), 2);

    sum = sum + s3 + s0acc + s1acc;
    return 0;
}