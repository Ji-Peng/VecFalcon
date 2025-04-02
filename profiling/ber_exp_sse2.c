#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "cpucycles.h"

typedef uint64_t fpr;

#define FPR(i, e) \
    ((i) < 0 ? (((uint64_t)1 << 63) ^ FPR_(-(i), e)) : FPR_(i, e))
#define FPR_(i, e)                                  \
    (((uint64_t)(int64_t)(i) &                      \
      (((uint64_t)1 << 63) | 0x000FFFFFFFFFFFFF)) + \
     ((uint64_t)(((uint32_t)(e) + 1075) & 0x7FF) << 52))

#define MUL64HI(a, b) \
    ((uint64_t)(((unsigned __int128)(a) * (unsigned __int128)(b)) >> 64))

/*
 * Input: 0 <= x < log(2) Output: trunc(x*2^63)
 */
static inline int64_t mtwop63(__m128d x)
{
    static const union {
        fpr f[2];
        __m128d x;
    } twop63 = {{
        FPR(4503599627370496, 11),
        FPR(4503599627370496, 11),
    }};

    return _mm_cvttsd_si64(_mm_mul_sd(x, twop63.x));
}

// _mm_cvttpd_epi64 requires CPUID Flags: AVX512DQ + AVX512VL

/*
 * Compute ccs*exp(-x)*2^63, rounded to an integer. This function assumes
 * that 0 <= x < log(2), and 0 <= ccs <= 1. It returns a value in [0,2^63].
 *
 * Obtained from C-FN-DSA project (commit id:
 * 1cdc9c5bdd5b5894475febd7e23abbcb5056197b).
 * We only consider 64-bit Linux platforms.
 */
static uint64_t expm_p63(__m128d x, __m128d ccs)
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
    /* On 64-bit x86, we have 64x64->128 multiplication, then we can use
       it, it's normally constant-time.
       MSVC uses a different syntax for this operation. */
    for (size_t i = 1; i < (sizeof EXPM_COEFFS) / sizeof(uint64_t); i++) {
        unsigned __int128 c = (unsigned __int128)z * (unsigned __int128)y;
        y = EXPM_COEFFS[i] - (uint64_t)(c >> 64);
    }
    y = (uint64_t)(((unsigned __int128)w * (unsigned __int128)y) >> 64);
    return y;
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

static void expm_p63_2w(uint64_t *r0, uint64_t *r1, __m128d x01,
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
    /* On 64-bit x86, we have 64x64->128 multiplication, then we can use
    it, it's normally constant-time.
    MSVC uses a different syntax for this operation. */
    for (size_t i = 1; i < (sizeof EXPM_COEFFS) / sizeof(uint64_t); i++) {
        y0 = EXPM_COEFFS[i] - MUL64HI(z0, y0);
        y1 = EXPM_COEFFS[i] - MUL64HI(z1, y1);
    }
    *r0 = MUL64HI(w0, y0);
    *r1 = MUL64HI(w1, y1);
}

static void expm_p63_4w(uint64_t *r0, uint64_t *r1, uint64_t *r2,
                        uint64_t *r3, __m128d x01, __m128d ccs01,
                        __m128d x23, __m128d ccs23)
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
    uint64_t y2 = EXPM_COEFFS[0];
    uint64_t y3 = EXPM_COEFFS[0];
    uint64_t z0, z1, z2, z3, w0, w1, w2, w3;
    mtwop63_2w(&z0, &z1, x01);
    mtwop63_2w(&z2, &z3, x23);
    mtwop63_2w(&w0, &w1, ccs01);
    mtwop63_2w(&w2, &w3, ccs23);

    z0 <<= 1;
    z1 <<= 1;
    z2 <<= 1;
    z3 <<= 1;
    w0 <<= 1;
    w1 <<= 1;
    w2 <<= 1;
    w3 <<= 1;
    for (size_t i = 1; i < (sizeof EXPM_COEFFS) / sizeof(uint64_t); i++) {
        y0 = EXPM_COEFFS[i] - MUL64HI(z0, y0);
        y1 = EXPM_COEFFS[i] - MUL64HI(z1, y1);
        y2 = EXPM_COEFFS[i] - MUL64HI(z2, y2);
        y3 = EXPM_COEFFS[i] - MUL64HI(z3, y3);
    }
    *r0 = MUL64HI(w0, y0);
    *r1 = MUL64HI(w1, y1);
    *r2 = MUL64HI(w2, y2);
    *r3 = MUL64HI(w3, y3);
}
/* log(2) */
#define LOG2 FPR(6243314768165359, -53)
/* 1/log(2) */
#define INV_LOG2 FPR(6497320848556798, -52)
#define INV_2SQRSIGMA0 FPR(5435486223186882, -55)
typedef uint64_t fpr;
#define FPR(i, e) \
    ((i) < 0 ? (((uint64_t)1 << 63) ^ FPR_(-(i), e)) : FPR_(i, e))
#define FPR_(i, e)                                  \
    (((uint64_t)(int64_t)(i) &                      \
      (((uint64_t)1 << 63) | 0x000FFFFFFFFFFFFF)) + \
     ((uint64_t)(((uint32_t)(e) + 1075) & 0x7FF) << 52))

#define FPR_ZERO FPR(0, -1075)
#define FPR_NZERO (FPR_ZERO ^ ((uint64_t)1 << 63))
#define FPR_ONE FPR(4503599627370496, -52)
typedef union {
    fpr f;
#if FNDSA_NEON
    float64x1_t v;
#endif
#if FNDSA_RV64D
    f64 v;
#endif
} fpr_u;
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

static int ber_exp(__m128d x, __m128d ccs)
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

    uint64_t z = ((expm_p63(r, ccs) << 1) - 1) >> s;

    for (int i = 56; i >= 0; i -= 8) {
        unsigned w = 0x66;
        unsigned bz = (unsigned)(z >> i) & 0xFF;
        if (w != bz) {
            return w < bz;
        }
    }
    return 0;
}

/* Sample a bit with probability ccs*exp(-x) (for x >= 0). */
static void ber_exp_2w(int *r0, int *r1, __m128d x, __m128d ccs)
{
    static const union {
        fpr f[2];
        __m128d x;
    } LOG2_u = {{LOG2, LOG2}}, INV_LOG2_u = {{INV_LOG2, INV_LOG2}};

    static const union {
        uint32_t d[4];
        __m128i x;
    } CONST_63 = {{63, 0, 63, 0}}, CONST_1 = {{1, 0, 1, 0}};

    static union {
        uint64_t d[2];
        __m128i x;
    } t0;

    /** _mm_cvttpd_epi32 does not affect the upper 64 bits */
    __m128i si = _mm_setzero_si128();
    si = _mm_cvttpd_epi32(_mm_mul_pd(x, INV_LOG2_u.x));
    __m128d r = _mm_sub_pd(x, _mm_mul_pd(_mm_cvtepi32_pd(si), LOG2_u.x));
    /** [0,0,si_1,si_0] -> [0,si_1,0,si_0], 0x98=0b10011000 */
    si = _mm_shuffle_epi32(si, 0x98);
    __m128i t1 = _mm_sub_epi32(CONST_63.x, si);
    t1 = _mm_srli_epi32(t1, 26);
    si = _mm_or_si128(si, t1);

    expm_p63_2w(&t0.d[0], &t0.d[1], r, ccs);
    t0.x = _mm_slli_epi64(t0.x, 1);
    t0.x = _mm_sub_epi64(t0.x, CONST_1.x);
    t0.x = _mm_srl_epi64(t0.x, si);

    *r0 = 0;
    *r1 = 0;
    for (int i = 56; i >= 0; i -= 8) {
        unsigned w = 0x66;
        unsigned bz = (unsigned)(t0.d[0] >> i) & 0xFF;
        if (w != bz) {
            *r0 = (w < bz);
            break;
        }
    }
    for (int i = 56; i >= 0; i -= 8) {
        unsigned w = 0x66;
        unsigned bz = (unsigned)(t0.d[1] >> i) & 0xFF;
        if (w != bz) {
            *r1 = (w < bz);
            return;
        }
    }
}

// return z for debug
static void ber_exp_z(uint64_t *z, __m128d x, __m128d ccs)
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
    uint64_t p63_r = expm_p63(r, ccs);
    *z = ((p63_r << 1) - 1) >> s;
}

// return z for debug
static void ber_exp_2w_z(uint64_t *z0, uint64_t *z1, __m128d x,
                         __m128d ccs)
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
    *z0 = ((p63_t0 << 1) - 1) >> si.d[0];
    *z1 = ((p63_t1 << 1) - 1) >> si.d[1];
}

#define WARMUP_N 1000
#define TESTS_N 100000

void speed_expm_p63()
{
    // volatile: avoid compiler optimization
    volatile __m128d x01, ccs01, x1, ccs1, x23, ccs23;
    uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    volatile uint64_t r = 0;

    // 0 <= x < log(2), 0 <= ccs <= 1
    x01 = _mm_set1_pd(0.4931471805599453);
    ccs01 = _mm_set1_pd(0.5931471805599453);
    x1 = _mm_set1_pd(0.4931471805599453);
    ccs1 = _mm_set1_pd(0.5931471805599453);
    x23 = _mm_set1_pd(0.4931471805599453);
    ccs23 = _mm_set1_pd(0.5931471805599453);

    init_perf_counters();

    PERF(r0 += expm_p63(x01, ccs01), expm_p63, , WARMUP_N, TESTS_N);
    PERF_N(expm_p63_2w(&r0, &r1, x01, ccs01), expm_p63_2w, , WARMUP_N,
           TESTS_N, 2);
    r = r + r0 + r1;
    PERF_N(expm_p63_4w(&r0, &r1, &r2, &r3, x01, ccs01, x23, ccs23),
           expm_p63_4w, , WARMUP_N, TESTS_N, 4);
    r = r + r0 + r1 + r2 + r3;
}

void speed_ber_exp_z()
{
    __m128d x, ccs;
    uint64_t z0, z1;
    volatile uint64_t z;

    x = _mm_set1_pd(0.4931471805599453);
    ccs = _mm_set1_pd(0.5931471805599453);

    init_perf_counters();
    PERF(ber_exp_z(&z0, x, ccs), ber_exp_z, , WARMUP_N, TESTS_N);
    PERF_N(ber_exp_2w_z(&z0, &z1, x, ccs), ber_exp_2w_z, , WARMUP_N,
           TESTS_N, 2);

    z = z0 + z1;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void debug_sampler_next_sse2()
{
    srand(time(NULL));
    __m128d mu_ref;
    __m128d isigma_ref;
    __m128d mu;
    __m128d isigma;
    double random_num;
    for (;;) {
        static union {
            fpr f[2];
            __m128d x;
        } HALF_u = {{FPR(4503599627370496, -53),
                     FPR(4503599627370496, -53)}},
          INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};
        static union {
            int32_t d[4];
            __m128i x;
        } t0, si, z_bi, z_sq;

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
        random_num = ((double)rand() / (RAND_MAX / (100.0 + 1.0))) - 50.0;
        mu_ref = mu = _mm_set1_pd(random_num);
        random_num = (double)rand() / RAND_MAX;
        isigma_ref = isigma = _mm_set1_pd(random_num);

        /** ref. */
        int32_t s_ref = _mm_cvttsd_si32(mu_ref);
        s_ref -=
            _mm_comilt_sd(mu_ref, _mm_cvtsi32_sd(_mm_setzero_pd(), s_ref));
        /** 2w. */
        si.x = _mm_cvttpd_epi32(mu);
        __m128d sd = _mm_cvtepi32_pd(si.x);
        t0.d[0] = _mm_comilt_sd(mu, sd);
        t0.d[1] = _mm_comilt_sd(_mm_shuffle_pd(mu, mu, 3),
                                _mm_shuffle_pd(sd, sd, 3));
        si.x = _mm_sub_epi32(si.x, t0.x);
        /** ref. */
        __m128d r_ref =
            _mm_sub_sd(mu_ref, _mm_cvtsi32_sd(_mm_setzero_pd(), s_ref));
        __m128d dss_ref =
            _mm_mul_sd(_mm_mul_sd(isigma_ref, isigma_ref), HALF_u.x);
        __m128d ccs_ref = _mm_mul_sd(
            isigma_ref, _mm_load_sd((const double *)SIGMA_MIN + 9));
        /** 2w. */
        __m128d r = _mm_sub_pd(mu, _mm_cvtepi32_pd(si.x));
        __m128d dss = _mm_mul_pd(_mm_mul_pd(isigma, isigma), HALF_u.x);
        __m128d ccs = _mm_mul_pd(isigma, SIGMA_MINx2[9].x);

        /** ref. */
        __m128d x_ref;
        for (;;) {
            int32_t zbi_ref = -3, zsq_ref = 9;
            x_ref = _mm_sub_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), zbi_ref),
                               r_ref);
            x_ref = _mm_mul_sd(_mm_mul_sd(x_ref, x_ref), dss_ref);
            x_ref = _mm_sub_sd(
                x_ref,
                _mm_mul_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), zsq_ref),
                           INV_2SQRSIGMA0_u.x));
            // if (ber_exp(x_ref, ccs_ref)) {
            //     return s_ref + zbi_ref;
            // }
            break;
        }
        /** 2w. */
        z_bi.d[0] = z_bi.d[1] = -3;
        z_sq.d[0] = z_sq.d[1] = 9;
        __m128d x = _mm_sub_pd(_mm_cvtepi32_pd(z_bi.x), r);
        x = _mm_mul_pd(_mm_mul_pd(x, x), dss);
        x = _mm_sub_pd(
            x, _mm_mul_pd(_mm_cvtepi32_pd(z_sq.x), INV_2SQRSIGMA0_u.x));
        // ber_exp_2w(&r0, &r1, ss, x, ccs);

        break;
    }
}

void test_ber_exp()
{
    double x0, x1, ccs0;
    __m128d x, ccs;
    uint64_t z0, z1, z2, z3;

    for (int i = 0; i < 10000; i++) {
        x0 = (double)rand() / RAND_MAX * 4.0;
        x1 = (double)rand() / RAND_MAX * 4.0;
        ccs0 = (double)rand() / RAND_MAX;
        x = _mm_set_pd(x1, x0);
        ccs = _mm_set1_pd(ccs0);

        ber_exp_z(&z0, x, ccs);
        ber_exp_z(&z1, _mm_shuffle_pd(x, x, 3),
                  _mm_shuffle_pd(ccs, ccs, 3));
        ber_exp_2w_z(&z2, &z3, x, ccs);
        if (z0 != z2 || z1 != z3) {
            printf("error: %ld %ld %ld %ld\n", z0, z1, z2, z3);
            printf("x0, x1, ccs0: %lf %lf %lf\n", x0, x1, ccs0);
            return;
        }
    }
}

void test_expm_p63()
{
    double x0, x1, ccs0;
    uint64_t r0, r1, r2, r3;
    __m128d x, ccs;

    srand(time(NULL));
    for (int i = 0; i < 1000000; i++) {
        x0 = (double)rand() / RAND_MAX * log(2);
        x1 = (double)rand() / RAND_MAX * log(2);
        ccs0 = (double)rand() / RAND_MAX;
        x = _mm_set_pd(x1, x0);
        ccs = _mm_set1_pd(ccs0);
        r0 = expm_p63(x, ccs);
        r1 =
            expm_p63(_mm_shuffle_pd(x, x, 3), _mm_shuffle_pd(ccs, ccs, 3));
        expm_p63_2w(&r2, &r3, x, ccs);
        if (r0 != r2 || r1 != r3) {
            printf("error: %ld %ld %ld %ld\n", r0, r1, r2, r3);
            return;
        }
    }
}

static int32_t sampler_next_sse2_ref(__m128d mu, __m128d isigma)
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
    __m128d ccs =
        _mm_mul_sd(isigma, _mm_load_sd((const double *)SIGMA_MIN + 9));
    /* We sample on centre r. */
    for (;;) {
        int32_t z0 = 3;
        int32_t b = 0;
        int32_t z = b + ((b << 1) - 1) * z0;
        __m128d x = _mm_sub_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z), r);
        x = _mm_mul_sd(_mm_mul_sd(x, x), dss);
        x = _mm_sub_sd(
            x, _mm_mul_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z0 * z0),
                          INV_2SQRSIGMA0_u.x));
        ber_exp(x, ccs);
        printf("sample value: %d\n", s + z);
        break;
    }
}

static int32_t sampler_next_sse2_store(__m128d mu, __m128d isigma)
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
    __m128d ccs =
        _mm_mul_sd(isigma, _mm_load_sd((const double *)SIGMA_MIN + 9));
    /* We sample on centre r. */
    for (;;) {
        int32_t z_bimodal = -3, z_square = 9;

        __m128d x =
            _mm_sub_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_bimodal), r);
        x = _mm_mul_sd(_mm_mul_sd(x, x), dss);
        x = _mm_sub_sd(
            x, _mm_mul_sd(_mm_cvtsi32_sd(_mm_setzero_pd(), z_square),
                          INV_2SQRSIGMA0_u.x));
        ber_exp(x, ccs);
        printf("sample value: %d\n", s + z_bimodal);
        break;
    }
}

static void sampler_next_sse2_2w(int32_t *s0, int32_t *s1, __m128d mu,
                                 __m128d isigma)
{
    static union {
        fpr f[2];
        __m128d x;
    } HALF_u = {{FPR(4503599627370496, -53), FPR(4503599627370496, -53)}},
      INV_2SQRSIGMA0_u = {{INV_2SQRSIGMA0, INV_2SQRSIGMA0}};

    static union {
        int32_t d[4];
        __m128i x;
    } t0, si, z_bi, z_sq;

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

    /* Split center mu into s + r, for an integer s, and 0 <= r < 1. */
    si.x = _mm_cvttpd_epi32(mu);
    __m128d sd = _mm_cvtepi32_pd(si.x);
    // compare low double
    t0.d[0] = _mm_comilt_sd(mu, sd);
    // compare high double
    t0.d[1] = _mm_comilt_sd(_mm_shuffle_pd(mu, mu, 3),
                            _mm_shuffle_pd(sd, sd, 3));
    si.x = _mm_sub_epi32(si.x, t0.x);
    __m128d r = _mm_sub_pd(mu, _mm_cvtepi32_pd(si.x));
    /* dss = 1/(2*sigma^2) = 0.5*(isigma^2)  */
    __m128d dss = _mm_mul_pd(_mm_mul_pd(isigma, isigma), HALF_u.x);
    /* css = sigma_min / sigma = sigma_min * isigma  */
    __m128d ccs = _mm_mul_pd(isigma, SIGMA_MINx2[9].x);

    int r0, r1;
    int32_t *s_no;
    int32_t si_t;
sampler_next_sse2_2w_start:
    /* We sample on centre r. */
    // GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[0], &z_sq.d[0]);
    // GAUSSIAN0_STORE_get_next(ss->gauss_store, &z_bi.d[1], &z_sq.d[1]);
    z_bi.d[0] = z_bi.d[1] = -3;
    z_sq.d[0] = z_sq.d[1] = 9;
    __m128d x = _mm_sub_pd(_mm_cvtepi32_pd(z_bi.x), r);
    x = _mm_mul_pd(_mm_mul_pd(x, x), dss);
    x = _mm_sub_pd(
        x, _mm_mul_pd(_mm_cvtepi32_pd(z_sq.x), INV_2SQRSIGMA0_u.x));
    ber_exp_2w(&r0, &r1, x, ccs);
    printf("sample value: %d %d\n", si.d[0] + z_bi.d[0],
           si.d[1] + z_bi.d[1]);
}

void test_corr_sampler_next_sse2()
{
    srand(time(NULL));
    double random_num;
    __m128d mu, isigma;
    int r0, r1;

    for (int i = 0; i < 1; i++) {
        random_num = ((double)rand() / RAND_MAX) * 100.0 - 50.0;
        mu = _mm_set1_pd(-1.3);
        random_num = (double)rand() / RAND_MAX;
        isigma = _mm_set1_pd(random_num);
        sampler_next_sse2_ref(mu, isigma);
        sampler_next_sse2_store(mu, isigma);
        sampler_next_sse2_2w(&r0, &r1, mu, isigma);
    }
    printf("\n");
}

int main()
{
    // speed_expm_p63();
    // test_sampler_next_sse2();
    // test_expm_p63();
    test_corr_sampler_next_sse2();
    // test_ber_exp();
    // speed_ber_exp_z();
    return 0;
}