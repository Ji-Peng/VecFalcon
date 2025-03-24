#include <immintrin.h>
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

int main()
{
    speed_expm_p63();
    return 0;
}