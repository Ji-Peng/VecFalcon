#ifndef INTRIN_H
#define INTRIN_H

#include <immintrin.h>

#define VZERO _mm256_setzero_si256()
#define VSET1(X) _mm256_set1_epi32(X)

#define VADD(X, Y) _mm256_add_epi32(X, Y)
#define VSUB(X, Y) _mm256_sub_epi32(X, Y)
#define VSHR(X, Y) _mm256_srli_epi32(X, Y)

#define VLOAD256(X) _mm256_load_epi32(X)
#define VSTORE256(X, Y) _mm256_store_si256(X, Y)

#endif