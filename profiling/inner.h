#ifndef FNDSA_INNER_H__
#define FNDSA_INNER_H__

/* ==================================================================== */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if FNDSA_AVX2 || FNDSA_SSE2
#    include <immintrin.h>
#    if defined __GNUC__ || defined __clang__
#        include <x86intrin.h>
#    endif
#endif

#if FNDSA_AVX2
#    if defined __GNUC__ || defined __clang__
#        define TARGET_AVX2 __attribute__((target("avx2,lzcnt")))
#    else
#        define TARGET_AVX2
#    endif
#else
#    define TARGET_AVX2
#endif

#ifndef FNDSA_LITTLE_ENDIAN
#    if defined __LITTLE_ENDIAN__ ||                                  \
        (defined __BYTE_ORDER__ && defined __ORDER_LITTLE_ENDIAN__ && \
         __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) ||                \
        defined _M_IX86 || defined _M_X64 || defined _M_ARM64
#        define FNDSA_LITTLE_ENDIAN 1
#    else
#        define FNDSA_LITTLE_ENDIAN 0
#    endif
#endif

/* Some architectures tolerate well unaligned accesses to 64-bit words. */
#ifndef FNDSA_UNALIGNED_64
#    if defined __x86_64__ || defined _M_X64 || defined __i386__ || \
        defined _M_IX86 || defined __aarch64__ || defined _M_ARM64
#        define FNDSA_UNALIGNED_64 1
#    else
#        define FNDSA_UNALIGNED_64 0
#    endif
#endif

/* Some architectures tolerate well unaligned accesses to 16-bit words. */
#ifndef FNDSA_UNALIGNED_16
#    if defined __x86_64__ || defined _M_X64 || defined __i386__ ||     \
        defined _M_IX86 || FNDSA_ASM_CORTEXM4 || defined __aarch64__ || \
        defined _M_ARM64
#        define FNDSA_UNALIGNED_16 1
#    else
#        define FNDSA_UNALIGNED_16 0
#    endif
#endif

typedef struct {
    uint64_t A[25];
    unsigned dptr, rate;
} shake_context;

/* Initialize context, size = 128 or 256 (for SHAKE128 or SHAKE256). */
void shake_init(shake_context *sc, unsigned size);
/* Inject some bytes in context. */
void shake_inject(shake_context *sc, const void *in, size_t len);
/* Flip context from input to output mode. */
void shake_flip(shake_context *sc);
/* Extract some bytes from context. If out is NULL, then len bytes are
   still virtually extracted, but discarded.
   In systems with little-endian encoding, the discarded bytes
   can still be obtained from the context; this is used for saving some
   RAM (especially stack space) on embedded systems. */
void shake_extract(shake_context *sc, void *out, size_t len);

/* Get the next byte from a SHAKE context. */
static inline uint8_t shake_next_u8(shake_context *sc)
{
    if (sc->dptr == sc->rate) {
        uint8_t x;
        shake_extract(sc, &x, 1);
        return x;
    }
#if FNDSA_LITTLE_ENDIAN
    uint8_t *d = (uint8_t *)(void *)sc;
    return d[sc->dptr++];
#else
    uint8_t x = (uint8_t)(sc->A[sc->dptr >> 3] >> ((sc->dptr & 7) << 3));
    sc->dptr++;
    return x;
#endif
}

/* Get the next 16-bit word from SHAKE. */
static inline unsigned shake_next_u16(shake_context *sc)
{
    if (sc->dptr + 1 >= sc->rate) {
        uint8_t x[2];
        shake_extract(sc, x, 2);
        return (unsigned)x[0] | ((unsigned)x[1] << 8);
    }
#if FNDSA_LITTLE_ENDIAN
    uint8_t *d = (uint8_t *)(void *)sc;
#    if FNDSA_UNALIGNED_16
    unsigned v = *(uint16_t *)(d + sc->dptr);
#    else
    unsigned v = (unsigned)d[sc->dptr] | ((unsigned)d[sc->dptr + 1] << 8);
#    endif
    sc->dptr += 2;
    return v;
#else
    unsigned x0 = (uint8_t)(sc->A[sc->dptr >> 3] >> ((sc->dptr & 7) << 3));
    sc->dptr++;
    unsigned x1 = (uint8_t)(sc->A[sc->dptr >> 3] >> ((sc->dptr & 7) << 3));
    sc->dptr++;
    return x0 | (x1 << 8);
#endif
}

/* Get the next 24-bit word from SHAKE. */
static inline unsigned shake_next_u24(shake_context *sc)
{
    if (sc->dptr + 2 >= sc->rate) {
        uint8_t x[3];
        shake_extract(sc, x, 3);
        return (unsigned)x[0] | ((unsigned)x[1] << 8) |
               ((unsigned)x[2] << 16);
    }
#if FNDSA_LITTLE_ENDIAN
    uint8_t *d = (uint8_t *)(void *)sc;
#    if FNDSA_UNALIGNED_16
    unsigned v = (*(uint32_t *)(d + sc->dptr)) & 0xffffff;
#    else
    unsigned v = (unsigned)d[sc->dptr] | ((unsigned)d[sc->dptr + 1] << 8) |
                 ((unsigned)d[sc->dptr + 2] << 16);
#    endif
    sc->dptr += 3;
    return v;
#else
    unsigned x0 = (uint8_t)(sc->A[sc->dptr >> 3] >> ((sc->dptr & 7) << 3));
    sc->dptr++;
    unsigned x1 = (uint8_t)(sc->A[sc->dptr >> 3] >> ((sc->dptr & 7) << 3));
    sc->dptr++;
    unsigned x2 = (uint8_t)(sc->A[sc->dptr >> 3] >> ((sc->dptr & 7) << 3));
    sc->dptr++;
    return x0 | (x1 << 8) | (x2 << 16);
#endif
}

/* Get the next 64-bit word from SHAKE. */
static inline uint64_t shake_next_u64(shake_context *sc)
{
    if ((sc->dptr + 7) >= sc->rate) {
#if FNDSA_LITTLE_ENDIAN
        uint64_t v;
        shake_extract(sc, &v, 8);
        return v;
#else
        uint8_t x[8];
        shake_extract(sc, x, 8);
        return (uint64_t)x[0] | ((uint64_t)x[1] << 8) |
               ((uint64_t)x[2] << 16) | ((uint64_t)x[3] << 24) |
               ((uint64_t)x[4] << 32) | ((uint64_t)x[5] << 40) |
               ((uint64_t)x[6] << 48) | ((uint64_t)x[7] << 56);
#endif
    }
    uint64_t x;
#if FNDSA_LITTLE_ENDIAN && FNDSA_UNALIGNED_64
    x = *(uint64_t *)((uint8_t *)(void *)sc + sc->dptr);
#else
    size_t j = sc->dptr >> 3;
    unsigned n = sc->dptr & 7;
    if (n == 0) {
        x = sc->A[j];
    } else {
        x = sc->A[j] >> (n << 3);
        x |= sc->A[j + 1] << (64 - (n << 3));
    }
#endif
    sc->dptr += 8;
    return x;
}

/* By default, we use a simple SHAKE256 for internal PRNG needs
   (in keygen to generate (f,g), in signing for the Gaussian sampling).
   If FNDSA_SHAKE256X4 is non-zero, then SHAKE256x4 is used: it is a
   PRNG consisting of four SHAKE256 running in parallel, with interleaved
   outputs. This has two main effects:
     - On x86 with AVX2 support, this makes signing faster (by about 20%).
     - It increases stack usage by about 1.1 kB, which can be a concern
       for small embedded systems (e.g. ARM Cortex-M4).
   It otherwise has no real perceivable effect, except that (of course)
   it changes the exact key pairs and signature values obtained from a
   given seed. */
#ifndef FNDSA_SHAKE256X4
#    define FNDSA_SHAKE256X4 0
#endif

#if FNDSA_SHAKE256X4
/*
 * SHAKE256x4 is a PRNG based on SHAKE256; it runs four SHAKE256 instances
 * in parallel, interleaving their outputs with 64-bit granularity. The
 * four instances are initialized with a common seed, followed by a single
 * byte of value 0x00, 0x01, 0x02 or 0x03, depending on the SHAKE instance.
 */

typedef struct {
    uint64_t state[100];
    uint8_t buf[4 * 136];
    unsigned ptr;
#    if FNDSA_AVX2
    int use_avx2;
#    endif
} shake256x4_context;

/* Initialize a SHAKE256x4 context from a given seed.
   WARNING: seed length MUST NOT exceed 134 bytes. */
#    define shake256x4_init fndsa_shake256x4_init
void shake256x4_init(shake256x4_context *sc, const void *seed,
                     size_t seed_len);
/* Refill the SHAKE256x4 output buffer. */
#    define shake256x4_refill fndsa_shake256x4_refill
void shake256x4_refill(shake256x4_context *sc);

/* Get the next byte of pseudorandom output. */
static inline uint8_t shake256x4_next_u8(shake256x4_context *sc)
{
    if (sc->ptr >= sizeof sc->buf) {
        shake256x4_refill(sc);
    }
    return sc->buf[sc->ptr++];
}

/* Get the next 16-bit word of pseudorandom output. */
static inline unsigned shake256x4_next_u16(shake256x4_context *sc)
{
    if (sc->ptr >= (sizeof sc->buf) - 1) {
        shake256x4_refill(sc);
    }
    unsigned x =
        (unsigned)sc->buf[sc->ptr] | ((unsigned)sc->buf[sc->ptr + 1] << 8);
    sc->ptr += 2;
    return x;
}

/* Get the next 24-bit word of pseudorandom output. */
static inline unsigned shake256x4_next_u24(shake256x4_context *sc)
{
    if (sc->ptr >= (sizeof sc->buf) - 2) {
        shake256x4_refill(sc);
    }
    unsigned x = (unsigned)sc->buf[sc->ptr] |
                 ((unsigned)sc->buf[sc->ptr + 1] << 8) |
                 ((unsigned)sc->buf[sc->ptr + 2] << 16);
    sc->ptr += 3;
    return x;
}

/* Get the next 64-bit word of pseudorandom output. */
static inline uint64_t shake256x4_next_u64(shake256x4_context *sc)
{
    if (sc->ptr >= (sizeof sc->buf) - 7) {
        shake256x4_refill(sc);
    }
    uint64_t x = (uint64_t)sc->buf[sc->ptr] |
                 ((uint64_t)sc->buf[sc->ptr + 1] << 8) |
                 ((uint64_t)sc->buf[sc->ptr + 2] << 16) |
                 ((uint64_t)sc->buf[sc->ptr + 3] << 24) |
                 ((uint64_t)sc->buf[sc->ptr + 4] << 32) |
                 ((uint64_t)sc->buf[sc->ptr + 5] << 40) |
                 ((uint64_t)sc->buf[sc->ptr + 6] << 48) |
                 ((uint64_t)sc->buf[sc->ptr + 7] << 56);
    sc->ptr += 8;
    return x;
}
#endif

/*
 * SHA-3 implementation.
 * This is a variation on the SHAKE implementation, which implements
 * SHA-3 with outputs of 224, 256, 384 and 512 bits. The output size
 * (in bits) is provided as parameter to sha3_init(). Input is provided
 * with sha3_update(). The output is computed with sha3_close(); this
 * function also reinitializes the context.
 */
typedef shake_context sha3_context;

/* Initialize context, size = 224, 256, 384 or 512 */
void sha3_init(sha3_context *sc, unsigned size);
/* Inject some bytes in context. */
void sha3_update(sha3_context *sc, const void *in, size_t len);
/* Compute the output and reinitialize the context. */
void sha3_close(sha3_context *sc, void *out);

#endif