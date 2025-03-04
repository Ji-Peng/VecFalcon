#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cpucycles.h"
#include "speed_print.h"

/*
 * This code uses only the external API.
 */

#include "falcon.h"

static void *xmalloc(size_t len)
{
    void *buf;

    if (len == 0) {
        return NULL;
    }
    buf = malloc(len);
    if (buf == NULL) {
        fprintf(stderr, "memory allocation error\n");
        exit(EXIT_FAILURE);
    }
    return buf;
}

static void xfree(void *buf)
{
    if (buf != NULL) {
        free(buf);
    }
}

typedef struct {
    unsigned logn;
    shake256_context rng;
    uint8_t *tmp;
    size_t tmp_len;
    uint8_t *pk;
    uint8_t *sk;
    uint8_t *esk;
    uint8_t *sig;
    size_t sig_len;
    uint8_t *sigct;
    size_t sigct_len;
} bench_context;

static inline size_t maxsz(size_t a, size_t b)
{
    return a > b ? a : b;
}

#define CC(x)           \
    do {                \
        int ccr = (x);  \
        if (ccr != 0) { \
            return ccr; \
        }               \
    } while (0)

static int test_speed_falcon(unsigned logn, size_t num)
{
    bench_context bctx;
    bench_context *bc = &bctx;
    size_t len, i;
    uint64_t *t;

    t = xmalloc(num * 10 * sizeof(uint64_t));
    printf("n = %4u\n", 1u << logn);
    fflush(stdout);

    bctx.logn = logn;
    if (shake256_init_prng_from_system(&bctx.rng) != 0) {
        fprintf(stderr, "random seeding failed\n");
        exit(EXIT_FAILURE);
    }
    len = FALCON_TMPSIZE_KEYGEN(logn);
    len = maxsz(len, FALCON_TMPSIZE_SIGNDYN(logn));
    len = maxsz(len, FALCON_TMPSIZE_SIGNTREE(logn));
    len = maxsz(len, FALCON_TMPSIZE_EXPANDPRIV(logn));
    len = maxsz(len, FALCON_TMPSIZE_VERIFY(logn));
    bctx.tmp = xmalloc(len);
    bctx.tmp_len = len;
    bctx.pk = xmalloc(FALCON_PUBKEY_SIZE(logn));
    bctx.sk = xmalloc(FALCON_PRIVKEY_SIZE(logn));
    bctx.esk = xmalloc(FALCON_EXPANDEDKEY_SIZE(logn));
    bctx.sig = xmalloc(FALCON_SIG_COMPRESSED_MAXSIZE(logn));
    bctx.sig_len = 0;
    bctx.sigct = xmalloc(FALCON_SIG_CT_SIZE(logn));
    bctx.sigct_len = 0;

    for (i = 0; i < 100; i++) {
        t[i] = cpucycles();
        CC(falcon_keygen_make(
            &bc->rng, bc->logn, bc->sk, FALCON_PRIVKEY_SIZE(bc->logn),
            bc->pk, FALCON_PUBKEY_SIZE(bc->logn), bc->tmp, bc->tmp_len));
    }
    print_results("falcon_keygen_make: ", t, 100);

    for (i = 0; i < num * 10; i++) {
        t[i] = cpucycles();
        bc->sig_len = FALCON_SIG_COMPRESSED_MAXSIZE(bc->logn);
        CC(falcon_sign_dyn(&bc->rng, bc->sig, &bc->sig_len,
                           FALCON_SIG_COMPRESSED, bc->sk,
                           FALCON_PRIVKEY_SIZE(bc->logn), "data", 4,
                           bc->tmp, bc->tmp_len));
    }
    print_results_average("falcon_sign_dyn: ", t, num * 10);

    CC(falcon_keygen_make(
        &bc->rng, bc->logn, bc->sk, FALCON_PRIVKEY_SIZE(bc->logn), bc->pk,
        FALCON_PUBKEY_SIZE(bc->logn), bc->tmp, bc->tmp_len));

    for (i = 0; i < num * 10; i++) {
        t[i] = cpucycles();
        bc->sigct_len = FALCON_SIG_CT_SIZE(bc->logn);
        CC(falcon_sign_dyn(&bc->rng, bc->sigct, &bc->sigct_len,
                           FALCON_SIG_CT, bc->sk,
                           FALCON_PRIVKEY_SIZE(bc->logn), "data", 4,
                           bc->tmp, bc->tmp_len));
    }
    print_results_average("falcon_sign_dyn_ct: ", t, num * 10);

    for (i = 0; i < num; i++) {
        t[i] = cpucycles();
        CC(falcon_expand_privkey(
            bc->esk, FALCON_EXPANDEDKEY_SIZE(bc->logn), bc->sk,
            FALCON_PRIVKEY_SIZE(bc->logn), bc->tmp, bc->tmp_len));
    }
    print_results("falcon_expand_privkey: ", t, num);

    for (i = 0; i < num * 10; i++) {
        t[i] = cpucycles();
        bc->sig_len = FALCON_SIG_COMPRESSED_MAXSIZE(bc->logn);
        CC(falcon_sign_tree(&bc->rng, bc->sig, &bc->sig_len,
                            FALCON_SIG_COMPRESSED, bc->esk, "data", 4,
                            bc->tmp, bc->tmp_len));
    }
    print_results_average("falcon_sign_tree: ", t, num * 10);

    CC(falcon_expand_privkey(bc->esk, FALCON_EXPANDEDKEY_SIZE(bc->logn),
                             bc->sk, FALCON_PRIVKEY_SIZE(bc->logn),
                             bc->tmp, bc->tmp_len));
    for (i = 0; i < num * 10; i++) {
        t[i] = cpucycles();
        bc->sigct_len = FALCON_SIG_CT_SIZE(bc->logn);
        CC(falcon_sign_tree(&bc->rng, bc->sigct, &bc->sigct_len,
                            FALCON_SIG_CT, bc->esk, "data", 4, bc->tmp,
                            bc->tmp_len));
    }
    print_results_average("falcon_sign_tree_ct: ", t, num * 10);

    size_t pk_len = FALCON_PUBKEY_SIZE(bc->logn);
    for (i = 0; i < num; i++) {
        t[i] = cpucycles();
        CC(falcon_verify(bc->sig, bc->sig_len, FALCON_SIG_COMPRESSED,
                         bc->pk, pk_len, "data", 4, bc->tmp, bc->tmp_len));
    }
    print_results("falcon_verify: ", t, num);

    pk_len = FALCON_PUBKEY_SIZE(bc->logn);
    for (i = 0; i < num; i++) {
        t[i] = cpucycles();
        CC(falcon_verify(bc->sigct, bc->sigct_len, FALCON_SIG_CT, bc->pk,
                         pk_len, "data", 4, bc->tmp, bc->tmp_len));
    }
    print_results("falcon_verify_ct: ", t, num);

    printf("\n");

    xfree(t);
    xfree(bctx.tmp);
    xfree(bctx.pk);
    xfree(bctx.sk);
    xfree(bctx.esk);
    xfree(bctx.sig);
    xfree(bctx.sigct);
    return 0;
}

int main()
{
    size_t num = 1000;
    test_speed_falcon(9, num);
    test_speed_falcon(10, num);
    return 0;
}