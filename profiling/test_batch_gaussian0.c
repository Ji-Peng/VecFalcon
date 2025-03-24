#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "batch_gaussian0.h"
#include "cpucycles.h"

void test_gaussian0()
{
    sampler_state ss;
    BATCH_STORE_GAUSSIAN0 *store = NULL;
    uint8_t seed[32] = {0};

    sampler_init(&ss, 9, seed, 32);
    store = BATCH_STORE_GAUSSIAN0_new(&ss);

    for (int i = 0; i < 10000000; i++) {
        printf("%d, ", BATCH_STORE_GAUSSIAN0_get_next(store));
    }
    printf("\n");

    BATCH_STORE_GAUSSIAN0_free(store);
}

#define WARMUP_N 1000
// 136*4-(136*4%3)=543
// 543*3 % 9 = 0
#define TESTS_N ((543 * 3) * 100)

void speed_gaussian0()
{
    sampler_state ss;
    BATCH_STORE_GAUSSIAN0 *store = NULL;
    uint8_t seed[32] = {0};
    int n_way;

#if FNDSA_AVX2 == 1 && BATCH_GAUSSIAN0 == 1
    n_way = 16;
#elif FNDSA_SSE2 == 1 && BATCH_GAUSSIAN0 == 1
    n_way = 8;
#else
    n_way = 1;
#endif

    init_perf_counters();
    sampler_init(&ss, 9, seed, 32);
    store = BATCH_STORE_GAUSSIAN0_new(&ss);

    PERF_N(gaussian0(&ss, store->data.coeffs), gaussian0,
           sampler_init(&ss, 9, seed, 32), WARMUP_N, TESTS_N, n_way);

    PERF(
        BATCH_STORE_GAUSSIAN0_get_next(store),
        BATCH_STORE_GAUSSIAN0_get_next,
        {
            sampler_init(&ss, 9, seed, 32);
            store->current_pos = store->batch_size;
        },
        WARMUP_N, TESTS_N);
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: %s <argument>\n", argv[0]);
        printf("Argument must be either 'test' or 'speed'.\n");
        return 1;
    }

    char *arg = argv[1];

    if (strcmp(arg, "test") == 0) {
        test_gaussian0();
    } else if (strcmp(arg, "speed") == 0) {
        speed_gaussian0();
    } else {
        printf("Invalid argument: '%s'\n", arg);
        printf("Argument must be either 'test' or 'speed'.\n");
        return 1;
    }

    return 0;
}