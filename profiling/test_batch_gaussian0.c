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
    int32_t z_bimodal, z_square;

    sampler_init(&ss, 9, seed, 32);
    store = BATCH_STORE_GAUSSIAN0_new(&ss);

    for (int i = 0; i < 10000; i++) {
        BATCH_STORE_GAUSSIAN0_get_next(store, &z_bimodal, &z_square);
        printf("%d,%d, ", z_bimodal, z_square);
        if (i != 0 && i % 32 == 0)
            printf("\n");
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
    int32_t z_bimodal, z_square;
    int n_way = 16;

    init_perf_counters();
    sampler_init(&ss, 9, seed, 32);
    store = BATCH_STORE_GAUSSIAN0_new(&ss);

    PERF_N(
        gaussian0(&ss, store->_z_bimodal.coeffs, store->_z_square.coeffs),
        gaussian0, sampler_init(&ss, 9, seed, 32), WARMUP_N, TESTS_N,
        n_way);

    PERF(
        BATCH_STORE_GAUSSIAN0_get_next(store, &z_bimodal, &z_square),
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