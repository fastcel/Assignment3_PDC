
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define MAX_THREADS 64
#define SAMPLES 3

// Mix a seed into a well-distributed state.
static uint64_t splitmix64(uint64_t *state) {

    uint64_t z = (
        *state += UINT64_C(0x9E3779B97F4A7C15)
    );

    z = (z ^ (z >> 30))
      * UINT64_C(0xBF58476D1CE4E5B9);

    z = (z ^ (z >> 27))
      * UINT64_C(0x94D049BB133111EB);

    return z ^ (z >> 31);
}

// Small deterministic PRNG.
static uint64_t next_random(uint64_t *state) {

    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return x * UINT64_C(2685821657736338717);
}

// Uniform value in (0, 1).
static double uniform_random(uint64_t *state) {

    uint64_t bits = next_random(state) >> 11;

    return ((double)bits + 0.5)
         / 9007199254740992.0;
}

int main(void) {

    const uint64_t BASE_SEED =
        UINT64_C(0x123456789ABCDEF);

    int workers = omp_get_max_threads();

    if (workers < 1 || workers > MAX_THREADS) {
        fprintf(stderr, "Unsupported thread count\n");
        return 1;
    }

    double bad[MAX_THREADS][SAMPLES] = {{0}};
    double good[MAX_THREADS][SAMPLES] = {{0}};

    omp_set_dynamic(0);

    #pragma omp parallel num_threads(workers)
    {
        int tid = omp_get_thread_num();

        // BAD: Every thread uses the same seed.
        uint64_t bad_state = BASE_SEED;

        // BETTER: Derive a different initial
        // state for every worker.
        uint64_t seed_material =
            BASE_SEED
            + (uint64_t)tid *
              UINT64_C(0x9E3779B97F4A7C15);

        uint64_t good_state =
            splitmix64(&seed_material);

        if (good_state == 0) {
            good_state = UINT64_C(1);
        }

        for (int i = 0; i < SAMPLES; i++) {

            bad[tid][i] =
                uniform_random(&bad_state);

            good[tid][i] =
                uniform_random(&good_state);
        }
    }

    // Print outside the parallel region so
    // output appears in a predictable order.

    printf("===== SAME SEED (INCORRECT) =====\n");

    for (int t = 0; t < workers; t++) {

        printf("Thread %d:", t);

        for (int i = 0; i < SAMPLES; i++) {
            printf(" %.8f", bad[t][i]);
        }

        printf("\n");
    }

    printf("\n===== DIFFERENT STREAM STATES =====\n");

    for (int t = 0; t < workers; t++) {

        printf("Thread %d:", t);

        for (int i = 0; i < SAMPLES; i++) {
            printf(" %.8f", good[t][i]);
        }

        printf("\n");
    }

    printf("\nExperiment complete.\n");

    return 0;
}
