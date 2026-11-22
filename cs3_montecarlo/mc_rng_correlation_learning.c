
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <omp.h>

#define MAX_THREADS 32

typedef struct {
    double sum;
    double sum_sq;
} Result;

// ====================================
// STEP 1: Seed mixing
// ====================================

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z =
        (*state += UINT64_C(0x9E3779B97F4A7C15));

    z = (z ^ (z >> 30))
      * UINT64_C(0xBF58476D1CE4E5B9);

    z = (z ^ (z >> 27))
      * UINT64_C(0x94D049BB133111EB);

    return z ^ (z >> 31);
}

// ====================================
// STEP 2: Random-number generation
// ====================================

static uint64_t next_random(uint64_t *state) {
    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return x * UINT64_C(2685821657736338717);
}

static double uniform_random(uint64_t *state) {
    uint64_t bits = next_random(state) >> 11;

    return ((double)bits + 0.5)
         / 9007199254740992.0;
}

static double normal_random(uint64_t *state) {
    double u1 = uniform_random(state);
    double u2 = uniform_random(state);

    return sqrt(-2.0 * log(u1))
         * cos(6.28318530717958647692 * u2);
}

// ====================================
// STEP 3: Simulate one discounted payoff
// ====================================

static double simulate(uint64_t *state) {

    const double S0 = 100.0;
    const double K = 100.0;

    // r=0.05, sigma=0.20, T=1
    const double drift = 0.03;
    const double diffusion = 0.20;

    double z = normal_random(state);

    double final_price =
        S0 * exp(drift + diffusion * z);

    return exp(-0.05) *
           fmax(final_price - K, 0.0);
}

// ====================================
// STEP 4: Statistical standard error
// ====================================

static double standard_error(
    Result result,
    uint64_t n
) {
    double count = (double)n;

    double variance =
        (result.sum_sq -
         result.sum * result.sum / count)
        / (count - 1.0);

    if (variance < 0.0)
        variance = 0.0;

    return sqrt(variance / count);
}

// ====================================
// STEP 5: Main experiment
// ====================================

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr,
            "Usage: %s threads paths_per_thread\n",
            argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);

    uint64_t per_thread =
        strtoull(argv[2], NULL, 10);

    if (threads < 1 ||
        threads > MAX_THREADS ||
        per_thread < 2 ||
        per_thread > 10000000) {

        fprintf(stderr, "Invalid arguments\n");
        return 1;
    }

    const uint64_t BASE_SEED =
        UINT64_C(0x123456789ABCDEF);

    Result bad[MAX_THREADS] = {{0}};
    Result good[MAX_THREADS] = {{0}};

    omp_set_dynamic(0);

    // ====================================
    // STEP 6: Run both RNG approaches
    // ====================================

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();

        // BAD: Every thread has the same
        // initial state.
        uint64_t bad_state = BASE_SEED;

        // BETTER: Mix each thread's seed.
        uint64_t seed_material =
            BASE_SEED
            + (uint64_t)tid *
              UINT64_C(0x9E3779B97F4A7C15);

        uint64_t good_state =
            splitmix64(&seed_material);

        if (good_state == 0)
            good_state = UINT64_C(1);

        Result bad_local = {0.0, 0.0};
        Result good_local = {0.0, 0.0};

        for (uint64_t i = 0; i < per_thread; i++) {

            double x = simulate(&bad_state);
            double y = simulate(&good_state);

            bad_local.sum += x;
            bad_local.sum_sq += x * x;

            good_local.sum += y;
            good_local.sum_sq += y * y;
        }

        bad[tid] = bad_local;
        good[tid] = good_local;
    }

    // ====================================
    // STEP 7: Combine thread results
    // ====================================

    Result bad_total = {0.0, 0.0};
    Result good_total = {0.0, 0.0};

    int identical = 1;

    for (int t = 0; t < threads; t++) {

        bad_total.sum += bad[t].sum;
        bad_total.sum_sq += bad[t].sum_sq;

        good_total.sum += good[t].sum;
        good_total.sum_sq += good[t].sum_sq;

        if (bad[t].sum != bad[0].sum ||
            bad[t].sum_sq != bad[0].sum_sq) {

            identical = 0;
        }
    }

    uint64_t nominal =
        per_thread * (uint64_t)threads;

    // ====================================
    // STEP 8: Calculate standard errors
    // ====================================

    // Incorrect assumption:
    // All simulated paths were independent.
    double naive_se =
        standard_error(bad_total, nominal);

    // Correct for the exact duplication:
    // Only one block contains unique draws.
    double effective_se =
        standard_error(bad[0], per_thread);

    double good_se =
        standard_error(good_total, nominal);

    // ====================================
    // STEP 9: Print results
    // ====================================

    printf("===== RNG CORRELATION EXPERIMENT =====\n");

    printf("Threads: %d\n", threads);

    printf("Paths per thread: %" PRIu64 "\n",
           per_thread);

    printf("Nominal paths: %" PRIu64 "\n",
           nominal);

    printf("\n===== IDENTICAL SEEDS =====\n");

    printf("Identical worker results: %s\n",
           identical ? "YES" : "NO");

    printf("Distinct generated paths: %" PRIu64 "\n",
           per_thread);

    printf("Monte Carlo price: $%.8f\n",
           bad_total.sum / (double)nominal);

    printf("Naive standard error: %.8f\n",
           naive_se);

    printf("Effective standard error: %.8f\n",
           effective_se);

    printf("SE underestimation factor: %.3fx\n",
           effective_se / naive_se);

    printf("\n===== SEPARATELY SEEDED =====\n");

    printf("Monte Carlo price: $%.8f\n",
           good_total.sum / (double)nominal);

    printf("Estimated standard error: %.8f\n",
           good_se);

    printf("\nExperiment complete.\n");

    return 0;
}
