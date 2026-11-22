
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#include <errno.h>
#include <math.h>
#include <omp.h>

// =====================================
// STEP 1: Random-number generators
// =====================================

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z =
        (*state += UINT64_C(0x9E3779B97F4A7C15));

    z = (z ^ (z >> 30))
        * UINT64_C(0xBF58476D1CE4E5B9);

    z = (z ^ (z >> 27))
        * UINT64_C(0x94D049BB133111EB);

    return z ^ (z >> 31);
}

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

    const double TWO_PI =
        6.28318530717958647692;

    return sqrt(-2.0 * log(u1))
         * cos(TWO_PI * u2);
}

// =====================================
// STEP 2: Option pricing mathematics
// =====================================

static double normal_cdf(double x) {
    return 0.5 * erfc(-x / sqrt(2.0));
}

static double black_scholes(
    double S0,
    double K,
    double r,
    double sigma,
    double T
) {
    double d1 = (
        log(S0 / K)
        + (r + 0.5 * sigma * sigma) * T
    ) / (sigma * sqrt(T));

    double d2 = d1 - sigma * sqrt(T);

    return S0 * normal_cdf(d1)
         - K * exp(-r * T) * normal_cdf(d2);
}

static double simulate_payoff(
    uint64_t *state,
    double S0,
    double K,
    double drift,
    double diffusion,
    double discount
) {
    double Z = normal_random(state);

    double ST = S0 * exp(
        drift + diffusion * Z
    );

    double payoff = fmax(ST - K, 0.0);

    return discount * payoff;
}

// =====================================
// STEP 3: Accumulated statistics
// =====================================

typedef struct {
    double sum;
    double sum_squares;
} PartialResult;

static void print_statistics(
    const char *label,
    PartialResult result,
    uint64_t paths,
    double exact,
    double seconds
) {
    double M = (double)paths;

    double price = result.sum / M;

    double variance = (
        result.sum_squares
        - result.sum * result.sum / M
    ) / (M - 1.0);

    if (variance < 0.0)
        variance = 0.0;

    double se = sqrt(variance / M);

    printf("\n===== %s =====\n", label);

    printf("Monte Carlo price: $%.8f\n", price);
    printf("Standard error: %.8f\n", se);

    printf("Absolute error: %.8f\n",
           fabs(price - exact));

    printf("95%% CI: [$%.8f, $%.8f]\n",
           price - 1.96 * se,
           price + 1.96 * se);

    printf("Execution time: %.6f seconds\n",
           seconds);

    printf("Throughput: %.3f million paths/s\n",
           M / (seconds * 1e6));
}

// =====================================
// STEP 4: Main
// =====================================

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr,
                "Usage: %s paths threads\n",
                argv[0]);
        return 1;
    }

    errno = 0;
    char *end = NULL;

    unsigned long long requested =
        strtoull(argv[1], &end, 10);

    if (errno || argv[1][0] == '-' ||
        end == argv[1] || *end != '\0' ||
        requested < 2 ||
        requested > LLONG_MAX) {
        fprintf(stderr, "Invalid path count\n");
        return 1;
    }

    uint64_t M = (uint64_t)requested;

    errno = 0;
    end = NULL;

    long requested_threads =
        strtol(argv[2], &end, 10);

    if (errno || end == argv[2] ||
        *end != '\0' ||
        requested_threads < 1 ||
        requested_threads > 64) {
        fprintf(stderr, "Invalid thread count\n");
        return 1;
    }

    int threads = (int)requested_threads;

    // =====================================
    // STEP 5: Fixed option parameters
    // =====================================

    const double S0 = 100.0;
    const double K = 100.0;
    const double r = 0.05;
    const double sigma = 0.20;
    const double T = 1.0;

    const uint64_t BASE_SEED =
        UINT64_C(0x123456789ABCDEF);

    double drift =
        (r - 0.5 * sigma * sigma) * T;

    double diffusion = sigma * sqrt(T);
    double discount = exp(-r * T);

    double exact =
        black_scholes(S0, K, r, sigma, T);

    omp_set_dynamic(0);

    // =====================================
    // STEP 6: Serial reference
    // =====================================

    uint64_t serial_state = BASE_SEED;

    PartialResult serial = {0.0, 0.0};

    double start = omp_get_wtime();

    for (uint64_t i = 0; i < M; i++) {

        double x = simulate_payoff(
            &serial_state,
            S0, K, drift, diffusion, discount
        );

        serial.sum += x;
        serial.sum_squares += x * x;
    }

    double serial_time = omp_get_wtime() - start;

    // =====================================
    // STEP 7: Allocate partial results
    // =====================================

    PartialResult *partials = calloc(
        (size_t)threads,
        sizeof(PartialResult)
    );

    if (!partials) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // =====================================
    // STEP 8: Parallel simulation
    // =====================================

    int actual_threads = 0;

    start = omp_get_wtime();

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        actual_threads = omp_get_num_threads();

        uint64_t seed_material =
            BASE_SEED
            + (uint64_t)tid *
              UINT64_C(0x9E3779B97F4A7C15);

        uint64_t state =
            splitmix64(&seed_material);

        if (state == 0)
            state = UINT64_C(1);

        // Accumulate locally inside each worker.
        double local_sum = 0.0;
        double local_squares = 0.0;

        #pragma omp for schedule(static)
        for (long long i = 0; i < (long long)M; i++) {

            double x = simulate_payoff(
                &state,
                S0, K, drift, diffusion, discount
            );

            local_sum += x;
            local_squares += x * x;
        }

        partials[tid].sum = local_sum;
        partials[tid].sum_squares = local_squares;
    }

    // =====================================
    // STEP 9: Fixed-order reduction
    // =====================================

    PartialResult parallel = {0.0, 0.0};

    for (int t = 0; t < actual_threads; t++) {
        parallel.sum += partials[t].sum;

        parallel.sum_squares +=
            partials[t].sum_squares;
    }

    double parallel_time =
        omp_get_wtime() - start;

    // =====================================
    // STEP 10: Display results
    // =====================================

    printf("===== FULL-SIZED MONTE CARLO =====\n");

    printf("Paths: %" PRIu64 "\n", M);

    printf("Requested threads: %d\n", threads);
    printf("Actual threads: %d\n", actual_threads);

    printf("Black-Scholes price: $%.8f\n", exact);

    print_statistics(
        "SERIAL",
        serial, M, exact, serial_time
    );

    print_statistics(
        "OPENMP",
        parallel, M, exact, parallel_time
    );

    printf("\n===== PERFORMANCE =====\n");

    printf("Speedup vs serial: %.3fx\n",
           serial_time / parallel_time);

    printf("Parallel efficiency: %.2f%%\n",
           100.0 * serial_time /
           (parallel_time * actual_threads));

    free(partials);

    return 0;
}
