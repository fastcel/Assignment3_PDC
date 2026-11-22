
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>
#include <omp.h>

// --------------------------------------
// STEP 1: Seed mixing
// --------------------------------------

static uint64_t splitmix64(uint64_t *state) {

    uint64_t z =
        (*state += UINT64_C(0x9E3779B97F4A7C15));

    z = (z ^ (z >> 30))
        * UINT64_C(0xBF58476D1CE4E5B9);

    z = (z ^ (z >> 27))
        * UINT64_C(0x94D049BB133111EB);

    return z ^ (z >> 31);
}

// --------------------------------------
// STEP 2: Thread-local random generator
// --------------------------------------

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

// --------------------------------------
// STEP 3: Standard normal random number
// --------------------------------------

static double normal_random(uint64_t *state) {

    double u1 = uniform_random(state);
    double u2 = uniform_random(state);

    const double TWO_PI =
        6.28318530717958647692;

    return sqrt(-2.0 * log(u1))
           * cos(TWO_PI * u2);
}

// --------------------------------------
// STEP 4: Black-Scholes reference
// --------------------------------------

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

// --------------------------------------
// STEP 5: Main function
// --------------------------------------

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr,
                "Usage: %s paths threads\n",
                argv[0]);
        return 1;
    }

    errno = 0;
    char *end = NULL;

    long long M = strtoll(argv[1], &end, 10);

    if (errno || end == argv[1] ||
        *end != '\0' || M < 2 ||
        M > 1000000) {

        fprintf(stderr,
                "Use between 2 and 1000000 paths.\n");
        return 1;
    }

    errno = 0;
    end = NULL;

    long requested_threads =
        strtol(argv[2], &end, 10);

    if (errno || end == argv[2] ||
        *end != '\0' ||
        requested_threads < 1 ||
        requested_threads > 64) {

        fprintf(stderr,
                "Thread count must be 1 to 64.\n");
        return 1;
    }

    int threads = (int)requested_threads;

    // Option parameters
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

    double sum = 0.0;
    double sum_squares = 0.0;

    omp_set_dynamic(0);

    // --------------------------------------
    // STEP 6: Parallel Monte Carlo
    // --------------------------------------

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(threads) \
        reduction(+:sum,sum_squares)
    {
        int tid = omp_get_thread_num();

        // Give each thread its own RNG state.
        uint64_t seed_material =
            BASE_SEED
            + (uint64_t)tid *
              UINT64_C(0x9E3779B97F4A7C15);

        uint64_t state =
            splitmix64(&seed_material);

        if (state == 0) {
            state = UINT64_C(1);
        }

        // Static scheduling provides a
        // repeatable assignment of iterations.
        #pragma omp for schedule(static)
        for (long long i = 0; i < M; i++) {

            double Z = normal_random(&state);

            double ST = S0 * exp(
                drift + diffusion * Z
            );

            double payoff = fmax(
                ST - K, 0.0
            );

            double discounted =
                discount * payoff;

            sum += discounted;

            sum_squares +=
                discounted * discounted;
        }
    }

    double elapsed = omp_get_wtime() - start;

    // --------------------------------------
    // STEP 7: Statistical calculations
    // --------------------------------------

    double price = sum / (double)M;

    double variance = (
        sum_squares
        - sum * sum / (double)M
    ) / (double)(M - 1);

    if (variance < 0.0) {
        variance = 0.0;
    }

    double standard_error =
        sqrt(variance / (double)M);

    double exact = black_scholes(
        S0, K, r, sigma, T
    );

    double absolute_error =
        fabs(price - exact);

    // --------------------------------------
    // STEP 8: Results
    // --------------------------------------

    printf("===== OPENMP MONTE CARLO =====\n");

    printf("Paths: %lld\n", M);
    printf("Threads: %d\n", threads);

    printf("\n===== RESULTS =====\n");

    printf("Monte Carlo price: $%.8f\n",
           price);

    printf("Black-Scholes price: $%.8f\n",
           exact);

    printf("Absolute error: %.8f\n",
           absolute_error);

    printf("Standard error: %.8f\n",
           standard_error);

    printf("Approx. 95%% CI: [$%.8f, $%.8f]\n",
           price - 1.96 * standard_error,
           price + 1.96 * standard_error);

    printf("Execution time: %.6f seconds\n",
           elapsed);

    printf("Throughput: %.3f million paths/s\n",
           (double)M / (elapsed * 1e6));

    return 0;
}
