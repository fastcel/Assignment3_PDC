
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// -------------------------------------
// STEP 1: Reproducible random generator
// -------------------------------------

static uint64_t next_random(uint64_t *state) {

    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return x * UINT64_C(2685821657736338717);
}

// Generate a uniform random number in (0, 1)
static double uniform_random(uint64_t *state) {

    uint64_t bits = next_random(state) >> 11;

    return ((double)bits + 0.5)
         / 9007199254740992.0;
}

// -------------------------------------
// STEP 2: Generate a standard normal Z
// Using the Box-Muller transformation
// -------------------------------------

static double normal_random(uint64_t *state) {

    double u1 = uniform_random(state);
    double u2 = uniform_random(state);

    const double TWO_PI =
        6.28318530717958647692;

    return sqrt(-2.0 * log(u1))
         * cos(TWO_PI * u2);
}

// -------------------------------------
// STEP 3: Standard normal CDF
// -------------------------------------

static double normal_cdf(double x) {

    return 0.5 * erfc(-x / sqrt(2.0));
}

// -------------------------------------
// STEP 4: Black-Scholes reference price
// -------------------------------------

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

// -------------------------------------
// STEP 5: Main Monte Carlo experiment
// -------------------------------------

int main(int argc, char **argv) {

    if (argc != 2) {
        fprintf(stderr,
            "Usage: %s number_of_paths\n",
            argv[0]);
        return 1;
    }

    char *end = NULL;

    unsigned long long requested =
        strtoull(argv[1], &end, 10);

    if (end == argv[1] || *end != '\0'
        || requested < 2) {

        fprintf(stderr,
            "Please enter at least 2 paths.\n");
        return 1;
    }

    uint64_t M = (uint64_t)requested;

    // Option parameters
    const double S0 = 100.0;
    const double K = 100.0;
    const double r = 0.05;
    const double sigma = 0.20;
    const double T = 1.0;

    // Fixed seed ensures repeatability.
    uint64_t state =
        UINT64_C(0x123456789ABCDEF);

    double sum = 0.0;
    double sum_squares = 0.0;

    // Precompute terms that don't change.
    double drift =
        (r - 0.5 * sigma * sigma) * T;

    double diffusion = sigma * sqrt(T);

    double discount = exp(-r * T);

    // -------------------------------------
    // STEP 6: Simulate independent paths
    // -------------------------------------

    for (uint64_t i = 0; i < M; i++) {

        double Z = normal_random(&state);

        double ST = S0 * exp(
            drift + diffusion * Z
        );

        double payoff = fmax(ST - K, 0.0);

        double discounted_payoff =
            discount * payoff;

        sum += discounted_payoff;

        sum_squares +=
            discounted_payoff * discounted_payoff;
    }

    // -------------------------------------
    // STEP 7: Estimate price and uncertainty
    // -------------------------------------

    double estimate = sum / (double)M;

    double sample_variance = (
        sum_squares - sum * sum / (double)M
    ) / (double)(M - 1);

    // Avoid a tiny negative due to rounding.
    if (sample_variance < 0.0) {
        sample_variance = 0.0;
    }

    double standard_error =
        sqrt(sample_variance / (double)M);

    // -------------------------------------
    // STEP 8: Analytical reference
    // -------------------------------------

    double exact = black_scholes(
        S0, K, r, sigma, T
    );

    double absolute_error = fabs(
        estimate - exact
    );

    double ci_lower =
        estimate - 1.96 * standard_error;

    double ci_upper =
        estimate + 1.96 * standard_error;

    // -------------------------------------
    // STEP 9: Display results
    // -------------------------------------

    printf("===== MONTE CARLO OPTION PRICING =====\n");

    printf("Simulated paths: %llu\n",
           (unsigned long long)M);

    printf("S0 = %.2f, K = %.2f\n", S0, K);
    printf("r = %.2f, sigma = %.2f, T = %.2f\n",
           r, sigma, T);

    printf("\n===== RESULTS =====\n");

    printf("Monte Carlo price: $%.6f\n",
           estimate);

    printf("Black-Scholes price: $%.6f\n",
           exact);

    printf("Absolute error: %.6f\n",
           absolute_error);

    printf("Standard error: %.6f\n",
           standard_error);

    printf("Approx. 95%% CI: [$%.6f, $%.6f]\n",
           ci_lower, ci_upper);

    return 0;
}
