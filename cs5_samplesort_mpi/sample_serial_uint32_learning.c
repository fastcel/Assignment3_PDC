#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <limits.h>

#include "sample_input_learning.h"

/*
 * Educational experiment:
 * Sort native 32-bit values without
 * storing the original element index.
 */

static int compare_values(
    const void *left,
    const void *right
) {
    uint32_t a = *(const uint32_t *)left;
    uint32_t b = *(const uint32_t *)right;

    return (a > b) - (a < b);
}

static double wall_time(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    return (double)ts.tv_sec
         + (double)ts.tv_nsec * 1e-9;
}

static uint32_t generate_value(
    uint64_t global_index,
    int distribution
) {
    uint64_t packed = learning_key(
        global_index,
        distribution
    );

    return (uint32_t)(packed >> 32);
}

static uint64_t value_fingerprint(uint32_t value) {
    return learning_mix64(
        (uint64_t)value ^
        UINT64_C(0xD6E8FEB86659FD93)
    );
}

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(
            stderr,
            "Usage: %s TOTAL uniform|duplicates\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }

    uint64_t total = 0;

    int distribution =
        learning_distribution(argv[2]);

    if (
        !learning_parse_count(argv[1], &total) ||
        distribution < 0 ||
        total > SIZE_MAX / sizeof(uint32_t)
    ) {
        fprintf(stderr, "Invalid arguments\n");
        return EXIT_FAILURE;
    }

    size_t n = (size_t)total;

    uint32_t *data = malloc(
        n * sizeof(uint32_t)
    );

    if (data == NULL) {
        fprintf(stderr, "Allocation failed\n");
        return EXIT_FAILURE;
    }

    uint64_t input_sum = 0;
    uint64_t input_xor = 0;

    // Generate exactly the same integer values
    // as the old packed-key implementation.

    for (size_t i = 0; i < n; i++) {

        uint32_t value = generate_value(
            (uint64_t)i,
            distribution
        );

        data[i] = value;

        uint64_t hash =
            value_fingerprint(value);

        input_sum += hash;
        input_xor ^= hash;
    }

    // Only the sorting operation is timed.

    double started = wall_time();

    qsort(
        data,
        n,
        sizeof(uint32_t),
        compare_values
    );

    double elapsed = wall_time() - started;

    // Allow duplicate adjacent values.
    // Reject only decreasing values.

    int ordered = 1;

    for (size_t i = 1; i < n; i++) {
        if (data[i - 1] > data[i]) {
            ordered = 0;
            break;
        }
    }

    uint64_t output_sum = 0;
    uint64_t output_xor = 0;

    for (size_t i = 0; i < n; i++) {

        uint64_t hash =
            value_fingerprint(data[i]);

        output_sum += hash;
        output_xor ^= hash;
    }

    int preserved =
        input_sum == output_sum &&
        input_xor == output_xor;

    int correct = ordered && preserved;

    printf("===== NATIVE 32-BIT SERIAL SORT =====\n");
    printf("Distribution: %s\n", argv[2]);
    printf("Total integers: %" PRIu64 "\n", total);
    printf("Element bytes: %zu\n", sizeof(uint32_t));
    printf(
        "Input storage: %.3f MB\n",
        (double)n * sizeof(uint32_t) / 1e6
    );
    printf("Sorting time: %.6f s\n", elapsed);

    printf(
        "Input fingerprint sum: 0x%016" PRIx64 "\n",
        input_sum
    );
    printf(
        "Input fingerprint xor: 0x%016" PRIx64 "\n",
        input_xor
    );

    printf(
        "Output fingerprint sum: 0x%016" PRIx64 "\n",
        output_sum
    );
    printf(
        "Output fingerprint xor: 0x%016" PRIx64 "\n",
        output_xor
    );

    printf(
        "Local ordering: %s\n",
        ordered ? "PASS" : "FAIL"
    );

    printf(
        "Elements preserved: %s\n",
        preserved ? "PASS" : "FAIL"
    );

    printf(
        "Correctness: %s\n",
        correct ? "PASS" : "FAIL"
    );

    free(data);

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
