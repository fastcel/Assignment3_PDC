
#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sample_input_learning.h"

// ============================================
// STEP 1: Comparison function
// ============================================

static int compare_keys(
    const void *left,
    const void *right
) {
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;

    return (a > b) - (a < b);
}

// ============================================
// STEP 2: Monotonic wall-clock timer
// ============================================

static double wall_time(void) {
    struct timespec now;

    if (
        clock_gettime(
            CLOCK_MONOTONIC,
            &now
        ) != 0
    ) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    return (double)now.tv_sec +
        (double)now.tv_nsec * 1e-9;
}

// ============================================
// STEP 3: Main
// ============================================

int main(int argc, char **argv) {

    uint64_t total = 0;

    int distribution = argc == 3
        ? learning_distribution(argv[2])
        : -1;

    if (
        argc != 3 ||
        !learning_parse_count(argv[1], &total) ||
        distribution < 0
    ) {
        fprintf(
            stderr,
            "Usage: ./sample_serial_baseline_learning "
            "TOTAL uniform|duplicates\n"
        );

        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 4: Allocate the input array
    // ========================================

    uint64_t *data = malloc(
        (size_t)total * sizeof(uint64_t)
    );

    if (data == NULL) {
        fprintf(
            stderr,
            "Unable to allocate %" PRIu64
            " sorting keys.\n",
            total
        );

        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 5: Generate the input
    // ========================================

    uint64_t initial_sum = 0;
    uint64_t initial_xor = 0;

    for (uint64_t i = 0; i < total; i++) {

        uint64_t key = learning_key(
            i,
            distribution
        );

        data[i] = key;

        uint64_t h =
            learning_fingerprint(key);

        initial_sum += h;
        initial_xor ^= h;
    }

    // ========================================
    // STEP 6: Time serial sorting
    // ========================================

    double start = wall_time();

    qsort(
        data,
        (size_t)total,
        sizeof(uint64_t),
        compare_keys
    );

    double finish = wall_time();

    // ========================================
    // STEP 7: Verify sorted ordering
    // ========================================

    int ordered = 1;

    for (uint64_t i = 1; i < total; i++) {

        if (data[i - 1] >= data[i]) {
            ordered = 0;
            break;
        }
    }

    // ========================================
    // STEP 8: Verify data preservation
    // ========================================

    uint64_t final_sum = 0;
    uint64_t final_xor = 0;

    for (uint64_t i = 0; i < total; i++) {

        uint64_t h =
            learning_fingerprint(data[i]);

        final_sum += h;
        final_xor ^= h;
    }

    int preserved =
        initial_sum == final_sum &&
        initial_xor == final_xor;

    int correct = ordered && preserved;

    // ========================================
    // STEP 9: Report the baseline
    // ========================================

    printf(
        "===== SERIAL SORTING BASELINE =====\n"
    );

    printf(
        "Distribution: %s\n",
        argv[2]
    );

    printf(
        "Total integers: %" PRIu64 "\n",
        total
    );

    printf(
        "Key size: %zu bytes\n",
        sizeof(uint64_t)
    );

    printf(
        "Serial sorting time: %.6f s\n",
        finish - start
    );

    printf(
        "Global fingerprint sum: 0x%016"
        PRIx64 "\n",
        final_sum
    );

    printf(
        "Global fingerprint xor: 0x%016"
        PRIx64 "\n",
        final_xor
    );

    printf(
        "Sorted ordering: %s\n",
        ordered ? "PASS" : "FAIL"
    );

    printf(
        "Fingerprint preservation: %s\n",
        preserved ? "PASS" : "FAIL"
    );

    printf(
        "Correctness: %s\n",
        correct ? "PASS" : "FAIL"
    );

    free(data);

    return correct
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
