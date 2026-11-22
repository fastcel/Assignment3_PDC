
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <time.h>

static const char bases[] = "ACGT";

// Fixed-seed pseudorandom generator
static uint64_t next_random(uint64_t *state) {
    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return x * UINT64_C(2685821657736338717);
}

static double elapsed_seconds(
    const struct timespec *start,
    const struct timespec *end
) {
    return (end->tv_sec - start->tv_sec)
         + (end->tv_nsec - start->tv_nsec) / 1e9;
}

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr,
            "Usage: %s L k\n", argv[0]);
        return 1;
    }

    // Read sequence length safely
    errno = 0;
    char *endptr = NULL;

    unsigned long long requested =
        strtoull(argv[1], &endptr, 10);

    if (errno || *endptr != '\0' ||
        requested == 0 ||
        requested > SIZE_MAX) {
        fprintf(stderr, "Invalid sequence length\n");
        return 1;
    }

    size_t L = (size_t)requested;

    int k = atoi(argv[2]);

    if (k < 1 || k > 12 || L < (size_t)k) {
        fprintf(stderr,
            "Require 1 <= k <= 12 and L >= k\n");
        return 1;
    }

    size_t bins = (size_t)1 << (2 * k);

    // One byte per DNA base, encoded 0..3
    uint8_t *dna = malloc(L);

    // One 64-bit counter per possible k-mer
    uint64_t *hist = calloc(bins, sizeof(uint64_t));

    if (!dna || !hist) {
        fprintf(stderr, "Memory allocation failed\n");
        free(dna);
        free(hist);
        return 1;
    }

    // ---------------------------------------
    // STEP 1: Generate deterministic DNA
    // ---------------------------------------

    uint64_t state = UINT64_C(0x123456789ABCDEF);

    uint64_t base_counts[4] = {0};

    for (size_t i = 0; i < L; i++) {

        // Select one of four bases from high bits
        uint8_t base =
            (uint8_t)(next_random(&state) >> 62);

        dna[i] = base;
        base_counts[base]++;
    }

    printf("Sequence length: %zu\n", L);
    printf("k: %d\n", k);
    printf("Histogram bins: %zu\n", bins);

    printf("First DNA bases: ");

    size_t preview = L < 32 ? L : 32;

    for (size_t i = 0; i < preview; i++) {
        putchar(bases[dna[i]]);
    }

    printf("\n");

    // ---------------------------------------
    // STEP 2: Serial k-mer counting
    // ---------------------------------------

    uint32_t mask =
        (UINT32_C(1) << (2 * k)) - 1u;

    uint32_t code = 0;

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (size_t i = 0; i < L; i++) {

        code = ((code << 2) | dna[i]) & mask;

        if (i + 1 >= (size_t)k) {
            hist[code]++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double seconds = elapsed_seconds(
        &start, &end
    );

    // ---------------------------------------
    // STEP 3: Verify histogram
    // ---------------------------------------

    uint64_t total = 0;
    size_t nonzero_bins = 0;

    // Reproducible fingerprint for diagnostics
    uint64_t fingerprint =
        UINT64_C(14695981039346656037);

    for (size_t i = 0; i < bins; i++) {

        uint64_t count = hist[i];

        total += count;

        if (count != 0) {
            nonzero_bins++;
        }

        fingerprint ^= count ^
            ((uint64_t)i *
             UINT64_C(0x9E3779B97F4A7C15));

        fingerprint *=
            UINT64_C(1099511628211);
    }

    uint64_t expected =
        (uint64_t)(L - (size_t)k + 1);

    printf("\n===== RESULTS =====\n");

    printf("Execution Time: %.6f seconds\n",
           seconds);

    printf("Expected k-mers: %" PRIu64 "\n",
           expected);

    printf("Counted k-mers:  %" PRIu64 "\n",
           total);

    printf("Nonzero bins: %zu\n", nonzero_bins);

    printf("Histogram fingerprint: %016" PRIx64 "\n",
           fingerprint);

    printf("\nBase frequencies:\n");

    for (int i = 0; i < 4; i++) {
        printf("%c: %" PRIu64 "\n",
               bases[i], base_counts[i]);
    }

    double throughput =
        seconds > 0.0
        ? (double)expected / (seconds * 1e6)
        : 0.0;

    printf("\nThroughput: %.3f million k-mers/s\n",
           throughput);

    int passed = (total == expected);

    printf("Correctness: %s\n",
           passed ? "PASS" : "FAIL");

    free(dna);
    free(hist);

    return passed ? 0 : 1;
}
