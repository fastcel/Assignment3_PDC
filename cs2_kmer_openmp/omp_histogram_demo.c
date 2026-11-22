
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <omp.h>

#define K 3
#define BINS (1 << (2 * K))

// Encode a single DNA base
static uint32_t encode_base(char base) {
    switch (base) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default:
            fprintf(stderr, "Invalid DNA base\n");
            exit(1);
    }
}

// Encode a complete k-mer
static uint32_t encode_window(
    const char *dna,
    int start
) {
    uint32_t code = 0;

    for (int j = 0; j < K; j++) {
        code = (code << 2)
             | encode_base(dna[start + j]);
    }

    return code;
}

// Compare every histogram bin
static int compare_histograms(
    const uint64_t *reference,
    const uint64_t *test
) {
    int mismatches = 0;

    for (int i = 0; i < BINS; i++) {
        if (reference[i] != test[i]) {
            mismatches++;
        }
    }

    return mismatches;
}

// Sum all histogram counts
static uint64_t histogram_total(
    const uint64_t *hist
) {
    uint64_t total = 0;

    for (int i = 0; i < BINS; i++) {
        total += hist[i];
    }

    return total;
}

int main(void) {

    const char *dna = "ACGTACGTACGTACGTACGT";

    int L = (int)strlen(dna);
    int windows = L - K + 1;

    uint64_t serial_hist[BINS] = {0};
    uint64_t atomic_hist[BINS] = {0};
    uint64_t merged_hist[BINS] = {0};

    omp_set_dynamic(0);

    int threads = omp_get_max_threads();

    // Allocate one histogram per thread
    uint64_t *private_hist = calloc(
        (size_t)threads * BINS,
        sizeof(uint64_t)
    );

    if (!private_hist) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    printf("===== K-MER HISTOGRAM DEMO =====\n");
    printf("DNA: %s\n", dna);
    printf("k: %d\n", K);
    printf("Windows: %d\n", windows);
    printf("Threads: %d\n", threads);

    // ==================================
    // METHOD 1: SERIAL REFERENCE
    // ==================================

    for (int i = 0; i < windows; i++) {

        uint32_t code = encode_window(dna, i);

        serial_hist[code]++;
    }

    // ==================================
    // METHOD 2: OPENMP ATOMIC
    // ==================================

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < windows; i++) {

        uint32_t code = encode_window(dna, i);

        // Protect shared histogram update
        #pragma omp atomic update
        atomic_hist[code]++;
    }

    // ==================================
    // METHOD 3: PRIVATE HISTOGRAMS
    // ==================================

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < windows; i++) {

        int tid = omp_get_thread_num();

        uint32_t code = encode_window(dna, i);

        // Each thread owns a separate row
        size_t index =
            (size_t)tid * BINS + code;

        private_hist[index]++;
    }

    // Merge thread-private histograms
    for (int t = 0; t < threads; t++) {

        for (int bin = 0; bin < BINS; bin++) {

            merged_hist[bin] +=
                private_hist[(size_t)t * BINS + bin];
        }
    }

    // ==================================
    // VERIFICATION
    // ==================================

    uint64_t serial_total =
        histogram_total(serial_hist);

    uint64_t atomic_total =
        histogram_total(atomic_hist);

    uint64_t private_total =
        histogram_total(merged_hist);

    int atomic_errors =
        compare_histograms(
            serial_hist, atomic_hist
        );

    int private_errors =
        compare_histograms(
            serial_hist, merged_hist
        );

    printf("\n===== RESULTS =====\n");

    printf("Expected total: %d\n", windows);

    printf("Serial total: %" PRIu64 "\n",
           serial_total);

    printf("Atomic total: %" PRIu64 "\n",
           atomic_total);

    printf("Private total: %" PRIu64 "\n",
           private_total);

    printf("Atomic mismatched bins: %d\n",
           atomic_errors);

    printf("Private mismatched bins: %d\n",
           private_errors);

    int passed =
        serial_total == (uint64_t)windows &&
        atomic_total == serial_total &&
        private_total == serial_total &&
        atomic_errors == 0 &&
        private_errors == 0;

    printf("\nCorrectness: %s\n",
           passed ? "PASS" : "FAIL");

    free(private_hist);

    return passed ? 0 : 1;
}
