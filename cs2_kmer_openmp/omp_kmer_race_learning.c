
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>

// Convert a DNA window into a 2-bit index.
// DNA is already encoded: A=0, C=1, G=2, T=3.
static uint32_t encode_window(
    const uint8_t *dna,
    size_t start,
    int k
) {
    uint32_t code = 0;

    for (int j = 0; j < k; j++) {
        code = (code << 2)
             | (uint32_t)dna[start + (size_t)j];
    }

    return code;
}

int main(int argc, char **argv) {

    if (argc != 4) {
        fprintf(stderr,
                "Usage: %s L k threads\n",
                argv[0]);
        return 1;
    }

    size_t L = (size_t)strtoull(argv[1], NULL, 10);
    int k = atoi(argv[2]);
    int threads = atoi(argv[3]);

    if ((k != 8 && k != 12) ||
        L < (size_t)k ||
        threads < 1 || threads > 64) {

        fprintf(stderr, "Invalid arguments\n");
        return 1;
    }

    size_t windows = L - (size_t)k + 1;
    size_t bins = (size_t)1 << (2 * k);

    uint8_t *dna = malloc(L);

    uint64_t *serial_hist =
        calloc(bins, sizeof(uint64_t));

    uint64_t *unsafe_hist =
        calloc(bins, sizeof(uint64_t));

    uint64_t *atomic_hist =
        calloc(bins, sizeof(uint64_t));

    if (!dna || !serial_hist ||
        !unsafe_hist || !atomic_hist) {

        fprintf(stderr, "Memory allocation failed\n");
        free(dna);
        free(serial_hist);
        free(unsafe_hist);
        free(atomic_hist);
        return 1;
    }

    // Generate a skewed DNA sequence.
    // Mostly A, with one C every 127 positions.
    for (size_t i = 0; i < L; i++) {
        dna[i] = (i % 127 == 0) ? 1 : 0;
    }

    omp_set_dynamic(0);

    // =====================================
    // SERIAL REFERENCE
    // =====================================

    for (size_t s = 0; s < windows; s++) {
        uint32_t code = encode_window(dna, s, k);
        serial_hist[code]++;
    }

    // =====================================
    // VERSION A: INTENTIONALLY UNSAFE
    // =====================================

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t s = 0; s < windows; s++) {

        uint32_t code = encode_window(dna, s, k);

        // DATA RACE: Multiple threads may
        // modify the same histogram bin.
        unsafe_hist[code]++;
    }

    double unsafe_time = omp_get_wtime() - start;

    // =====================================
    // VERSION B: ATOMIC
    // =====================================

    start = omp_get_wtime();

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t s = 0; s < windows; s++) {

        uint32_t code = encode_window(dna, s, k);

        #pragma omp atomic update
        atomic_hist[code]++;
    }

    double atomic_time = omp_get_wtime() - start;

    // =====================================
    // EXACT HISTOGRAM VERIFICATION
    // =====================================

    uint64_t serial_total = 0;
    uint64_t unsafe_total = 0;
    uint64_t atomic_total = 0;

    size_t unsafe_mismatches = 0;
    size_t atomic_mismatches = 0;

    for (size_t b = 0; b < bins; b++) {

        serial_total += serial_hist[b];
        unsafe_total += unsafe_hist[b];
        atomic_total += atomic_hist[b];

        if (serial_hist[b] != unsafe_hist[b])
            unsafe_mismatches++;

        if (serial_hist[b] != atomic_hist[b])
            atomic_mismatches++;
    }

    printf("\n===== REAL K-MER RACE TEST =====\n");

    printf("DNA length: %zu\n", L);
    printf("k: %d\n", k);
    printf("Threads: %d\n", threads);
    printf("Expected k-mers: %zu\n", windows);

    printf("\n===== HISTOGRAM TOTALS =====\n");

    printf("Serial: %" PRIu64 "\n", serial_total);
    printf("Unsafe: %" PRIu64 "\n", unsafe_total);
    printf("Atomic: %" PRIu64 "\n", atomic_total);

    printf("\n===== BIN COMPARISONS =====\n");

    printf("Unsafe mismatched bins: %zu\n",
           unsafe_mismatches);

    printf("Atomic mismatched bins: %zu\n",
           atomic_mismatches);

    // All-A k-mers map to histogram index 0.
    printf("\nAll-A bin (index 0):\n");

    printf("Serial: %" PRIu64 "\n",
           serial_hist[0]);

    printf("Unsafe: %" PRIu64 "\n",
           unsafe_hist[0]);

    printf("Atomic: %" PRIu64 "\n",
           atomic_hist[0]);

    printf("\n===== EXECUTION TIMES =====\n");

    printf("Unsafe: %.6f seconds\n", unsafe_time);
    printf("Atomic: %.6f seconds\n", atomic_time);

    printf("\nAtomic correctness: %s\n",
           atomic_mismatches == 0 &&
           atomic_total == (uint64_t)windows
           ? "PASS" : "FAIL");

    printf("\nNote: The unsafe version has a data race.\n");
    printf("Its behaviour is undefined, even if a\n");
    printf("particular run happens to appear correct.\n");

    free(dna);
    free(serial_hist);
    free(unsafe_hist);
    free(atomic_hist);

    return 0;
}
