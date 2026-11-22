
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <omp.h>

// ========================================
// STEP 1: Fixed-seed DNA generator
// ========================================

static uint64_t next_random(uint64_t *state) {
    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return x * UINT64_C(2685821657736338717);
}

// ========================================
// STEP 2: Serial reference
// ========================================

static void count_serial(
    const uint8_t *dna,
    size_t L,
    int k,
    uint32_t mask,
    uint64_t *hist
) {
    uint32_t code = 0;

    for (size_t i = 0; i < L; i++) {

        code = ((code << 2) | dna[i]) & mask;

        if (i + 1 >= (size_t)k)
            hist[code]++;
    }
}

// ========================================
// STEP 3: OpenMP atomic histogram
// ========================================

static void count_atomic(
    const uint8_t *dna,
    size_t windows,
    int k,
    uint32_t mask,
    uint64_t *hist,
    int threads
) {
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        size_t begin =
            windows * (size_t)tid / (size_t)nthreads;

        size_t end =
            windows * (size_t)(tid + 1) / (size_t)nthreads;

        if (begin < end) {

            uint32_t code = 0;

            // Encode the first window in this chunk.
            for (int j = 0; j < k; j++) {
                code = (code << 2)
                     | dna[begin + (size_t)j];
            }

            for (size_t s = begin; s < end; s++) {

                // Slide the window one base forward.
                if (s > begin) {
                    code = (
                        (code << 2)
                        | dna[s + (size_t)k - 1]
                    ) & mask;
                }

                // Protect the shared histogram.
                #pragma omp atomic update
                hist[code]++;
            }
        }
    }
}

// ========================================
// STEP 4: Per-thread private histograms
// ========================================

static void count_private(
    const uint8_t *dna,
    size_t windows,
    int k,
    uint32_t mask,
    size_t bins,
    uint64_t *private_hist,
    uint64_t *merged,
    int threads
) {
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        size_t begin =
            windows * (size_t)tid / (size_t)nthreads;

        size_t end =
            windows * (size_t)(tid + 1) / (size_t)nthreads;

        // Each thread owns its own histogram.
        uint64_t *local =
            private_hist + (size_t)tid * bins;

        if (begin < end) {

            uint32_t code = 0;

            // Initialize the first k-mer.
            for (int j = 0; j < k; j++) {
                code = (code << 2)
                     | dna[begin + (size_t)j];
            }

            for (size_t s = begin; s < end; s++) {

                if (s > begin) {
                    code = (
                        (code << 2)
                        | dna[s + (size_t)k - 1]
                    ) & mask;
                }

                // No atomic needed: private memory.
                local[code]++;
            }
        }
    }

    // Merge all private histograms.
    // This is included in the private method's time.
    for (int t = 0; t < threads; t++) {

        const uint64_t *local =
            private_hist + (size_t)t * bins;

        for (size_t b = 0; b < bins; b++) {
            merged[b] += local[b];
        }
    }
}

// ========================================
// STEP 5: Histogram fingerprint
// ========================================

static uint64_t fingerprint(
    const uint64_t *hist,
    size_t bins
) {
    uint64_t hash =
        UINT64_C(14695981039346656037);

    for (size_t b = 0; b < bins; b++) {

        hash ^= hist[b] ^
            ((uint64_t)b *
             UINT64_C(0x9E3779B97F4A7C15));

        hash *= UINT64_C(1099511628211);
    }

    return hash;
}

// ========================================
// STEP 6: Main experiment
// ========================================

int main(int argc, char **argv) {

    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s L k threads\n", argv[0]);
        return 1;
    }

    errno = 0;
    char *endptr = NULL;

    unsigned long long requested =
        strtoull(argv[1], &endptr, 10);

    int k = atoi(argv[2]);
    int threads = atoi(argv[3]);

    if (errno || endptr == argv[1] ||
        *endptr != '\0' ||
        requested > SIZE_MAX ||
        requested < 12 ||
        (k != 8 && k != 12) ||
        threads < 1 || threads > 64) {

        fprintf(stderr,
            "Require L >= 12, k = 8 or 12, "
            "and 1 <= threads <= 64\n");

        return 1;
    }

    size_t L = (size_t)requested;

    size_t bins = (size_t)1 << (2 * k);
    size_t windows = L - (size_t)k + 1;

    uint32_t mask =
        (UINT32_C(1) << (2 * k)) - 1u;

    if (bins > SIZE_MAX /
        (size_t)threads / sizeof(uint64_t)) {

        fprintf(stderr,
            "Private histogram allocation too large\n");

        return 1;
    }

    // ====================================
    // STEP 7: Allocate the data
    // ====================================

    uint8_t *dna = malloc(L);

    uint64_t *reference =
        calloc(bins, sizeof(uint64_t));

    uint64_t *atomic_hist =
        calloc(bins, sizeof(uint64_t));

    uint64_t *merged =
        calloc(bins, sizeof(uint64_t));

    uint64_t *private_hist =
        calloc(bins * (size_t)threads,
               sizeof(uint64_t));

    if (!dna || !reference || !atomic_hist ||
        !merged || !private_hist) {

        fprintf(stderr, "Memory allocation failed\n");

        free(dna);
        free(reference);
        free(atomic_hist);
        free(merged);
        free(private_hist);

        return 1;
    }

    // ====================================
    // STEP 8: Generate deterministic DNA
    // ====================================

    uint64_t state =
        UINT64_C(0x123456789ABCDEF);

    for (size_t i = 0; i < L; i++) {
        dna[i] =
            (uint8_t)(next_random(&state) >> 62);
    }

    omp_set_dynamic(0);

    // ====================================
    // STEP 9: Measure serial counting
    // ====================================

    double start = omp_get_wtime();

    count_serial(
        dna, L, k, mask, reference
    );

    double serial_time =
        omp_get_wtime() - start;

    // ====================================
    // STEP 10: Measure atomic counting
    // ====================================

    start = omp_get_wtime();

    count_atomic(
        dna, windows, k, mask,
        atomic_hist, threads
    );

    double atomic_time =
        omp_get_wtime() - start;

    // ====================================
    // STEP 11: Measure private histograms
    // ====================================

    start = omp_get_wtime();

    count_private(
        dna, windows, k, mask,
        bins, private_hist, merged,
        threads
    );

    double private_time =
        omp_get_wtime() - start;

    // ====================================
    // STEP 12: Exact correctness checking
    // ====================================

    size_t atomic_errors = 0;
    size_t private_errors = 0;

    uint64_t serial_total = 0;
    uint64_t atomic_total = 0;
    uint64_t private_total = 0;

    for (size_t b = 0; b < bins; b++) {

        serial_total += reference[b];
        atomic_total += atomic_hist[b];
        private_total += merged[b];

        if (reference[b] != atomic_hist[b])
            atomic_errors++;

        if (reference[b] != merged[b])
            private_errors++;
    }

    // ====================================
    // STEP 13: Display the results
    // ====================================

    printf("\n===== FULL-SIZED K-MER EXPERIMENT =====\n");

    printf("Sequence length: %zu\n", L);
    printf("k: %d\n", k);
    printf("Threads: %d\n", threads);
    printf("Expected windows: %zu\n", windows);
    printf("Histogram bins: %zu\n", bins);

    printf("Private histogram memory: %.2f MiB\n",
        (double)(
            bins * (size_t)threads *
            sizeof(uint64_t)
        ) / (1024.0 * 1024.0));

    printf("\n===== EXECUTION TIMES =====\n");

    printf("Serial: %.6f seconds\n", serial_time);
    printf("Atomic: %.6f seconds\n", atomic_time);
    printf("Private: %.6f seconds\n", private_time);

    printf("\n===== HISTOGRAM VERIFICATION =====\n");

    printf("Serial total: %" PRIu64 "\n",
           serial_total);

    printf("Atomic total: %" PRIu64 "\n",
           atomic_total);

    printf("Private total: %" PRIu64 "\n",
           private_total);

    printf("Atomic mismatched bins: %zu\n",
           atomic_errors);

    printf("Private mismatched bins: %zu\n",
           private_errors);

    printf("Serial fingerprint: %016" PRIx64 "\n",
           fingerprint(reference, bins));

    printf("\n===== SPEEDUP =====\n");

    printf("Atomic speedup: %.3fx\n",
           serial_time / atomic_time);

    printf("Private speedup: %.3fx\n",
           serial_time / private_time);

    int passed =
        serial_total == (uint64_t)windows &&
        atomic_total == serial_total &&
        private_total == serial_total &&
        atomic_errors == 0 &&
        private_errors == 0;

    printf("\nExact bin-by-bin correctness: %s\n",
           passed ? "PASS" : "FAIL");

    free(dna);
    free(reference);
    free(atomic_hist);
    free(merged);
    free(private_hist);

    return passed ? 0 : 1;
}
