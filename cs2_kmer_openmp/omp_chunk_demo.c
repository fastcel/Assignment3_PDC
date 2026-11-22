
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define K 3
#define MAX_WINDOWS 128
#define BINS 64

// Encode one DNA base using two bits
static uint32_t encode_base(char base) {
    switch (base) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return 0;
    }
}

// Encode a complete k-mer at position start
static uint32_t encode_window(
    const char *dna,
    int start,
    int k
) {
    uint32_t code = 0;

    for (int j = 0; j < k; j++) {
        code = (code << 2)
             | encode_base(dna[start + j]);
    }

    return code;
}

int main(void) {

    const char *dna = "ACGTACGTACGTACGTACGT";

    int L = (int)strlen(dna);
    int windows = L - K + 1;

    if (windows > MAX_WINDOWS) {
        printf("Too many windows\n");
        return 1;
    }

    uint32_t parallel_codes[MAX_WINDOWS] = {0};
    int owners[MAX_WINDOWS] = {0};

    uint64_t serial_hist[BINS] = {0};
    uint64_t parallel_hist[BINS] = {0};

    omp_set_dynamic(0);

    printf("DNA: %s\n", dna);
    printf("Length: %d\n", L);
    printf("k: %d\n", K);
    printf("Expected windows: %d\n", windows);
    printf("Requested threads: %d\n\n",
           omp_get_max_threads());

    // -------------------------------------
    // STEP 1: Serial reference
    // -------------------------------------

    for (int i = 0; i < windows; i++) {

        uint32_t code = encode_window(
            dna, i, K
        );

        serial_hist[code]++;
    }

    // -------------------------------------
    // STEP 2: Parallelize window positions
    // -------------------------------------

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < windows; i++) {

        // Each iteration owns a unique i.
        // Different threads may read
        // overlapping DNA characters safely.

        uint32_t code = encode_window(
            dna, i, K
        );

        parallel_codes[i] = code;
        owners[i] = omp_get_thread_num();
    }

    // -------------------------------------
    // STEP 3: Build histogram after joining
    // -------------------------------------

    for (int i = 0; i < windows; i++) {
        parallel_hist[parallel_codes[i]]++;
    }

    // -------------------------------------
    // STEP 4: Print ownership
    // -------------------------------------

    printf("===== WINDOW OWNERSHIP =====\n");

    for (int i = 0; i < windows; i++) {

        printf(
            "Thread %d | Start %2d | %.*s | Code %u\n",
            owners[i],
            i,
            K,
            dna + i,
            parallel_codes[i]
        );
    }

    // -------------------------------------
    // STEP 5: Exact histogram verification
    // -------------------------------------

    uint64_t serial_total = 0;
    uint64_t parallel_total = 0;

    int mismatched_bins = 0;

    for (int i = 0; i < BINS; i++) {

        serial_total += serial_hist[i];
        parallel_total += parallel_hist[i];

        if (serial_hist[i] != parallel_hist[i]) {
            mismatched_bins++;
        }
    }

    printf("\n===== VERIFICATION =====\n");

    printf("Serial total: %llu\n",
           (unsigned long long)serial_total);

    printf("Parallel total: %llu\n",
           (unsigned long long)parallel_total);

    printf("Expected total: %d\n", windows);

    printf("Mismatched bins: %d\n",
           mismatched_bins);

    int passed =
        serial_total == (uint64_t)windows &&
        parallel_total == (uint64_t)windows &&
        mismatched_bins == 0;

    printf("Correctness: %s\n",
           passed ? "PASS" : "FAIL");

    return passed ? 0 : 1;
}
