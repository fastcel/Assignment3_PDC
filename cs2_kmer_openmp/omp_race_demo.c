
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>

int main(void) {

    const long long L = 2000000;
    const int k = 8;

    const long long expected = L - k + 1;

    // All DNA bases are 'A', so every
    // 8-mer maps to histogram index 0.

    // Volatile does NOT make updates safe.
    volatile uint64_t unsafe_count = 0;

    uint64_t atomic_count = 0;

    omp_set_dynamic(0);

    printf("===== OPENMP RACE DEMONSTRATION =====\n");

    printf("DNA length: %lld\n", L);
    printf("k: %d\n", k);
    printf("Expected windows: %lld\n", expected);

    // -------------------------------------
    // VERSION A: Incorrect, unsynchronized
    // -------------------------------------

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < expected; i++) {

        // Multiple threads update
        // the same counter concurrently.

        unsafe_count++;
    }

    double unsafe_time = omp_get_wtime() - start;

    // -------------------------------------
    // VERSION B: Correct atomic increment
    // -------------------------------------

    start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < expected; i++) {

        #pragma omp atomic update
        atomic_count++;
    }

    double atomic_time = omp_get_wtime() - start;

    // -------------------------------------
    // RESULTS
    // -------------------------------------

    printf("\n===== RESULTS =====\n");

    printf("Expected: %lld\n", expected);

    printf("\nWithout atomic:\n");
    printf("Count: %" PRIu64 "\n",
           (uint64_t)unsafe_count);
    printf("Time: %.6f seconds\n", unsafe_time);

    printf("\nWith atomic:\n");
    printf("Count: %" PRIu64 "\n", atomic_count);
    printf("Time: %.6f seconds\n", atomic_time);

    printf("\nAtomic correctness: %s\n",
           atomic_count == (uint64_t)expected
           ? "PASS" : "FAIL");

    printf("\nNote: The unsafe result is undefined\n");
    printf("because multiple threads write to\n");
    printf("the same counter without protection.\n");

    return atomic_count == (uint64_t)expected ? 0 : 1;
}
