
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define CACHE_LINE 64
#define PADDED_STRIDE 8
#define RUNS 3

static double run_trial(
    int threads,
    uint64_t iterations,
    int stride
) {
    size_t slots = (size_t)threads * stride;
    size_t bytes = slots * sizeof(uint64_t);

    // aligned_alloc requires a size divisible by
    // the requested alignment.
    size_t allocated =
        (bytes + CACHE_LINE - 1) /
        CACHE_LINE * CACHE_LINE;

    uint64_t *data =
        aligned_alloc(CACHE_LINE, allocated);

    if (!data) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    memset(data, 0, allocated);

    int actual_threads = 0;

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        {
            actual_threads = omp_get_num_threads();
        }

        // Each thread updates a DIFFERENT counter.
        // Volatile forces observable memory accesses
        // for this specific experiment.
        volatile uint64_t *counter =
            &data[(size_t)tid * stride];

        for (uint64_t i = 0; i < iterations; i++) {
            (*counter)++;
        }
    }

    double elapsed = omp_get_wtime() - start;

    if (actual_threads != threads) {
        fprintf(stderr,
                "Expected %d threads, got %d\n",
                threads, actual_threads);
        free(data);
        exit(1);
    }

    // Check that no increments were lost.
    for (int t = 0; t < threads; t++) {
        if (data[(size_t)t * stride] != iterations) {
            fprintf(stderr,
                    "Incorrect result for thread %d\n",
                    t);
            free(data);
            exit(1);
        }
    }

    free(data);
    return elapsed;
}

static double median3(double a, double b, double c) {
    double temp;

    if (a > b) {
        temp = a; a = b; b = temp;
    }

    if (b > c) {
        temp = b; b = c; c = temp;
    }

    if (a > b) {
        temp = a; a = b; b = temp;
    }

    return b;
}

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr,
                "Usage: %s threads iterations\n",
                argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);
    uint64_t iterations = strtoull(
        argv[2], NULL, 10
    );

    if (threads < 1 || threads > 64 ||
        iterations == 0) {
        fprintf(stderr, "Invalid arguments\n");
        return 1;
    }

    omp_set_dynamic(0);

    double adjacent[RUNS];
    double padded[RUNS];

    printf("===== FALSE SHARING EXPERIMENT =====\n");
    printf("Threads: %d\n", threads);
    printf("Iterations per thread: %llu\n",
           (unsigned long long)iterations);

    printf("\n");

    for (int r = 0; r < RUNS; r++) {

        // Stride 1: neighboring 8-byte counters
        adjacent[r] = run_trial(
            threads, iterations, 1
        );

        // Stride 8: 64 bytes between counters
        padded[r] = run_trial(
            threads, iterations, PADDED_STRIDE
        );

        printf(
            "Run %d | Adjacent: %.6f s "
            "| Padded: %.6f s\n",
            r + 1, adjacent[r], padded[r]
        );
    }

    double adjacent_median = median3(
        adjacent[0], adjacent[1], adjacent[2]
    );

    double padded_median = median3(
        padded[0], padded[1], padded[2]
    );

    printf("\n===== MEDIANS =====\n");
    printf("Adjacent counters: %.6f seconds\n",
           adjacent_median);

    printf("Padded counters: %.6f seconds\n",
           padded_median);

    printf("Adjacent/Padded ratio: %.3fx\n",
           adjacent_median / padded_median);

    printf("\nCorrectness: PASS\n");

    return 0;
}
