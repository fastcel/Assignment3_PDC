#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#define N 24000000
#define REPEATS 5
#define ROUNDS 3

static double median3(double a, double b, double c) {
    if (a > b) {
        double t = a; a = b; b = t;
    }
    if (b > c) {
        double t = b; b = c; c = t;
    }
    if (a > b) {
        double t = a; a = b; b = t;
    }
    return b;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s THREADS\n", argv[0]);
        return 1;
    }

    char *end;
    long requested = strtol(argv[1], &end, 10);

    if (end == argv[1] || *end != '\0' ||
        requested < 1 || requested > 12) {
        fprintf(stderr, "Invalid thread count\n");
        return 1;
    }

    int threads = (int)requested;
    omp_set_dynamic(0);

    size_t n = N;
    size_t bytes = n * sizeof(double);

    double *a = malloc(bytes);
    double *b = malloc(bytes);
    double *c = malloc(bytes);

    if (!a || !b || !c) {
        fprintf(stderr, "Memory allocation failed\n");
        free(a); free(b); free(c);
        return 1;
    }

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < n; i++) {
        a[i] = 0.0;
        b[i] = 1.0;
        c[i] = 2.0;
    }

    double times[ROUNDS];

    for (int round = 0; round < ROUNDS; round++) {

        // Warm-up pass, excluded from timing.
        #pragma omp parallel for num_threads(threads)
        for (size_t i = 0; i < n; i++)
            a[i] = b[i] + 3.0 * c[i];

        double started = omp_get_wtime();

        for (int repetition = 0;
             repetition < REPEATS;
             repetition++) {

            double scalar = 3.0 + repetition * 0.01;

            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t i = 0; i < n; i++) {
                a[i] = b[i] + scalar * c[i];
            }
        }

        times[round] = omp_get_wtime() - started;
    }

    double elapsed = median3(
        times[0], times[1], times[2]
    );

    // STREAM-like triad:
    // Read b, read c, write a.
    // Count 24 logical bytes per element.

    double transferred = (double)n * 24.0 * REPEATS;
    double bandwidth = transferred / elapsed / 1e9;

    // Prevent the result from being entirely unused.
    double checksum = a[n / 2];

    printf("===== STREAM-LIKE MEMORY BENCHMARK =====\n");
    printf("Threads: %d\n", threads);
    printf("Elements per array: %zu\n", n);
    printf("Working set: %.1f MB\n",
           3.0 * bytes / 1e6);
    printf("Median elapsed time: %.6f s\n", elapsed);
    printf("Estimated throughput: %.3f GB/s\n",
           bandwidth);
    printf("Checksum: %.6f\n", checksum);
    printf("Correctness: %s\n",
           checksum > 6.9 && checksum < 7.1
               ? "PASS" : "FAIL");

    free(a);
    free(b);
    free(c);

    return 0;
}
