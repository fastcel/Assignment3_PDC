
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {

    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s N T padding\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    int padding = atoi(argv[3]);

    if (N < 3 || N > 10000 ||
        T < 0 || (padding != 0 && padding != 1)) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    const double alpha = 0.1;

    // Physical row width, including optional padding
    size_t pitch = (size_t)N + padding;
    size_t cells = (size_t)N * pitch;

    double *gridA = calloc(cells, sizeof(double));
    double *gridB = calloc(cells, sizeof(double));

    if (!gridA || !gridB) {
        fprintf(stderr, "Memory allocation failed\n");
        free(gridA);
        free(gridB);
        return 1;
    }

    double *current = gridA;
    double *next = gridB;

    // Initialize central hot cell
    current[(size_t)(N / 2) * pitch + N / 2] = 100.0;

    printf("Grid Size: %d x %d\n", N, N);
    printf("Time Steps: %d\n", T);
    printf("Padding: %d\n", padding);
    printf("Physical Row Width: %zu\n", pitch);

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Heat diffusion simulation
    for (int t = 0; t < T; t++) {

        for (int i = 1; i < N - 1; i++) {

            size_t row = (size_t)i * pitch;

            for (int j = 1; j < N - 1; j++) {

                size_t idx = row + j;

                double old = current[idx];

                next[idx] = old + alpha * (
                    current[idx - pitch] +
                    current[idx + pitch] +
                    current[idx - 1] +
                    current[idx + 1] -
                    4.0 * old
                );
            }
        }

        // Swap grids
        double *temp = current;
        current = next;
        next = temp;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec)
        + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Validation statistics
    double checksum = 0.0;
    double sum_squares = 0.0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            double value =
                current[(size_t)i * pitch + j];

            checksum += value;
            sum_squares += value * value;
        }
    }

    double center =
        current[(size_t)(N / 2) * pitch + N / 2];

    printf("\n--- Results ---\n");
    printf("Execution Time: %.6f seconds\n", elapsed);
    printf("Checksum: %.10f\n", checksum);
    printf("Sum of Squares: %.10f\n", sum_squares);
    printf("Centre Temperature: %.10f\n", center);

    free(gridA);
    free(gridB);

    return 0;
}
