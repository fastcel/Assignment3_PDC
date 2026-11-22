
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IDX(i, j) ((size_t)(i) * N + (j))

int main(int argc, char *argv[]) {

    // Default parameters
    int N = 5;
    int T = 2;
    double alpha = 0.1;

    // Read command-line arguments
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (argc > 2) {
        T = atoi(argv[2]);
    }

    if (argc > 3 || N < 3 || T < 0) {
        fprintf(stderr, "Usage: %s [N>=3] [T>=0]\n", argv[0]);
        return 1;
    }

    printf("Heat Diffusion Simulation (Serial)\n");
    printf("Grid Size: %d x %d\n", N, N);
    printf("Time Steps: %d\n", T);
    printf("Alpha: %.2f\n", alpha);

    // Allocate two grids
    size_t cells = (size_t)N * N;

    double *gridA = calloc(cells, sizeof(double));
    double *gridB = calloc(cells, sizeof(double));

    if (gridA == NULL || gridB == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        free(gridA);
        free(gridB);
        return 1;
    }

    double *current = gridA;
    double *next = gridB;

    // Initialize the centre as a hot cell
    current[IDX(N / 2, N / 2)] = 100.0;

    // Display initial grid for small inputs
    if (N <= 8) {
        printf("\nInitial Temperature Grid:\n");

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%6.1f", current[IDX(i, j)]);
            }
            printf("\n");
        }
    }

    // Start timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Main simulation
    for (int t = 0; t < T; t++) {

        // Update interior cells
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {

                double up = current[IDX(i - 1, j)];
                double down = current[IDX(i + 1, j)];
                double left = current[IDX(i, j - 1)];
                double right = current[IDX(i, j + 1)];

                next[IDX(i, j)] = current[IDX(i, j)]
                    + alpha * (
                        up + down + left + right
                        - 4.0 * current[IDX(i, j)]
                    );
            }
        }

        // Swap grids
        double *temp = current;
        current = next;
        next = temp;
    }

    // Stop timing
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec)
        + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Calculate checksums for correctness verification
    double checksum = 0.0;
    double sum_squares = 0.0;

    for (size_t i = 0; i < cells; i++) {
        checksum += current[i];
        sum_squares += current[i] * current[i];
    }

    // Print final grid for small inputs
    if (N <= 8) {
        printf("\nTemperature After %d Steps:\n", T);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%6.1f", current[IDX(i, j)]);
            }
            printf("\n");
        }
    }

    // Final results
    printf("\n--- Results ---\n");
    printf("Execution Time: %.6f seconds\n", elapsed);
    printf("Checksum: %.10f\n", checksum);
    printf("Sum of Squares: %.10f\n", sum_squares);
    printf("Centre Temperature: %.10f\n",
           current[IDX(N / 2, N / 2)]);

    // Release memory
    free(gridA);
    free(gridB);

    return 0;
}
