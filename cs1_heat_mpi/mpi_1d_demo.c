
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int N = 5;
    int T = 2;
    int K = 50;
    const double alpha = 0.1;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) T = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    if (argc > 4 || N < 3 || T < 0 || K < 1 || p > N) {
        if (rank == 0)
            fprintf(stderr,
                "Usage: mpirun -np P %s [N>=3] [T>=0] [K>=1]\n",
                argv[0]);
        MPI_Finalize();
        return 1;
    }

    // Divide global rows between processes.
    int base = N / p;
    int remainder = N % p;

    int localRows = base + (rank < remainder);
    int startRow = rank * base +
                   (rank < remainder ? rank : remainder);

    // Two extra rows are reserved for halos.
    size_t cells = (size_t)(localRows + 2) * N;

    double *gridA = calloc(cells, sizeof(double));
    double *gridB = calloc(cells, sizeof(double));

    if (!gridA || !gridB) {
        fprintf(stderr, "Rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *current = gridA;
    double *next = gridB;

    // Identify neighbouring MPI processes.
    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == p - 1) ? MPI_PROC_NULL : rank + 1;

    // Initialize the hot centre cell.
    int centerRow = N / 2;
    int centerCol = N / 2;

    if (centerRow >= startRow &&
        centerRow < startRow + localRows) {

        int localCenter = centerRow - startRow + 1;

        current[(size_t)localCenter * N + centerCol] = 100.0;
    }

    // Synchronize before starting the timer.
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    double globalChange = 0.0;

    for (int t = 0; t < T; t++) {

        // Exchange first owned row with the upper rank.
        // Receive the lower rank's first row.
        MPI_Sendrecv(
            current + N, N, MPI_DOUBLE,
            up, 10,
            current + (size_t)(localRows + 1) * N,
            N, MPI_DOUBLE, down, 10,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        // Exchange last owned row with the lower rank.
        // Receive the upper rank's last row.
        MPI_Sendrecv(
            current + (size_t)localRows * N,
            N, MPI_DOUBLE, down, 20,
            current,
            N, MPI_DOUBLE, up, 20,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        int checkChange = ((t + 1) % K == 0 ||
                           t == T - 1);

        double localChange = 0.0;

        // Update the locally owned interior cells.
        for (int i = 1; i <= localRows; i++) {

            int globalRow = startRow + i - 1;

            // Global top/bottom boundaries are fixed.
            if (globalRow == 0 || globalRow == N - 1)
                continue;

            for (int j = 1; j < N - 1; j++) {

                size_t idx = (size_t)i * N + j;

                double oldValue = current[idx];

                double newValue = oldValue + alpha * (
                    current[idx - N] +
                    current[idx + N] +
                    current[idx - 1] +
                    current[idx + 1] -
                    4.0 * oldValue
                );

                next[idx] = newValue;

                if (checkChange) {
                    double difference = fabs(newValue - oldValue);

                    if (difference > localChange)
                        localChange = difference;
                }
            }
        }

        // Obtain global maximum change every K steps,
        // and also at the final step.
        if (checkChange) {
            MPI_Allreduce(
                &localChange, &globalChange,
                1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD
            );
        }

        // Swap the two grids.
        double *temp = current;
        current = next;
        next = temp;
    }

    // Record the slowest process's execution time.
    double localTime = MPI_Wtime() - startTime;
    double parallelTime = 0.0;

    MPI_Reduce(
        &localTime, &parallelTime,
        1, MPI_DOUBLE, MPI_MAX,
        0, MPI_COMM_WORLD
    );

    // Compute local validation statistics.
    double localSum = 0.0;
    double localSquares = 0.0;
    double localCenterValue = 0.0;

    for (int i = 1; i <= localRows; i++) {
        for (int j = 0; j < N; j++) {
            double value = current[(size_t)i * N + j];

            localSum += value;
            localSquares += value * value;
        }
    }

    if (centerRow >= startRow &&
        centerRow < startRow + localRows) {

        int localCenter = centerRow - startRow + 1;

        localCenterValue =
            current[(size_t)localCenter * N + centerCol];
    }

    double checksum = 0.0;
    double sumSquares = 0.0;
    double centerValue = 0.0;

    MPI_Reduce(&localSum, &checksum, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&localSquares, &sumSquares, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&localCenterValue, &centerValue,
               1, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);

    // Gather the complete grid only for small tests.
    if (N <= 8) {
        double *fullGrid = NULL;
        int *counts = NULL;
        int *displs = NULL;

        if (rank == 0) {
            fullGrid = malloc((size_t)N * N * sizeof(double));
            counts = malloc((size_t)p * sizeof(int));
            displs = malloc((size_t)p * sizeof(int));

            if (!fullGrid || !counts || !displs)
                MPI_Abort(MPI_COMM_WORLD, 1);

            for (int r = 0; r < p; r++) {
                int rows = base + (r < remainder);
                int first = r * base +
                            (r < remainder ? r : remainder);

                counts[r] = rows * N;
                displs[r] = first * N;
            }
        }

        MPI_Gatherv(
            current + N, localRows * N, MPI_DOUBLE,
            fullGrid, counts, displs, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        if (rank == 0) {
            printf("\nFinal Temperature Grid:\n");

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++)
                    printf("%6.1f", fullGrid[(size_t)i * N + j]);

                printf("\n");
            }
        }

        free(fullGrid);
        free(counts);
        free(displs);
    }

    if (rank == 0) {
        printf("\nHeat Diffusion (MPI 1D Blocking)\n");
        printf("Grid Size: %d x %d\n", N, N);
        printf("Time Steps: %d\n", T);
        printf("MPI Processes: %d\n", p);
        printf("Convergence K: %d\n", K);

        printf("\n--- Results ---\n");
        printf("Execution Time: %.6f seconds\n", parallelTime);
        printf("Checksum: %.10f\n", checksum);
        printf("Sum of Squares: %.10f\n", sumSquares);
        printf("Centre Temperature: %.10f\n", centerValue);
        printf("Last Maximum Change: %.10f\n", globalChange);
    }

    free(gridA);
    free(gridB);

    MPI_Finalize();
    return 0;
}
