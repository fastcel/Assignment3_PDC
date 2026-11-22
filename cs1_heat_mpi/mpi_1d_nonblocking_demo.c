
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Update one locally owned row.
void update_row(
    const double *current,
    double *next,
    int localRow,
    int globalRow,
    int N,
    double alpha,
    int checkChange,
    double *localMax
) {
    // The global boundary temperatures remain fixed.
    if (globalRow == 0 || globalRow == N - 1)
        return;

    for (int j = 1; j < N - 1; j++) {

        size_t idx = (size_t)localRow * N + j;

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
            double change = fabs(newValue - oldValue);

            if (change > *localMax)
                *localMax = change;
        }
    }
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, p;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int N = 5;
    int T = 2;
    int K = 1;

    const double alpha = 0.1;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) T = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    if (argc > 4 || N < 3 || T < 0 || K < 1 || p > N) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: mpirun -np P %s [N>=3] [T>=0] [K>=1]\n",
                argv[0]);
        }

        MPI_Finalize();
        return 1;
    }

    // Divide rows between processes.
    int base = N / p;
    int remainder = N % p;

    int localRows = base + (rank < remainder);

    int startRow = rank * base +
                   (rank < remainder ? rank : remainder);

    // Reserve two additional halo rows.
    size_t cells = (size_t)(localRows + 2) * N;

    double *gridA = calloc(cells, sizeof(double));
    double *gridB = calloc(cells, sizeof(double));

    if (!gridA || !gridB) {
        fprintf(stderr, "Rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *current = gridA;
    double *next = gridB;

    int up = rank == 0 ? MPI_PROC_NULL : rank - 1;

    int down = rank == p - 1 ?
               MPI_PROC_NULL : rank + 1;

    // Initialize the hot centre cell.
    int centerRow = N / 2;
    int centerCol = N / 2;

    if (centerRow >= startRow &&
        centerRow < startRow + localRows) {

        int localCenter = centerRow - startRow + 1;

        current[(size_t)localCenter * N + centerCol] =
            100.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    double totalWaitTime = 0.0;
    double totalInteriorTime = 0.0;
    double globalChange = 0.0;

    for (int t = 0; t < T; t++) {

        MPI_Request requests[4];

        // Receive upper halo.
        MPI_Irecv(
            current,
            N, MPI_DOUBLE,
            up, 20,
            MPI_COMM_WORLD,
            &requests[0]
        );

        // Receive lower halo.
        MPI_Irecv(
            current + (size_t)(localRows + 1) * N,
            N, MPI_DOUBLE,
            down, 10,
            MPI_COMM_WORLD,
            &requests[1]
        );

        // Send first owned row upward.
        MPI_Isend(
            current + N,
            N, MPI_DOUBLE,
            up, 10,
            MPI_COMM_WORLD,
            &requests[2]
        );

        // Send last owned row downward.
        MPI_Isend(
            current + (size_t)localRows * N,
            N, MPI_DOUBLE,
            down, 20,
            MPI_COMM_WORLD,
            &requests[3]
        );

        int checkChange =
            ((t + 1) % K == 0 || t == T - 1);

        double localChange = 0.0;

        // Calculate rows that do not need halo data.
        double interiorStart = MPI_Wtime();

        for (int i = 2; i < localRows; i++) {

            int globalRow = startRow + i - 1;

            update_row(
                current, next,
                i, globalRow, N, alpha,
                checkChange, &localChange
            );
        }

        totalInteriorTime +=
            MPI_Wtime() - interiorStart;

        // Ensure all halo communication has completed.
        double waitStart = MPI_Wtime();

        MPI_Waitall(
            4, requests, MPI_STATUSES_IGNORE
        );

        totalWaitTime += MPI_Wtime() - waitStart;

        // Update first owned row.
        update_row(
            current, next,
            1, startRow, N, alpha,
            checkChange, &localChange
        );

        // Update last owned row, if different.
        if (localRows > 1) {
            update_row(
                current, next,
                localRows,
                startRow + localRows - 1,
                N, alpha,
                checkChange, &localChange
            );
        }

        // Global convergence check.
        if (checkChange) {
            MPI_Allreduce(
                &localChange, &globalChange,
                1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD
            );
        }

        // Swap grids.
        double *temp = current;
        current = next;
        next = temp;
    }

    double localTime = MPI_Wtime() - startTime;

    double elapsed = 0.0;
    double maxWaitTime = 0.0;
    double maxInteriorTime = 0.0;

    MPI_Reduce(
        &localTime, &elapsed,
        1, MPI_DOUBLE, MPI_MAX,
        0, MPI_COMM_WORLD
    );

    MPI_Reduce(
        &totalWaitTime, &maxWaitTime,
        1, MPI_DOUBLE, MPI_MAX,
        0, MPI_COMM_WORLD
    );

    MPI_Reduce(
        &totalInteriorTime, &maxInteriorTime,
        1, MPI_DOUBLE, MPI_MAX,
        0, MPI_COMM_WORLD
    );

    // Correctness statistics.
    double localSum = 0.0;
    double localSquares = 0.0;
    double localCenterValue = 0.0;

    for (int i = 1; i <= localRows; i++) {
        for (int j = 0; j < N; j++) {

            double value =
                current[(size_t)i * N + j];

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

    MPI_Reduce(
        &localSum, &checksum,
        1, MPI_DOUBLE, MPI_SUM,
        0, MPI_COMM_WORLD
    );

    MPI_Reduce(
        &localSquares, &sumSquares,
        1, MPI_DOUBLE, MPI_SUM,
        0, MPI_COMM_WORLD
    );

    MPI_Reduce(
        &localCenterValue, &centerValue,
        1, MPI_DOUBLE, MPI_SUM,
        0, MPI_COMM_WORLD
    );

    // Gather complete grid for small tests.
    if (N <= 8) {

        double *fullGrid = NULL;
        int *counts = NULL;
        int *displs = NULL;

        if (rank == 0) {

            fullGrid = malloc(
                (size_t)N * N * sizeof(double)
            );

            counts = malloc(p * sizeof(int));
            displs = malloc(p * sizeof(int));

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
            current + N,
            localRows * N,
            MPI_DOUBLE,
            fullGrid,
            counts,
            displs,
            MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        if (rank == 0) {
            printf("\nFinal Temperature Grid:\n");

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    printf("%6.1f",
                           fullGrid[(size_t)i * N + j]);
                }

                printf("\n");
            }
        }

        free(fullGrid);
        free(counts);
        free(displs);
    }

    if (rank == 0) {

        printf("\nHeat Diffusion (MPI 1D Non-Blocking)\n");

        printf("Grid Size: %d x %d\n", N, N);
        printf("Time Steps: %d\n", T);
        printf("MPI Processes: %d\n", p);
        printf("Convergence K: %d\n", K);

        printf("\n--- Results ---\n");

        printf("Execution Time: %.6f seconds\n",
               elapsed);

        printf("Max Waitall Time: %.6f seconds\n",
               maxWaitTime);

        printf("Max Interior Compute Time: %.6f seconds\n",
               maxInteriorTime);

        printf("Checksum: %.10f\n", checksum);
        printf("Sum of Squares: %.10f\n", sumSquares);
        printf("Centre Temperature: %.10f\n",
               centerValue);
        printf("Last Maximum Change: %.10f\n",
               globalChange);
    }

    free(gridA);
    free(gridB);

    MPI_Finalize();
    return 0;
}
