
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Determine how many cells a process owns
int block_size(int N, int parts, int coord) {
    return N / parts + (coord < N % parts);
}

// Determine where a process's block begins
int block_start(int N, int parts, int coord) {
    return coord * (N / parts)
         + (coord < N % parts ? coord : N % parts);
}

// Index into a locally allocated grid with halos
#define IDX(i, j) ((size_t)(i) * pitch + (j))

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int N = 5;
    int T = 2;
    int K = 1;
    const double alpha = 0.1;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) T = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    if (argc > 4 || N < 3 || T < 0 || K < 1) {
        fprintf(stderr, "Invalid arguments\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create a 2D process grid
    int dims[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dims);

    if (dims[0] > N || dims[1] > N) {
        fprintf(stderr, "Too many MPI processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int periods[2] = {0, 0};
    MPI_Comm cart;

    MPI_Cart_create(
        MPI_COMM_WORLD, 2, dims,
        periods, 0, &cart
    );

    int rank;
    MPI_Comm_rank(cart, &rank);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);

    // Find neighbouring ranks
    int up, down, left, right;

    MPI_Cart_shift(cart, 0, 1, &up, &down);
    MPI_Cart_shift(cart, 1, 1, &left, &right);

    // Calculate our local rectangular block
    int rows = block_size(N, dims[0], coords[0]);
    int cols = block_size(N, dims[1], coords[1]);

    int rowStart = block_start(N, dims[0], coords[0]);
    int colStart = block_start(N, dims[1], coords[1]);

    // Allocate one halo layer on every side
    int pitch = cols + 2;

    size_t cells = (size_t)(rows + 2) * pitch;

    double *gridA = calloc(cells, sizeof(double));
    double *gridB = calloc(cells, sizeof(double));

    if (!gridA || !gridB) {
        fprintf(stderr, "Rank %d: allocation failed\n", rank);
        MPI_Abort(cart, 1);
    }

    double *current = gridA;
    double *next = gridB;

    // Initialize the global centre temperature
    int center = N / 2;

    if (center >= rowStart &&
        center < rowStart + rows &&
        center >= colStart &&
        center < colStart + cols) {

        int localRow = center - rowStart + 1;
        int localCol = center - colStart + 1;

        current[IDX(localRow, localCol)] = 100.0;
    }

    // A column is non-contiguous in row-major memory
    MPI_Datatype columnType;

    MPI_Type_vector(
        rows, 1, pitch,
        MPI_DOUBLE, &columnType
    );

    MPI_Type_commit(&columnType);

    MPI_Barrier(cart);
    double startTime = MPI_Wtime();

    double globalChange = 0.0;

    for (int t = 0; t < T; t++) {

        // Exchange upper and lower halo rows
        MPI_Sendrecv(
            current + IDX(1, 1), cols, MPI_DOUBLE,
            up, 100,
            current + IDX(rows + 1, 1),
            cols, MPI_DOUBLE,
            down, 100,
            cart, MPI_STATUS_IGNORE
        );

        MPI_Sendrecv(
            current + IDX(rows, 1),
            cols, MPI_DOUBLE,
            down, 101,
            current + IDX(0, 1),
            cols, MPI_DOUBLE,
            up, 101,
            cart, MPI_STATUS_IGNORE
        );

        // Exchange left and right halo columns
        MPI_Sendrecv(
            current + IDX(1, 1),
            1, columnType,
            left, 200,
            current + IDX(1, cols + 1),
            1, columnType,
            right, 200,
            cart, MPI_STATUS_IGNORE
        );

        MPI_Sendrecv(
            current + IDX(1, cols),
            1, columnType,
            right, 201,
            current + IDX(1, 0),
            1, columnType,
            left, 201,
            cart, MPI_STATUS_IGNORE
        );

        int checkChange =
            ((t + 1) % K == 0 || t == T - 1);

        double localChange = 0.0;

        // Update locally owned interior cells
        for (int i = 1; i <= rows; i++) {

            int globalRow = rowStart + i - 1;

            if (globalRow == 0 || globalRow == N - 1)
                continue;

            for (int j = 1; j <= cols; j++) {

                int globalCol = colStart + j - 1;

                if (globalCol == 0 || globalCol == N - 1)
                    continue;

                size_t pos = IDX(i, j);
                double oldValue = current[pos];

                double newValue = oldValue + alpha * (
                    current[IDX(i - 1, j)] +
                    current[IDX(i + 1, j)] +
                    current[IDX(i, j - 1)] +
                    current[IDX(i, j + 1)] -
                    4.0 * oldValue
                );

                next[pos] = newValue;

                if (checkChange) {
                    double diff = fabs(newValue - oldValue);

                    if (diff > localChange)
                        localChange = diff;
                }
            }
        }

        if (checkChange) {
            MPI_Allreduce(
                &localChange, &globalChange,
                1, MPI_DOUBLE, MPI_MAX, cart
            );
        }

        // Swap grids
        double *temp = current;
        current = next;
        next = temp;
    }

    double localTime = MPI_Wtime() - startTime;
    double elapsed = 0.0;

    MPI_Reduce(
        &localTime, &elapsed,
        1, MPI_DOUBLE, MPI_MAX, 0, cart
    );

    // Compute validation statistics
    double localSum = 0.0;
    double localSquares = 0.0;
    double localCenter = 0.0;

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            double value = current[IDX(i, j)];

            localSum += value;
            localSquares += value * value;
        }
    }

    if (center >= rowStart &&
        center < rowStart + rows &&
        center >= colStart &&
        center < colStart + cols) {

        int i = center - rowStart + 1;
        int j = center - colStart + 1;

        localCenter = current[IDX(i, j)];
    }

    double checksum = 0.0;
    double sumSquares = 0.0;
    double centerValue = 0.0;

    MPI_Reduce(
        &localSum, &checksum,
        1, MPI_DOUBLE, MPI_SUM, 0, cart
    );

    MPI_Reduce(
        &localSquares, &sumSquares,
        1, MPI_DOUBLE, MPI_SUM, 0, cart
    );

    MPI_Reduce(
        &localCenter, &centerValue,
        1, MPI_DOUBLE, MPI_SUM, 0, cart
    );

    // Reconstruct complete grid for small tests
    if (N <= 8) {
        double *localFull = calloc(
            (size_t)N * N, sizeof(double)
        );

        double *globalFull = NULL;

        if (!localFull)
            MPI_Abort(cart, 1);

        if (rank == 0) {
            globalFull = calloc(
                (size_t)N * N, sizeof(double)
            );

            if (!globalFull)
                MPI_Abort(cart, 1);
        }

        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                int globalRow = rowStart + i - 1;
                int globalCol = colStart + j - 1;

                localFull[(size_t)globalRow * N +
                          globalCol] = current[IDX(i, j)];
            }
        }

        MPI_Reduce(
            localFull, globalFull,
            N * N, MPI_DOUBLE, MPI_SUM,
            0, cart
        );

        if (rank == 0) {
            printf("\nFinal Temperature Grid:\n");

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    printf("%6.1f",
                           globalFull[(size_t)i * N + j]);
                }
                printf("\n");
            }
        }

        free(localFull);
        free(globalFull);
    }

    if (rank == 0) {
        printf("\nHeat Diffusion (MPI 2D Blocking)\n");
        printf("Grid Size: %d x %d\n", N, N);
        printf("Time Steps: %d\n", T);
        printf("MPI Processes: %d\n", worldSize);
        printf("Process Grid: %d x %d\n", dims[0], dims[1]);
        printf("Convergence K: %d\n", K);

        printf("\n--- Results ---\n");
        printf("Execution Time: %.6f seconds\n", elapsed);
        printf("Checksum: %.10f\n", checksum);
        printf("Sum of Squares: %.10f\n", sumSquares);
        printf("Centre Temperature: %.10f\n", centerValue);
        printf("Last Maximum Change: %.10f\n", globalChange);
    }

    MPI_Type_free(&columnType);

    free(gridA);
    free(gridB);

    MPI_Comm_free(&cart);
    MPI_Finalize();

    return 0;
}
