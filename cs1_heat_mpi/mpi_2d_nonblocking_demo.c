
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define IDX(i, j) ((size_t)(i) * pitch + (size_t)(j))

// Calculate local block size
static int block_size(int N, int parts, int coord) {
    return N / parts + (coord < N % parts);
}

// Calculate global starting position
static int block_start(int N, int parts, int coord) {
    return coord * (N / parts)
         + (coord < N % parts ? coord : N % parts);
}

// Apply the heat diffusion formula to one cell
static inline void update_cell(
    const double *current,
    double *next,
    size_t pos,
    size_t pitch,
    double alpha,
    int check_change,
    double *local_change
) {
    double old_value = current[pos];

    double new_value = old_value + alpha * (
        current[pos - pitch] +
        current[pos + pitch] +
        current[pos - 1] +
        current[pos + 1] -
        4.0 * old_value
    );

    next[pos] = new_value;

    if (check_change) {
        double difference = fabs(new_value - old_value);

        if (difference > *local_change)
            *local_change = difference;
    }
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = (argc > 1) ? atoi(argv[1]) : 5;
    int T = (argc > 2) ? atoi(argv[2]) : 2;
    int K = (argc > 3) ? atoi(argv[3]) : 1;

    const double alpha = 0.1;

    if (argc > 4 || N < 3 || T < 0 || K < 1) {
        fprintf(stderr, "Invalid input parameters\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // STEP 1: Create the 2D Cartesian topology

    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);

    if (dims[0] > N || dims[1] > N) {
        fprintf(stderr, "Too many MPI processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int periods[2] = {0, 0};
    MPI_Comm cart;

    MPI_Cart_create(
        MPI_COMM_WORLD,
        2, dims, periods, 0, &cart
    );

    int rank;
    int coords[2];

    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 2, coords);

    // STEP 2: Find neighbouring ranks

    int up, down, left, right;

    MPI_Cart_shift(cart, 0, 1, &up, &down);
    MPI_Cart_shift(cart, 1, 1, &left, &right);

    // STEP 3: Determine local block dimensions

    int rows = block_size(N, dims[0], coords[0]);
    int cols = block_size(N, dims[1], coords[1]);

    int row_start = block_start(N, dims[0], coords[0]);
    int col_start = block_start(N, dims[1], coords[1]);

    size_t pitch = (size_t)cols + 2;
    size_t cells = ((size_t)rows + 2) * pitch;

    // Allocate two grids including halo cells
    double *gridA = calloc(cells, sizeof(double));
    double *gridB = calloc(cells, sizeof(double));

    if (!gridA || !gridB) {
        fprintf(stderr, "Rank %d: allocation failed\n", rank);
        MPI_Abort(cart, 1);
    }

    double *current = gridA;
    double *next = gridB;

    // STEP 4: Initialize the central hot cell

    int center = N / 2;

    if (center >= row_start &&
        center < row_start + rows &&
        center >= col_start &&
        center < col_start + cols) {

        int local_i = center - row_start + 1;
        int local_j = center - col_start + 1;

        current[IDX(local_i, local_j)] = 100.0;
    }

    // STEP 5: Create MPI datatype for columns

    MPI_Datatype column_type;

    MPI_Type_vector(
        rows, 1, (int)pitch,
        MPI_DOUBLE, &column_type
    );

    MPI_Type_commit(&column_type);

    MPI_Barrier(cart);
    double start_time = MPI_Wtime();

    double post_time = 0.0;
    double interior_time = 0.0;
    double wait_time = 0.0;
    double boundary_time = 0.0;
    double global_change = 0.0;

    // STEP 6: Main simulation loop

    for (int t = 0; t < T; t++) {

        MPI_Request requests[8];

        double phase_start = MPI_Wtime();

        // Post non-blocking receives
        MPI_Irecv(
            current + IDX(0, 1),
            cols, MPI_DOUBLE,
            up, 101, cart, &requests[0]
        );

        MPI_Irecv(
            current + IDX(rows + 1, 1),
            cols, MPI_DOUBLE,
            down, 100, cart, &requests[1]
        );

        MPI_Irecv(
            current + IDX(1, 0),
            1, column_type,
            left, 201, cart, &requests[2]
        );

        MPI_Irecv(
            current + IDX(1, cols + 1),
            1, column_type,
            right, 200, cart, &requests[3]
        );

        // Post non-blocking sends
        MPI_Isend(
            current + IDX(1, 1),
            cols, MPI_DOUBLE,
            up, 100, cart, &requests[4]
        );

        MPI_Isend(
            current + IDX(rows, 1),
            cols, MPI_DOUBLE,
            down, 101, cart, &requests[5]
        );

        MPI_Isend(
            current + IDX(1, 1),
            1, column_type,
            left, 200, cart, &requests[6]
        );

        MPI_Isend(
            current + IDX(1, cols),
            1, column_type,
            right, 201, cart, &requests[7]
        );

        post_time += MPI_Wtime() - phase_start;

        int check_change =
            ((t + 1) % K == 0 || t == T - 1);

        double local_change = 0.0;

        // STEP 7: Compute independent interior cells
        phase_start = MPI_Wtime();

        for (int i = 2; i < rows; i++) {
            for (int j = 2; j < cols; j++) {

                update_cell(
                    current, next,
                    IDX(i, j), pitch,
                    alpha, check_change,
                    &local_change
                );
            }
        }

        interior_time += MPI_Wtime() - phase_start;

        // STEP 8: Wait for communication to finish
        phase_start = MPI_Wtime();

        MPI_Waitall(
            8, requests, MPI_STATUSES_IGNORE
        );

        wait_time += MPI_Wtime() - phase_start;

        // STEP 9: OPTIMIZED BOUNDARY COMPUTATION
        phase_start = MPI_Wtime();

        // Top and bottom owned rows
        for (int edge = 0; edge < 2; edge++) {

            if (edge == 1 && rows == 1)
                continue;

            int i = (edge == 0) ? 1 : rows;
            int global_i = row_start + i - 1;

            // Fixed global boundary
            if (global_i == 0 || global_i == N - 1)
                continue;

            for (int j = 1; j <= cols; j++) {

                int global_j = col_start + j - 1;

                if (global_j == 0 || global_j == N - 1)
                    continue;

                update_cell(
                    current, next,
                    IDX(i, j), pitch,
                    alpha, check_change,
                    &local_change
                );
            }
        }

        // Left and right owned columns
        // Exclude corners to avoid double updates
        for (int i = 2; i < rows; i++) {

            int global_i = row_start + i - 1;

            if (global_i == 0 || global_i == N - 1)
                continue;

            for (int edge = 0; edge < 2; edge++) {

                if (edge == 1 && cols == 1)
                    continue;

                int j = (edge == 0) ? 1 : cols;
                int global_j = col_start + j - 1;

                if (global_j == 0 || global_j == N - 1)
                    continue;

                update_cell(
                    current, next,
                    IDX(i, j), pitch,
                    alpha, check_change,
                    &local_change
                );
            }
        }

        boundary_time += MPI_Wtime() - phase_start;

        // STEP 10: Global convergence check
        if (check_change) {
            MPI_Allreduce(
                &local_change,
                &global_change,
                1, MPI_DOUBLE,
                MPI_MAX, cart
            );
        }

        // Swap the grids
        double *temp = current;
        current = next;
        next = temp;
    }

    // STEP 11: Reduce timing measurements

    double local_elapsed = MPI_Wtime() - start_time;

    double max_elapsed = 0.0;
    double max_post = 0.0;
    double max_interior = 0.0;
    double max_wait = 0.0;
    double max_boundary = 0.0;

    MPI_Reduce(
        &local_elapsed, &max_elapsed,
        1, MPI_DOUBLE, MPI_MAX, 0, cart
    );

    MPI_Reduce(
        &post_time, &max_post,
        1, MPI_DOUBLE, MPI_MAX, 0, cart
    );

    MPI_Reduce(
        &interior_time, &max_interior,
        1, MPI_DOUBLE, MPI_MAX, 0, cart
    );

    MPI_Reduce(
        &wait_time, &max_wait,
        1, MPI_DOUBLE, MPI_MAX, 0, cart
    );

    MPI_Reduce(
        &boundary_time, &max_boundary,
        1, MPI_DOUBLE, MPI_MAX, 0, cart
    );

    // STEP 12: Calculate correctness statistics

    double local_sum = 0.0;
    double local_sq = 0.0;
    double local_center = 0.0;

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {

            double value = current[IDX(i, j)];

            local_sum += value;
            local_sq += value * value;
        }
    }

    if (center >= row_start &&
        center < row_start + rows &&
        center >= col_start &&
        center < col_start + cols) {

        int local_i = center - row_start + 1;
        int local_j = center - col_start + 1;

        local_center = current[IDX(local_i, local_j)];
    }

    double checksum = 0.0;
    double sum_squares = 0.0;
    double center_value = 0.0;

    MPI_Reduce(
        &local_sum, &checksum,
        1, MPI_DOUBLE, MPI_SUM, 0, cart
    );

    MPI_Reduce(
        &local_sq, &sum_squares,
        1, MPI_DOUBLE, MPI_SUM, 0, cart
    );

    MPI_Reduce(
        &local_center, &center_value,
        1, MPI_DOUBLE, MPI_SUM, 0, cart
    );

    // STEP 13: Reconstruct small grids for verification

    if (N <= 8) {

        double *local_full = calloc(
            (size_t)N * N, sizeof(double)
        );

        double *global_full = NULL;

        if (rank == 0) {
            global_full = calloc(
                (size_t)N * N, sizeof(double)
            );
        }

        if (!local_full ||
            (rank == 0 && !global_full)) {
            MPI_Abort(cart, 1);
        }

        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {

                int gi = row_start + i - 1;
                int gj = col_start + j - 1;

                local_full[(size_t)gi * N + gj] =
                    current[IDX(i, j)];
            }
        }

        MPI_Reduce(
            local_full, global_full,
            N * N, MPI_DOUBLE,
            MPI_SUM, 0, cart
        );

        if (rank == 0) {
            printf("\nFinal Temperature Grid:\n");

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {

                    printf(
                        "%6.1f",
                        global_full[(size_t)i * N + j]
                    );
                }

                printf("\n");
            }
        }

        free(local_full);
        free(global_full);
    }

    // STEP 14: Print results

    if (rank == 0) {

        printf("\nHeat Diffusion (MPI 2D Non-Blocking Optimized)\n");
        printf("Grid Size: %d x %d\n", N, N);
        printf("Time Steps: %d\n", T);
        printf("MPI Processes: %d\n", world_size);
        printf("Process Grid: %d x %d\n",
               dims[0], dims[1]);
        printf("Convergence K: %d\n", K);

        printf("\n--- Results ---\n");

        printf("Execution Time: %.6f seconds\n",
               max_elapsed);
        printf("Max Post Time: %.6f seconds\n",
               max_post);
        printf("Max Interior Compute Time: %.6f seconds\n",
               max_interior);
        printf("Max Waitall Time: %.6f seconds\n",
               max_wait);
        printf("Max Boundary Compute Time: %.6f seconds\n",
               max_boundary);

        printf("Checksum: %.10f\n", checksum);
        printf("Sum of Squares: %.10f\n", sum_squares);
        printf("Centre Temperature: %.10f\n",
               center_value);
        printf("Last Maximum Change: %.10f\n",
               global_change);
    }

    MPI_Type_free(&column_type);

    free(gridA);
    free(gridB);

    MPI_Comm_free(&cart);
    MPI_Finalize();

    return 0;
}
