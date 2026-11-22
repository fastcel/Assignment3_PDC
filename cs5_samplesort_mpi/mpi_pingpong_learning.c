
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BYTES (16 * 1024 * 1024)
#define ROUNDS 3
#define WARMUPS 20

// ============================================
// STEP 1: Determine repetitions
// ============================================

static int repetitions(int bytes) {
    if (bytes <= 4096) {
        return 2000;
    }

    if (bytes <= 65536) {
        return 500;
    }

    if (bytes <= 1048576) {
        return 100;
    }

    return 20;
}

// ============================================
// STEP 2: Calculate median of three values
// ============================================

static double median_three(
    double a,
    double b,
    double c
) {
    if (a > b) {
        double temp = a;
        a = b;
        b = temp;
    }

    if (b > c) {
        double temp = b;
        b = c;
        c = temp;
    }

    if (a > b) {
        double temp = a;
        a = b;
        b = temp;
    }

    return b;
}

// ============================================
// STEP 3: Perform a single ping-pong
// ============================================

static void pingpong(
    unsigned char *buffer,
    int bytes,
    int rank
) {
    if (rank == 0) {

        MPI_Send(
            buffer,
            bytes,
            MPI_BYTE,
            1,
            100,
            MPI_COMM_WORLD
        );

        MPI_Recv(
            buffer,
            bytes,
            MPI_BYTE,
            1,
            101,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

    } else {

        MPI_Recv(
            buffer,
            bytes,
            MPI_BYTE,
            0,
            100,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        MPI_Send(
            buffer,
            bytes,
            MPI_BYTE,
            0,
            101,
            MPI_COMM_WORLD
        );
    }
}

// ============================================
// STEP 4: Initialize MPI
// ============================================

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank;
    int processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    if (processes != 2) {

        if (rank == 0) {
            fprintf(
                stderr,
                "This experiment requires "
                "exactly two MPI processes.\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 5: Allocate communication buffer
    // ========================================

    unsigned char *buffer = malloc(MAX_BYTES);

    if (buffer == NULL) {
        fprintf(
            stderr,
            "Rank %d: allocation failed.\n",
            rank
        );

        MPI_Abort(
            MPI_COMM_WORLD,
            EXIT_FAILURE
        );
    }

    memset(
        buffer,
        rank + 1,
        MAX_BYTES
    );

    // ========================================
    // STEP 6: Message sizes in bytes
    // ========================================

    const int sizes[] = {
        0,
        8,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216
    };

    const int size_count =
        sizeof(sizes) / sizeof(sizes[0]);

    // CSV output for future analysis.

    if (rank == 0) {
        printf(
            "bytes,iterations,"
            "one_way_seconds,"
            "effective_GBps\n"
        );

        fflush(stdout);
    }

    // ========================================
    // STEP 7: Benchmark each message size
    // ========================================

    for (int s = 0; s < size_count; s++) {

        int bytes = sizes[s];

        int iterations =
            repetitions(bytes);

        double times[ROUNDS] = {0};

        for (int round = 0;
             round < ROUNDS;
             round++) {

            // --------------------------------
            // Warm up MPI communication
            // --------------------------------

            MPI_Barrier(MPI_COMM_WORLD);

            for (int i = 0; i < WARMUPS; i++) {

                pingpong(
                    buffer,
                    bytes,
                    rank
                );
            }

            // --------------------------------
            // Time repeated round trips
            // --------------------------------

            MPI_Barrier(MPI_COMM_WORLD);

            double start = MPI_Wtime();

            for (int i = 0;
                 i < iterations;
                 i++) {

                pingpong(
                    buffer,
                    bytes,
                    rank
                );
            }

            double end = MPI_Wtime();

            // Rank 0 measured complete
            // round trips.
            //
            // Divide by 2 to estimate
            // one-way message time.

            if (rank == 0) {

                times[round] =
                    (end - start) /
                    (2.0 * iterations);
            }
        }

        // ====================================
        // STEP 8: Report median results
        // ====================================

        if (rank == 0) {

            double one_way = median_three(
                times[0],
                times[1],
                times[2]
            );

            // Decimal GB/s:
            // 1 GB = 1,000,000,000 bytes.

            double bandwidth = 0.0;

            if (bytes > 0 && one_way > 0) {

                bandwidth =
                    ((double)bytes / one_way)
                    / 1e9;
            }

            printf(
                "%d,%d,%.12f,%.6f\n",
                bytes,
                iterations,
                one_way,
                bandwidth
            );

            fflush(stdout);
        }
    }

    // ========================================
    // STEP 9: Cleanup
    // ========================================

    free(buffer);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
