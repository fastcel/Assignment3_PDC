
#include <mpi.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "sample_input_learning.h"

#define MAX_PROCESSES 8
#define MAX_OVERSAMPLING 4

// ============================================
// STEP 1: Compare packed keys
// ============================================

static int compare_keys(
    const void *a,
    const void *b
) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;

    return (x > y) - (x < y);
}

// ============================================
// STEP 2: Locate a bucket boundary
// ============================================

static int upper_bound_key(
    const uint64_t *data,
    int count,
    uint64_t splitter
) {
    int low = 0;
    int high = count;

    while (low < high) {
        int middle = low + (high - low) / 2;

        if (data[middle] <= splitter) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    return low;
}

// ============================================
// STEP 3: Main
// ============================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    uint64_t total = 0;

    int distribution = argc == 4
        ? learning_distribution(argv[2])
        : -1;

    int oversampling = 0;

    if (argc == 4) {
        if (strcmp(argv[3], "1") == 0) {
            oversampling = 1;
        } else if (strcmp(argv[3], "4") == 0) {
            oversampling = 4;
        }
    }

    if (
        argc != 4 ||
        processes < 2 ||
        processes > MAX_PROCESSES ||
        !learning_parse_count(argv[1], &total) ||
        distribution < 0 ||
        oversampling == 0 ||
        total < (uint64_t)processes * 64
    ) {
        if (rank == 0) {
            fprintf(
                stderr,
                "Usage: mpirun -np P "
                "./inspect_samplesort_traffic_learning "
                "TOTAL uniform|duplicates 1|4\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 4: Compute local global-index range
    // ============================================

    uint64_t base = total / (uint64_t)processes;
    uint64_t remainder = total % (uint64_t)processes;

    uint64_t local_count =
        base + ((uint64_t)rank < remainder);

    uint64_t global_offset =
        base * (uint64_t)rank +
        (
            (uint64_t)rank < remainder
                ? (uint64_t)rank
                : remainder
        );

    int samples_per_rank =
        oversampling * (processes - 1);

    if (
        local_count > INT_MAX ||
        local_count <= (uint64_t)samples_per_rank
    ) {
        if (rank == 0) {
            fprintf(stderr, "Invalid local count.\n");
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int local_n = (int)local_count;

    // ========================================
    // STEP 5: Generate the identical input
    // ============================================

    uint64_t *local = malloc(
        (size_t)local_n * sizeof(uint64_t)
    );

    if (local == NULL) {
        fprintf(
            stderr,
            "Rank %d: allocation failed.\n",
            rank
        );

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < local_n; i++) {
        local[i] = learning_key(
            global_offset + (uint64_t)i,
            distribution
        );
    }

    // ========================================
    // STEP 6: Sort locally
    // ============================================

    qsort(
        local,
        (size_t)local_n,
        sizeof(uint64_t),
        compare_keys
    );

    // ========================================
    // STEP 7: Select regular samples
    // ============================================

    uint64_t samples[
        MAX_PROCESSES * MAX_OVERSAMPLING
    ] = {0};

    for (int i = 0; i < samples_per_rank; i++) {
        uint64_t index =
            ((uint64_t)(i + 1) * local_count) /
            (uint64_t)(samples_per_rank + 1);

        samples[i] = local[index];
    }

    int total_samples =
        processes * samples_per_rank;

    uint64_t gathered[
        MAX_PROCESSES *
        MAX_PROCESSES *
        MAX_OVERSAMPLING
    ] = {0};

    // ========================================
    // STEP 8: Gather and select splitters
    // ============================================

    MPI_Gather(
        samples,
        samples_per_rank,
        MPI_UINT64_T,
        gathered,
        samples_per_rank,
        MPI_UINT64_T,
        0,
        MPI_COMM_WORLD
    );

    uint64_t splitters[MAX_PROCESSES] = {0};

    if (rank == 0) {
        qsort(
            gathered,
            (size_t)total_samples,
            sizeof(uint64_t),
            compare_keys
        );

        for (int i = 1; i < processes; i++) {
            splitters[i - 1] =
                gathered[
                    i * total_samples / processes
                ];
        }
    }

    MPI_Bcast(
        splitters,
        processes - 1,
        MPI_UINT64_T,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 9: Compute exact send counts
    // ============================================

    int send_counts[MAX_PROCESSES] = {0};

    int previous = 0;

    for (int dest = 0;
         dest < processes - 1;
         dest++) {

        int boundary = upper_bound_key(
            local,
            local_n,
            splitters[dest]
        );

        send_counts[dest] = boundary - previous;

        previous = boundary;
    }

    send_counts[processes - 1] =
        local_n - previous;

    // ========================================
    // STEP 10: Exchange counts, not keys
    // ============================================

    int recv_counts[MAX_PROCESSES] = {0};

    MPI_Alltoall(
        send_counts,
        1,
        MPI_INT,
        recv_counts,
        1,
        MPI_INT,
        MPI_COMM_WORLD
    );

    int final_local_count = 0;

    for (int src = 0; src < processes; src++) {
        final_local_count += recv_counts[src];
    }

    // ========================================
    // STEP 11: Gather full send-count matrix
    // ============================================

    int send_matrix[
        MAX_PROCESSES * MAX_PROCESSES
    ] = {0};

    int initial_counts[MAX_PROCESSES] = {0};
    int final_counts[MAX_PROCESSES] = {0};

    MPI_Gather(
        send_counts,
        processes,
        MPI_INT,
        send_matrix,
        processes,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gather(
        &local_n,
        1,
        MPI_INT,
        initial_counts,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gather(
        &final_local_count,
        1,
        MPI_INT,
        final_counts,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 12: Analyze actual payload traffic
    // ============================================

    int valid = 1;

    if (rank == 0) {
        uint64_t total_remote_keys = 0;
        uint64_t max_remote_send_keys = 0;
        uint64_t max_remote_recv_keys = 0;
        uint64_t verified_total = 0;

        printf(
            "===== SAMPLE SORT TRAFFIC INSPECTION =====\n"
        );

        printf("Distribution: %s\n", argv[2]);

        printf(
            "Total integers: %" PRIu64 "\n",
            total
        );

        printf("Processes: %d\n", processes);
        printf("Oversampling: %d\n", oversampling);
        printf("Bytes per packed key: %zu\n",
               sizeof(uint64_t));

        printf(
            "\n===== SEND COUNT MATRIX =====\n"
        );

        printf("Rows = sending ranks\n");
        printf("Columns = destination ranks\n");

        for (int src = 0; src < processes; src++) {
            printf("Rank %d:", src);

            for (int dest = 0;
                 dest < processes;
                 dest++) {

                int count =
                    send_matrix[src * processes + dest];

                if (count < 0) {
                    valid = 0;
                }

                printf(" %d", count);
            }

            printf("\n");
        }

        printf(
            "\n===== PER-RANK TRAFFIC =====\n"
        );

        for (int r = 0; r < processes; r++) {
            uint64_t row_sum = 0;
            uint64_t column_sum = 0;

            for (int j = 0; j < processes; j++) {
                row_sum += (uint64_t)
                    send_matrix[r * processes + j];

                column_sum += (uint64_t)
                    send_matrix[j * processes + r];
            }

            uint64_t self_count = (uint64_t)
                send_matrix[r * processes + r];

            uint64_t remote_send =
                row_sum - self_count;

            uint64_t remote_recv =
                column_sum - self_count;

            if (
                row_sum != (uint64_t)initial_counts[r] ||
                column_sum != (uint64_t)final_counts[r]
            ) {
                valid = 0;
            }

            total_remote_keys += remote_send;
            verified_total += row_sum;

            if (remote_send > max_remote_send_keys) {
                max_remote_send_keys = remote_send;
            }

            if (remote_recv > max_remote_recv_keys) {
                max_remote_recv_keys = remote_recv;
            }

            printf(
                "Rank %d: initial=%" PRIu64
                " final=%" PRIu64
                " remote_send_bytes=%" PRIu64
                " remote_recv_bytes=%" PRIu64 "\n",
                r,
                row_sum,
                column_sum,
                remote_send * 8,
                remote_recv * 8
            );
        }

        if (verified_total != total) {
            valid = 0;
        }

        printf(
            "\n===== TRAFFIC SUMMARY =====\n"
        );

        printf(
            "Total remote payload bytes: %" PRIu64
            "\n",
            total_remote_keys * 8
        );

        printf(
            "Max remote send bytes per rank: %" PRIu64
            "\n",
            max_remote_send_keys * 8
        );

        printf(
            "Max remote receive bytes per rank: %" PRIu64
            "\n",
            max_remote_recv_keys * 8
        );

        printf(
            "Mean remote send bytes per rank: %.0f\n",
            (double)(total_remote_keys * 8) /
                processes
        );

        printf(
            "Traffic count consistency: %s\n",
            valid ? "PASS" : "FAIL"
        );
    }

    MPI_Bcast(
        &valid,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    free(local);
    MPI_Finalize();

    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
