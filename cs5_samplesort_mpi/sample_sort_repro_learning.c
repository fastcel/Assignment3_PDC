
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
// STEP 1: Compare packed sorting keys
// ============================================

static int compare_keys(
    const void *left,
    const void *right
) {
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;

    return (a > b) - (a < b);
}


// ============================================
// STEP 2: Find a partition boundary
// ============================================

// Returns the first index containing
// a key greater than the given splitter.

static int upper_bound_key(
    const uint64_t *data,
    int count,
    uint64_t splitter
) {
    int low = 0;
    int high = count;

    while (low < high) {
        int mid = low + (high - low) / 2;

        if (data[mid] <= splitter) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return low;
}


// ============================================
// STEP 3: Allocate memory safely
// ============================================

static uint64_t *allocate_keys(
    size_t count,
    int rank
) {
    size_t actual = count > 0 ? count : 1;

    if (actual > SIZE_MAX / sizeof(uint64_t)) {
        fprintf(
            stderr,
            "Rank %d: allocation size overflow.\n",
            rank
        );

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    uint64_t *ptr = malloc(
        actual * sizeof(uint64_t)
    );

    if (ptr == NULL) {
        fprintf(
            stderr,
            "Rank %d: memory allocation failed.\n",
            rank
        );

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    return ptr;
}


// ============================================
// STEP 4: Initialize MPI
// ============================================

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank = 0;
    int processes = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);


    // ========================================
    // STEP 5: Validate input arguments
    // ========================================

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
        processes < 1 ||
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
                "./sample_sort_repro_learning "
                "TOTAL uniform|duplicates 1|4\n"
                "Supported MPI ranks: 1 to 8\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }


    // ========================================
    // STEP 6: Partition the global indices
    // ========================================

    uint64_t base =
        total / (uint64_t)processes;

    uint64_t remainder =
        total % (uint64_t)processes;

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
            fprintf(
                stderr,
                "Unsupported local input size.\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int local_n = (int)local_count;


    // ========================================
    // STEP 7: Generate reproducible input
    // ========================================

    uint64_t *local = allocate_keys(
        (size_t)local_n,
        rank
    );

    uint64_t local_input_sum = 0;
    uint64_t local_input_xor = 0;

    for (int i = 0; i < local_n; i++) {

        uint64_t global_index =
            global_offset + (uint64_t)i;

        uint64_t key = learning_key(
            global_index,
            distribution
        );

        local[i] = key;

        uint64_t hash =
            learning_fingerprint(key);

        local_input_sum += hash;
        local_input_xor ^= hash;
    }

    // Input generation and verification
    // fingerprints are excluded from the
    // measured sorting time.


    // ========================================
    // STEP 8: Begin timing
    // ========================================

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();


    // ========================================
    // STEP 9: Sort each rank's local values
    // ========================================

    qsort(
        local,
        (size_t)local_n,
        sizeof(uint64_t),
        compare_keys
    );

    double t1 = MPI_Wtime();


    // ========================================
    // STEP 10: Select regular samples
    // ========================================

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

    uint64_t *gathered = NULL;

    if (rank == 0) {
        gathered = allocate_keys(
            (size_t)total_samples,
            rank
        );
    }


    // ========================================
    // STEP 11: Gather the regular samples
    // ========================================

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


    // ========================================
    // STEP 12: Select global splitters
    // ========================================

    uint64_t splitters[MAX_PROCESSES] = {0};

    if (rank == 0 && total_samples > 0) {

        qsort(
            gathered,
            (size_t)total_samples,
            sizeof(uint64_t),
            compare_keys
        );

        for (int i = 1; i < processes; i++) {

            int index =
                i * total_samples / processes;

            splitters[i - 1] = gathered[index];
        }
    }

    MPI_Bcast(
        splitters,
        processes - 1,
        MPI_UINT64_T,
        0,
        MPI_COMM_WORLD
    );

    double t2 = MPI_Wtime();


    // ========================================
    // STEP 13: Partition local sorted data
    // ========================================

    int send_counts[MAX_PROCESSES] = {0};
    int send_displs[MAX_PROCESSES] = {0};

    int previous = 0;

    for (int dest = 0;
         dest < processes - 1;
         dest++) {

        int boundary = upper_bound_key(
            local,
            local_n,
            splitters[dest]
        );

        send_displs[dest] = previous;
        send_counts[dest] = boundary - previous;

        previous = boundary;
    }

    send_displs[processes - 1] = previous;

    send_counts[processes - 1] =
        local_n - previous;


    // ========================================
    // STEP 14: Exchange send/receive counts
    // ========================================

    int recv_counts[MAX_PROCESSES] = {0};
    int recv_displs[MAX_PROCESSES] = {0};

    MPI_Alltoall(
        send_counts,
        1,
        MPI_INT,
        recv_counts,
        1,
        MPI_INT,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 15: Allocate received partition
    // ========================================

    int received_total = 0;

    for (int src = 0; src < processes; src++) {

        recv_displs[src] = received_total;

        if (
            recv_counts[src] < 0 ||
            recv_counts[src] >
                INT_MAX - received_total
        ) {
            fprintf(
                stderr,
                "Rank %d: receive count overflow.\n",
                rank
            );

            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        received_total += recv_counts[src];
    }

    uint64_t *received = allocate_keys(
        (size_t)received_total,
        rank
    );

    double t3 = MPI_Wtime();


    // ========================================
    // STEP 16: Exchange actual packed keys
    // ========================================

    MPI_Alltoallv(
        local,
        send_counts,
        send_displs,
        MPI_UINT64_T,
        received,
        recv_counts,
        recv_displs,
        MPI_UINT64_T,
        MPI_COMM_WORLD
    );

    double t4 = MPI_Wtime();


    // ========================================
    // STEP 17: Sort received partitions
    // ========================================

    qsort(
        received,
        (size_t)received_total,
        sizeof(uint64_t),
        compare_keys
    );

    double t5 = MPI_Wtime();


    // ========================================
    // STEP 18: Verify each local partition
    // ========================================

    int local_sorted = 1;

    for (int i = 1; i < received_total; i++) {

        if (received[i - 1] >= received[i]) {
            local_sorted = 0;
            break;
        }
    }

    int all_sorted = 0;

    MPI_Reduce(
        &local_sorted,
        &all_sorted,
        1,
        MPI_INT,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 19: Verify the global count
    // ========================================

    uint64_t final_local_count =
        (uint64_t)received_total;

    uint64_t global_final_count = 0;

    MPI_Reduce(
        &final_local_count,
        &global_final_count,
        1,
        MPI_UINT64_T,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 20: Verify input preservation
    // ========================================

    uint64_t local_output_sum = 0;
    uint64_t local_output_xor = 0;

    for (int i = 0; i < received_total; i++) {

        uint64_t hash =
            learning_fingerprint(received[i]);

        local_output_sum += hash;
        local_output_xor ^= hash;
    }

    uint64_t global_input_sum = 0;
    uint64_t global_input_xor = 0;

    uint64_t global_output_sum = 0;
    uint64_t global_output_xor = 0;

    MPI_Reduce(
        &local_input_sum,
        &global_input_sum,
        1,
        MPI_UINT64_T,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_input_xor,
        &global_input_xor,
        1,
        MPI_UINT64_T,
        MPI_BXOR,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_output_sum,
        &global_output_sum,
        1,
        MPI_UINT64_T,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_output_xor,
        &global_output_xor,
        1,
        MPI_UINT64_T,
        MPI_BXOR,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 21: Verify ordering between ranks
    // ========================================

    int has_values = received_total > 0;

    uint64_t local_min = has_values
        ? received[0]
        : UINT64_MAX;

    uint64_t local_max = has_values
        ? received[received_total - 1]
        : 0;

    int all_has[MAX_PROCESSES] = {0};
    int all_counts[MAX_PROCESSES] = {0};

    uint64_t all_mins[MAX_PROCESSES] = {0};
    uint64_t all_maxs[MAX_PROCESSES] = {0};

    MPI_Gather(
        &has_values,
        1,
        MPI_INT,
        all_has,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gather(
        &received_total,
        1,
        MPI_INT,
        all_counts,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gather(
        &local_min,
        1,
        MPI_UINT64_T,
        all_mins,
        1,
        MPI_UINT64_T,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gather(
        &local_max,
        1,
        MPI_UINT64_T,
        all_maxs,
        1,
        MPI_UINT64_T,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 22: Collect phase timings
    // ========================================

    double local_times[6] = {
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3,
        t5 - t4,
        t5 - t0
    };

    double max_times[6] = {0};

    MPI_Reduce(
        local_times,
        max_times,
        6,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 23: Final verification and report
    // ========================================

    int success = 1;

    if (rank == 0) {

        int global_ordered = 1;
        int have_previous = 0;

        uint64_t previous_max = 0;

        int largest_partition = 0;

        for (int i = 0; i < processes; i++) {

            if (all_counts[i] > largest_partition) {
                largest_partition = all_counts[i];
            }

            if (!all_has[i]) {
                continue;
            }

            if (
                have_previous &&
                previous_max >= all_mins[i]
            ) {
                global_ordered = 0;
            }

            previous_max = all_maxs[i];
            have_previous = 1;
        }

        int preserved =
            global_input_sum == global_output_sum &&
            global_input_xor == global_output_xor;

        int count_ok =
            global_final_count == total;

        success =
            all_sorted &&
            global_ordered &&
            preserved &&
            count_ok;

        printf(
            "===== REPRODUCIBLE MPI SAMPLE SORT =====\n"
        );

        printf(
            "Distribution: %s\n",
            argv[2]
        );

        printf(
            "Total integers: %" PRIu64 "\n",
            total
        );

        printf(
            "Processes: %d\n",
            processes
        );

        printf(
            "Oversampling: %d\n",
            oversampling
        );

        printf(
            "Packed key bytes: %zu\n",
            sizeof(uint64_t)
        );

        printf("Final counts:");

        for (int i = 0; i < processes; i++) {
            printf(" %d", all_counts[i]);
        }

        printf("\n");

        double average_count =
            (double)total / processes;

        printf(
            "Max/mean imbalance: %.3fx\n",
            largest_partition / average_count
        );

        printf(
            "Global fingerprint sum: 0x%016"
            PRIx64 "\n",
            global_output_sum
        );

        printf(
            "Global fingerprint xor: 0x%016"
            PRIx64 "\n",
            global_output_xor
        );

        printf(
            "Local sorting: %s\n",
            all_sorted ? "PASS" : "FAIL"
        );

        printf(
            "Global ordering: %s\n",
            global_ordered ? "PASS" : "FAIL"
        );

        printf(
            "Elements preserved: %s\n",
            count_ok ? "PASS" : "FAIL"
        );

        printf(
            "Fingerprint preservation: %s\n",
            preserved ? "PASS" : "FAIL"
        );

        printf(
            "Correctness: %s\n",
            success ? "PASS" : "FAIL"
        );

        printf("\n===== PHASE TIMINGS =====\n");

        printf(
            "Local sorting: %.6f s\n",
            max_times[0]
        );

        printf(
            "Sample selection: %.6f s\n",
            max_times[1]
        );

        printf(
            "Partition and counts: %.6f s\n",
            max_times[2]
        );

        printf(
            "Alltoallv redistribution: %.6f s\n",
            max_times[3]
        );

        printf(
            "Final sorting: %.6f s\n",
            max_times[4]
        );

        printf(
            "Total sorting time: %.6f s\n",
            max_times[5]
        );

        fflush(stdout);
    }

    MPI_Bcast(
        &success,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 24: Cleanup
    // ========================================

    free(local);
    free(received);

    if (rank == 0) {
        free(gathered);
    }

    MPI_Finalize();

    return success
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
