
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

#define LOCAL_COUNT 256
#define MAX_PROCESSES 8

// ============================================
// STEP 1: Deterministic random generator
// ============================================

static uint32_t next_random(uint32_t *state) {
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    *state = x;
    return x;
}

// ============================================
// STEP 2: Comparison function
// ============================================

static int compare_int(
    const void *a,
    const void *b
) {
    int x = *(const int *)a;
    int y = *(const int *)b;

    return (x > y) - (x < y);
}

// ============================================
// STEP 3: Find partition boundary
// ============================================

// Returns index of first value > splitter.

static int upper_bound(
    const int *values,
    int count,
    int splitter
) {
    int low = 0;
    int high = count;

    while (low < high) {
        int middle = low + (high - low) / 2;

        if (values[middle] <= splitter) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    return low;
}

// ============================================
// STEP 4: Safe allocation helper
// ============================================

static int *allocate_ints(
    size_t count,
    int rank
) {
    if (count == 0) {
        count = 1;
    }

    int *ptr = malloc(count * sizeof(int));

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
// STEP 5: Main program
// ============================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    if (argc != 3 ||
        processes < 2 ||
        processes > MAX_PROCESSES) {

        if (rank == 0) {
            fprintf(
                stderr,
                "Usage: mpirun -np P "
                "./sample_balance_learning "
                "uniform|duplicates 1|4\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int duplicate_mode = 0;

    if (strcmp(argv[1], "uniform") == 0) {
        duplicate_mode = 0;
    } else if (
        strcmp(argv[1], "duplicates") == 0
    ) {
        duplicate_mode = 1;
    } else {
        if (rank == 0) {
            fprintf(stderr, "Unknown distribution.\n");
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int oversampling = atoi(argv[2]);

    if (oversampling != 1 && oversampling != 4) {
        if (rank == 0) {
            fprintf(stderr, "Use oversampling 1 or 4.\n");
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 6: Generate distributed input
    // ========================================

    int local[LOCAL_COUNT];

    uint32_t state =
        UINT32_C(0xA341316C) ^
        ((uint32_t)(rank + 1) *
         UINT32_C(0x9E3779B9));

    for (int i = 0; i < LOCAL_COUNT; i++) {

        if (!duplicate_mode) {

            local[i] =
                (int)(next_random(&state) % 10000);

        } else {

            uint32_t roll =
                next_random(&state) % 10;

            if (roll < 9) {
                local[i] = 0;
            } else {
                local[i] = 1 + (int)(
                    next_random(&state) % 10000
                );
            }
        }
    }

    // ========================================
    // STEP 7: Sort each local partition
    // ========================================

    qsort(
        local,
        LOCAL_COUNT,
        sizeof(int),
        compare_int
    );

    // ========================================
    // STEP 8: Tiny serial correctness reference
    // ========================================

    // Only for this small learning experiment.
    // Do not gather all input values when
    // implementing the large distributed case.

    int total_original =
        processes * LOCAL_COUNT;

    int *reference = NULL;

    if (rank == 0) {
        reference = allocate_ints(
            (size_t)total_original,
            rank
        );
    }

    MPI_Gather(
        local,
        LOCAL_COUNT,
        MPI_INT,
        reference,
        LOCAL_COUNT,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        qsort(
            reference,
            total_original,
            sizeof(int),
            compare_int
        );
    }

    // ========================================
    // STEP 9: Select local regular samples
    // ========================================

    int sample_count =
        (processes - 1) * oversampling;

    int samples[LOCAL_COUNT];

    for (int i = 0; i < sample_count; i++) {

        int index =
            ((i + 1) * LOCAL_COUNT) /
            (sample_count + 1);

        samples[i] = local[index];
    }

    // ========================================
    // STEP 10: Gather samples at rank 0
    // ========================================

    int total_samples =
        processes * sample_count;

    int *gathered = NULL;

    if (rank == 0) {
        gathered = allocate_ints(
            (size_t)total_samples,
            rank
        );
    }

    MPI_Gather(
        samples,
        sample_count,
        MPI_INT,
        gathered,
        sample_count,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 11: Choose global splitters
    // ========================================

    int splitters[MAX_PROCESSES] = {0};

    if (rank == 0) {

        qsort(
            gathered,
            total_samples,
            sizeof(int),
            compare_int
        );

        for (int i = 1; i < processes; i++) {

            int index =
                (i * total_samples) / processes;

            splitters[i - 1] =
                gathered[index];
        }
    }

    MPI_Bcast(
        splitters,
        processes - 1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 12: Determine send counts
    // ========================================

    int send_counts[MAX_PROCESSES] = {0};
    int send_displs[MAX_PROCESSES] = {0};

    int previous = 0;

    for (int dest = 0;
         dest < processes - 1;
         dest++) {

        int end = upper_bound(
            local,
            LOCAL_COUNT,
            splitters[dest]
        );

        send_displs[dest] = previous;
        send_counts[dest] = end - previous;

        previous = end;
    }

    int last = processes - 1;

    send_displs[last] = previous;
    send_counts[last] = LOCAL_COUNT - previous;

    // ========================================
    // STEP 13: Exchange counts
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
    // STEP 14: Calculate receive offsets
    // ========================================

    int received_total = 0;

    for (int src = 0;
         src < processes;
         src++) {

        recv_displs[src] = received_total;
        received_total += recv_counts[src];
    }

    int *received = allocate_ints(
        (size_t)received_total,
        rank
    );

    // ========================================
    // STEP 15: Redistribute the integers
    // ========================================

    MPI_Alltoallv(
        local,
        send_counts,
        send_displs,
        MPI_INT,
        received,
        recv_counts,
        recv_displs,
        MPI_INT,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 16: Sort received data
    // ========================================

    qsort(
        received,
        received_total,
        sizeof(int),
        compare_int
    );

    // ========================================
    // STEP 17: Distributed correctness checks
    // ========================================

    int locally_sorted = 1;

    for (int i = 1; i < received_total; i++) {
        if (received[i - 1] > received[i]) {
            locally_sorted = 0;
        }
    }

    int all_locally_sorted = 0;

    MPI_Allreduce(
        &locally_sorted,
        &all_locally_sorted,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    int global_count = 0;

    MPI_Allreduce(
        &received_total,
        &global_count,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 18: Gather final counts
    // ========================================

    int final_counts[MAX_PROCESSES] = {0};
    int final_displs[MAX_PROCESSES] = {0};

    MPI_Gather(
        &received_total,
        1,
        MPI_INT,
        final_counts,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    int *final_array = NULL;

    if (rank == 0) {

        for (int i = 1; i < processes; i++) {

            final_displs[i] =
                final_displs[i - 1] +
                final_counts[i - 1];
        }

        final_array = allocate_ints(
            (size_t)total_original,
            rank
        );
    }

    // ========================================
    // STEP 19: Gather only for tiny validation
    // ========================================

    MPI_Gatherv(
        received,
        received_total,
        MPI_INT,
        final_array,
        final_counts,
        final_displs,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 20: Verify and report imbalance
    // ========================================

    int overall_pass = 1;

    if (rank == 0) {

        int exact_match =
            global_count == total_original;

        int globally_ordered = 1;

        for (int i = 0; i < total_original; i++) {

            if (exact_match &&
                final_array[i] != reference[i]) {

                exact_match = 0;
            }

            if (i > 0 &&
                final_array[i - 1] > final_array[i]) {

                globally_ordered = 0;
            }
        }

        int maximum_count = 0;

        for (int i = 0; i < processes; i++) {

            if (final_counts[i] > maximum_count) {
                maximum_count = final_counts[i];
            }
        }

        double mean_count =
            (double)total_original / processes;

        double imbalance =
            maximum_count / mean_count;

        overall_pass =
            all_locally_sorted &&
            globally_ordered &&
            global_count == total_original &&
            exact_match;

        printf(
            "===== SAMPLE SORT LOAD BALANCE =====\n"
        );

        printf(
            "Distribution: %s\n",
            argv[1]
        );

        printf(
            "Processes: %d\n",
            processes
        );

        printf(
            "Values per process initially: %d\n",
            LOCAL_COUNT
        );

        printf(
            "Oversampling factor: %d\n",
            oversampling
        );

        printf(
            "Samples per process: %d\n",
            sample_count
        );

        printf("Splitters:");

        for (int i = 0; i < processes - 1; i++) {
            printf(" %d", splitters[i]);
        }

        printf("\nFinal counts:");

        for (int i = 0; i < processes; i++) {
            printf(" %d", final_counts[i]);
        }

        printf("\n");

        printf(
            "Mean elements per process: %.2f\n",
            mean_count
        );

        printf(
            "Maximum elements per process: %d\n",
            maximum_count
        );

        printf(
            "Max/mean imbalance: %.3fx\n",
            imbalance
        );

        printf(
            "Total elements preserved: %s\n",
            global_count == total_original
                ? "PASS" : "FAIL"
        );

        printf(
            "Local sorting: %s\n",
            all_locally_sorted
                ? "PASS" : "FAIL"
        );

        printf(
            "Global ordering: %s\n",
            globally_ordered
                ? "PASS" : "FAIL"
        );

        printf(
            "Exact serial match: %s\n",
            exact_match
                ? "PASS" : "FAIL"
        );

        printf(
            "Correctness: %s\n",
            overall_pass
                ? "PASS" : "FAIL"
        );
    }

    MPI_Bcast(
        &overall_pass,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    free(received);

    if (rank == 0) {
        free(reference);
        free(gathered);
        free(final_array);
    }

    MPI_Finalize();

    return overall_pass
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
