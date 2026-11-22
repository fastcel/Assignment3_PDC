
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <mpi.h>

#define LOCAL_COUNT 16
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
// STEP 2: Integer comparison
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
// STEP 3: Print a sequence
// ============================================

static void print_array(
    const char *label,
    const int *values,
    int count
) {
    printf("%s", label);

    for (int i = 0; i < count; i++) {
        printf(" %d", values[i]);
    }

    printf("\n");
}

// ============================================
// STEP 4: Binary search for bucket boundaries
// ============================================

// Returns the first index whose value is
// greater than the splitter.
//
// Thus values <= splitter remain in the
// lower-numbered destination bucket.

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
// STEP 5: Initialize MPI
// ============================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    if (processes < 2 ||
        processes > MAX_PROCESSES) {

        if (rank == 0) {
            fprintf(
                stderr,
                "Use 2 to %d MPI processes.\n",
                MAX_PROCESSES
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 6: Generate and sort local data
    // ========================================

    int local[LOCAL_COUNT];

    uint32_t state =
        UINT32_C(0xA341316C) ^
        ((uint32_t)(rank + 1) *
         UINT32_C(0x9E3779B9));

    for (int i = 0; i < LOCAL_COUNT; i++) {
        local[i] =
            (int)(next_random(&state) % 100);
    }

    qsort(
        local,
        LOCAL_COUNT,
        sizeof(int),
        compare_int
    );

    // ========================================
    // STEP 7: Save an exact serial reference
    // ========================================

    // This gather is ONLY for the tiny
    // educational correctness test.
    //
    // The large distributed implementation
    // must not gather the full dataset at
    // one process.

    int reference[
        MAX_PROCESSES * LOCAL_COUNT
    ] = {0};

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
            processes * LOCAL_COUNT,
            sizeof(int),
            compare_int
        );
    }

    // ========================================
    // STEP 8: Select regular samples
    // ========================================

    int samples[MAX_PROCESSES] = {0};

    for (int i = 1; i < processes; i++) {
        int index =
            (i * LOCAL_COUNT) / processes;

        samples[i - 1] = local[index];
    }

    int gathered[
        MAX_PROCESSES * MAX_PROCESSES
    ] = {0};

    MPI_Gather(
        samples,
        processes - 1,
        MPI_INT,
        gathered,
        processes - 1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 9: Choose and broadcast splitters
    // ========================================

    int splitters[MAX_PROCESSES] = {0};

    if (rank == 0) {
        int sample_count =
            processes * (processes - 1);

        qsort(
            gathered,
            sample_count,
            sizeof(int),
            compare_int
        );

        for (int i = 1; i < processes; i++) {
            int index =
                (i * sample_count) / processes;

            splitters[i - 1] =
                gathered[index];
        }

        printf(
            "===== GLOBAL SPLITTERS =====\n"
        );

        print_array(
            "Splitters:",
            splitters,
            processes - 1
        );

        fflush(stdout);
    }

    MPI_Bcast(
        splitters,
        processes - 1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 10: Partition local sorted values
    // ========================================

    int send_counts[MAX_PROCESSES] = {0};
    int send_displs[MAX_PROCESSES] = {0};

    int previous = 0;

    for (int destination = 0;
         destination < processes - 1;
         destination++) {

        int end = upper_bound(
            local,
            LOCAL_COUNT,
            splitters[destination]
        );

        send_displs[destination] = previous;

        send_counts[destination] =
            end - previous;

        previous = end;
    }

    int last = processes - 1;

    send_displs[last] = previous;
    send_counts[last] = LOCAL_COUNT - previous;

    // ========================================
    // STEP 11: Exchange bucket sizes
    // ========================================

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

    // ========================================
    // STEP 12: Calculate receive displacements
    // ========================================

    int recv_displs[MAX_PROCESSES] = {0};

    int received_total = 0;

    for (int source = 0;
         source < processes;
         source++) {

        recv_displs[source] = received_total;

        received_total += recv_counts[source];
    }

    int allocation_count =
        received_total > 0 ? received_total : 1;

    int *received = malloc(
        (size_t)allocation_count * sizeof(int)
    );

    if (received == NULL) {
        fprintf(
            stderr,
            "Rank %d: allocation failed.\n",
            rank
        );

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // ========================================
    // STEP 13: Exchange the actual integers
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
    // STEP 14: Sort each received partition
    // ========================================

    qsort(
        received,
        received_total,
        sizeof(int),
        compare_int
    );

    // ========================================
    // STEP 15: Display each process's work
    // ========================================

    for (int current = 0;
         current < processes;
         current++) {

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == current) {
            printf(
                "\n===== PROCESS %d =====\n",
                rank
            );

            print_array(
                "Original local values:",
                local,
                LOCAL_COUNT
            );

            print_array(
                "Send counts:",
                send_counts,
                processes
            );

            print_array(
                "Receive counts:",
                recv_counts,
                processes
            );

            printf(
                "Final local count: %d\n",
                received_total
            );

            print_array(
                "Final sorted values:",
                received,
                received_total
            );

            fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================
    // STEP 16: Verify local sorting
    // ========================================

    int local_sorted = 1;

    for (int i = 1; i < received_total; i++) {
        if (received[i - 1] > received[i]) {
            local_sorted = 0;
        }
    }

    int all_sorted = 0;

    MPI_Allreduce(
        &local_sorted,
        &all_sorted,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 17: Verify total element count
    // ========================================

    int final_total = 0;

    MPI_Allreduce(
        &received_total,
        &final_total,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 18: Verify global partition order
    // ========================================

    int has_values = received_total > 0;

    int local_min = has_values
        ? received[0] : INT_MAX;

    int local_max = has_values
        ? received[received_total - 1] : INT_MIN;

    int all_has[MAX_PROCESSES] = {0};
    int all_mins[MAX_PROCESSES] = {0};
    int all_maxs[MAX_PROCESSES] = {0};

    MPI_Gather(
        &has_values, 1, MPI_INT,
        all_has, 1, MPI_INT,
        0, MPI_COMM_WORLD
    );

    MPI_Gather(
        &local_min, 1, MPI_INT,
        all_mins, 1, MPI_INT,
        0, MPI_COMM_WORLD
    );

    MPI_Gather(
        &local_max, 1, MPI_INT,
        all_maxs, 1, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // ========================================
    // STEP 19: Gather tiny test output
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

    if (rank == 0) {
        for (int i = 1; i < processes; i++) {
            final_displs[i] =
                final_displs[i - 1] +
                final_counts[i - 1];
        }
    }

    int final_array[
        MAX_PROCESSES * LOCAL_COUNT
    ] = {0};

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
    // STEP 20: Check exact global correctness
    // ========================================

    int overall_pass = 1;

    if (rank == 0) {
        int expected_total =
            processes * LOCAL_COUNT;

        int boundary_ok = 1;
        int previous_max = INT_MIN;
        int have_previous = 0;

        for (int i = 0; i < processes; i++) {
            if (!all_has[i]) {
                continue;
            }

            if (have_previous &&
                previous_max > all_mins[i]) {
                boundary_ok = 0;
            }

            previous_max = all_maxs[i];
            have_previous = 1;
        }

        int exact_match =
            final_total == expected_total;

        if (exact_match) {
            for (int i = 0; i < expected_total; i++) {
                if (reference[i] != final_array[i]) {
                    exact_match = 0;
                    break;
                }
            }
        }

        overall_pass =
            all_sorted &&
            boundary_ok &&
            final_total == expected_total &&
            exact_match;

        printf(
            "\n===== GLOBAL VERIFICATION =====\n"
        );

        printf(
            "Expected integers: %d\n",
            expected_total
        );

        printf(
            "Received integers: %d\n",
            final_total
        );

        printf(
            "Local sorting: %s\n",
            all_sorted ? "PASS" : "FAIL"
        );

        printf(
            "Global boundary ordering: %s\n",
            boundary_ok ? "PASS" : "FAIL"
        );

        printf(
            "Exact serial reference comparison: %s\n",
            exact_match ? "PASS" : "FAIL"
        );

        printf(
            "Global sorting: %s\n",
            overall_pass ? "PASS" : "FAIL"
        );

        print_array(
            "Final elements per process:",
            final_counts,
            processes
        );

        fflush(stdout);
    }

    MPI_Bcast(
        &overall_pass,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    free(received);

    MPI_Finalize();

    return overall_pass
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
