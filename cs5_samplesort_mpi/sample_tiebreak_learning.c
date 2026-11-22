
/*
 * Educational MPI experiment:
 * Duplicate-aware distributed sample sort.
 *
 * Uses (value, unique ID) pairs to distribute
 * repeated values across multiple MPI ranks.
 */

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOCAL_COUNT 256
#define MAX_PROCESSES 8

// ============================================
// STEP 1: Define a tagged integer
// ============================================

typedef struct {
    int32_t value;
    uint64_t id;
} Tagged;


// ============================================
// STEP 2: Deterministic random generator
// ============================================

static uint32_t next_random(uint32_t *state) {
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    return *state = x;
}


// ============================================
// STEP 3: Compare tagged integers
// ============================================

static int compare_tagged(
    const void *left,
    const void *right
) {
    const Tagged *a = left;
    const Tagged *b = right;

    if (a->value != b->value) {
        return
            (a->value > b->value) -
            (a->value < b->value);
    }

    return
        (a->id > b->id) -
        (a->id < b->id);
}


// ============================================
// STEP 4: Compare ordinary integers
// ============================================

static int compare_value(
    const void *left,
    const void *right
) {
    const int32_t a = *(const int32_t *)left;
    const int32_t b = *(const int32_t *)right;

    return (a > b) - (a < b);
}


// ============================================
// STEP 5: Binary-search partition boundary
// ============================================

static int upper_bound_tagged(
    const Tagged *values,
    int count,
    Tagged splitter
) {
    int first = 0;
    int last = count;

    while (first < last) {
        int middle =
            first + (last - first) / 2;

        if (compare_tagged(
                &values[middle],
                &splitter
            ) <= 0) {

            first = middle + 1;

        } else {
            last = middle;
        }
    }

    return first;
}


// ============================================
// STEP 6: Create an MPI datatype
// ============================================

static MPI_Datatype make_tagged_type(void) {
    Tagged sample;

    MPI_Aint base;
    MPI_Aint displacements[2];

    int lengths[2] = {1, 1};

    MPI_Datatype types[2] = {
        MPI_INT32_T,
        MPI_UINT64_T
    };

    MPI_Datatype raw;
    MPI_Datatype resized;

    MPI_Get_address(&sample, &base);

    MPI_Get_address(
        &sample.value,
        &displacements[0]
    );

    MPI_Get_address(
        &sample.id,
        &displacements[1]
    );

    displacements[0] -= base;
    displacements[1] -= base;

    MPI_Type_create_struct(
        2,
        lengths,
        displacements,
        types,
        &raw
    );

    MPI_Type_create_resized(
        raw,
        0,
        (MPI_Aint)sizeof(Tagged),
        &resized
    );

    MPI_Type_commit(&resized);
    MPI_Type_free(&raw);

    return resized;
}


// ============================================
// STEP 7: Initialize MPI and arguments
// ============================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    if (
        processes < 2 ||
        processes > MAX_PROCESSES ||
        argc != 3 ||
        (
            strcmp(argv[1], "uniform") != 0 &&
            strcmp(argv[1], "duplicates") != 0
        ) ||
        (
            strcmp(argv[2], "1") != 0 &&
            strcmp(argv[2], "4") != 0
        )
    ) {
        if (rank == 0) {
            fprintf(
                stderr,
                "Usage: mpirun -np P "
                "./sample_tiebreak_learning "
                "uniform|duplicates 1|4\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int duplicates =
        strcmp(argv[1], "duplicates") == 0;

    int oversampling = atoi(argv[2]);

    MPI_Datatype mpi_tagged =
        make_tagged_type();


    // ========================================
    // STEP 8: Generate identical test input
    // ========================================

    Tagged local[LOCAL_COUNT];

    int32_t original_values[LOCAL_COUNT];

    uint32_t rng =
        UINT32_C(0xA341316C) ^
        (
            (uint32_t)(rank + 1) *
            UINT32_C(0x9E3779B9)
        );

    for (int i = 0; i < LOCAL_COUNT; i++) {
        int32_t value;

        if (!duplicates) {
            value = (int32_t)(
                next_random(&rng) % 10000
            );

        } else {
            uint32_t roll =
                next_random(&rng) % 10;

            value = roll < 9
                ? 0
                : 1 + (int32_t)(
                    next_random(&rng) % 10000
                );
        }

        original_values[i] = value;

        local[i].value = value;

        local[i].id =
            (uint64_t)rank * LOCAL_COUNT +
            (uint64_t)i;
    }


    // ========================================
    // STEP 9: Tiny serial reference
    // ========================================

    // Only for small-scale verification.

    int32_t reference[
        MAX_PROCESSES * LOCAL_COUNT
    ];

    MPI_Gather(
        original_values,
        LOCAL_COUNT,
        MPI_INT32_T,
        reference,
        LOCAL_COUNT,
        MPI_INT32_T,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 10: Sort local tagged values
    // ========================================

    qsort(
        local,
        LOCAL_COUNT,
        sizeof(Tagged),
        compare_tagged
    );


    // ========================================
    // STEP 11: Select representative samples
    // ========================================

    int samples_per_rank =
        (processes - 1) * oversampling;

    int sample_total =
        processes * samples_per_rank;

    Tagged samples[LOCAL_COUNT];

    Tagged collected[
        MAX_PROCESSES * LOCAL_COUNT
    ];

    Tagged splitters[MAX_PROCESSES];

    for (int i = 0; i < samples_per_rank; i++) {
        int position =
            (i + 1) * LOCAL_COUNT /
            (samples_per_rank + 1);

        samples[i] = local[position];
    }


    // ========================================
    // STEP 12: Gather tagged samples
    // ========================================

    MPI_Gather(
        samples,
        samples_per_rank,
        mpi_tagged,
        collected,
        samples_per_rank,
        mpi_tagged,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 13: Select and broadcast splitters
    // ========================================

    if (rank == 0) {
        qsort(
            collected,
            (size_t)sample_total,
            sizeof(Tagged),
            compare_tagged
        );

        for (int i = 1; i < processes; i++) {
            splitters[i - 1] =
                collected[
                    i * sample_total / processes
                ];
        }
    }

    MPI_Bcast(
        splitters,
        processes - 1,
        mpi_tagged,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 14: Determine destination buckets
    // ========================================

    int send_counts[MAX_PROCESSES] = {0};
    int send_displs[MAX_PROCESSES] = {0};

    int previous = 0;

    for (int dest = 0;
         dest < processes - 1;
         dest++) {

        int end = upper_bound_tagged(
            local,
            LOCAL_COUNT,
            splitters[dest]
        );

        send_displs[dest] = previous;
        send_counts[dest] = end - previous;

        previous = end;
    }

    send_displs[processes - 1] = previous;

    send_counts[processes - 1] =
        LOCAL_COUNT - previous;


    // ========================================
    // STEP 15: Exchange element counts
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
    // STEP 16: Calculate receive offsets
    // ========================================

    int received_total = 0;

    for (int src = 0; src < processes; src++) {
        recv_displs[src] = received_total;
        received_total += recv_counts[src];
    }

    Tagged *received = malloc(
        (size_t)(
            received_total > 0
                ? received_total
                : 1
        ) * sizeof(Tagged)
    );

    if (received == NULL) {
        fprintf(
            stderr,
            "Rank %d: allocation failed\n",
            rank
        );

        MPI_Abort(
            MPI_COMM_WORLD,
            EXIT_FAILURE
        );
    }


    // ========================================
    // STEP 17: Redistribute using Alltoallv
    // ========================================

    MPI_Alltoallv(
        local,
        send_counts,
        send_displs,
        mpi_tagged,
        received,
        recv_counts,
        recv_displs,
        mpi_tagged,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 18: Sort received values
    // ========================================

    qsort(
        received,
        (size_t)received_total,
        sizeof(Tagged),
        compare_tagged
    );


    // ========================================
    // STEP 19: Distributed correctness checks
    // ========================================

    int local_ok = 1;

    for (int i = 1; i < received_total; i++) {
        if (
            compare_tagged(
                &received[i - 1],
                &received[i]
            ) >= 0
        ) {
            local_ok = 0;
        }
    }

    int all_local_ok = 0;

    MPI_Allreduce(
        &local_ok,
        &all_local_ok,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    int global_total = 0;

    MPI_Allreduce(
        &received_total,
        &global_total,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 20: Gather final tiny test result
    // ========================================

    int final_counts[MAX_PROCESSES] = {0};
    int final_displs[MAX_PROCESSES] = {0};

    Tagged final_array[
        MAX_PROCESSES * LOCAL_COUNT
    ];

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

    MPI_Gatherv(
        received,
        received_total,
        mpi_tagged,
        final_array,
        final_counts,
        final_displs,
        mpi_tagged,
        0,
        MPI_COMM_WORLD
    );


    // ========================================
    // STEP 21: Final validation and reporting
    // ========================================

    int success = 1;

    if (rank == 0) {
        int expected =
            processes * LOCAL_COUNT;

        qsort(
            reference,
            (size_t)expected,
            sizeof(int32_t),
            compare_value
        );

        int ordered = 1;
        int exact = global_total == expected;

        if (exact) {
            for (int i = 0; i < expected; i++) {

                if (
                    i > 0 &&
                    compare_tagged(
                        &final_array[i - 1],
                        &final_array[i]
                    ) >= 0
                ) {
                    ordered = 0;
                }

                if (
                    final_array[i].value !=
                    reference[i]
                ) {
                    exact = 0;
                }
            }

        } else {
            ordered = 0;
        }

        int maximum = 0;

        for (int i = 0; i < processes; i++) {
            if (final_counts[i] > maximum) {
                maximum = final_counts[i];
            }
        }

        success =
            all_local_ok &&
            ordered &&
            exact &&
            global_total == expected;

        printf(
            "===== DUPLICATE-AWARE SAMPLE SORT =====\n"
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
            "Oversampling factor: %d\n",
            oversampling
        );

        printf(
            "Samples per process: %d\n",
            samples_per_rank
        );

        printf("Tagged splitters (value, id):");

        for (int i = 0; i < processes - 1; i++) {
            printf(
                " (%d, %llu)",
                splitters[i].value,
                (unsigned long long)splitters[i].id
            );
        }

        printf("\nFinal counts:");

        for (int i = 0; i < processes; i++) {
            printf(" %d", final_counts[i]);
        }

        printf(
            "\nMax/mean imbalance: %.3fx\n",
            (double)maximum / LOCAL_COUNT
        );

        printf(
            "Total elements preserved: %s\n",
            global_total == expected
                ? "PASS" : "FAIL"
        );

        printf(
            "Local sorting: %s\n",
            all_local_ok ? "PASS" : "FAIL"
        );

        printf(
            "Global ordering: %s\n",
            ordered ? "PASS" : "FAIL"
        );

        printf(
            "Exact serial match (values): %s\n",
            exact ? "PASS" : "FAIL"
        );

        printf(
            "Correctness: %s\n",
            success ? "PASS" : "FAIL"
        );
    }

    MPI_Bcast(
        &success,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    free(received);

    MPI_Type_free(&mpi_tagged);

    MPI_Finalize();

    return success
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
