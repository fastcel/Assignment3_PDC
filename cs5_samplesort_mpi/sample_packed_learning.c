
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <errno.h>

#define MAX_PROCESSES 8

// ============================================
// STEP 1: Generate deterministic random data
// ============================================

static uint32_t next_random(uint32_t *state) {
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    return *state = x;
}


// ============================================
// STEP 2: Pack value and unique ID
// ============================================

static uint64_t pack_key(
    uint32_t value,
    uint32_t id
) {
    return ((uint64_t)value << 32) | id;
}


// ============================================
// STEP 3: Compare packed integers
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
// STEP 4: Find upper bucket boundary
// ============================================

static int upper_bound_key(
    const uint64_t *values,
    int count,
    uint64_t splitter
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
// STEP 5: Allocation helper
// ============================================

static uint64_t *allocate_keys(
    size_t count,
    int rank
) {
    if (count == 0) {
        count = 1;
    }

    uint64_t *ptr = malloc(
        count * sizeof(uint64_t)
    );

    if (ptr == NULL) {
        fprintf(
            stderr,
            "Rank %d: allocation failed.\n",
            rank
        );

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return ptr;
}


// ============================================
// STEP 6: Fingerprint for data verification
// ============================================

static uint64_t fingerprint(uint64_t value) {
    uint64_t z =
        value + UINT64_C(0x9E3779B97F4A7C15);

    z = (z ^ (z >> 30)) *
        UINT64_C(0xBF58476D1CE4E5B9);

    z = (z ^ (z >> 27)) *
        UINT64_C(0x94D049BB133111EB);

    return z ^ (z >> 31);
}


// ============================================
// STEP 7: Main
// ============================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    // ========================================
    // STEP 8: Validate arguments
    // ========================================

    if (argc != 4 ||
        processes < 2 ||
        processes > MAX_PROCESSES) {

        if (rank == 0) {
            fprintf(
                stderr,
                "Usage: mpirun -np P "
                "./sample_packed_learning "
                "TOTAL uniform|duplicates 1|4\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    errno = 0;

    char *end = NULL;

    unsigned long long input_total =
        strtoull(argv[1], &end, 10);

    if (
        errno != 0 ||
        end == argv[1] ||
        *end != '\0' ||
        input_total > 100000000ULL ||
        input_total < (unsigned long long)
            (processes * 64)
    ) {
        if (rank == 0) {
            fprintf(
                stderr,
                "TOTAL must be between P*64 "
                "and 100000000.\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    uint64_t total = (uint64_t)input_total;

    int duplicate_mode;

    if (strcmp(argv[2], "uniform") == 0) {
        duplicate_mode = 0;
    } else if (
        strcmp(argv[2], "duplicates") == 0
    ) {
        duplicate_mode = 1;
    } else {
        if (rank == 0) {
            fprintf(stderr, "Invalid distribution.\n");
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int oversampling;

    if (strcmp(argv[3], "1") == 0) {
        oversampling = 1;
    } else if (strcmp(argv[3], "4") == 0) {
        oversampling = 4;
    } else {
        if (rank == 0) {
            fprintf(stderr, "Use oversampling 1 or 4.\n");
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 9: Divide input across ranks
    // ========================================

    uint64_t base = total / processes;
    uint64_t remainder = total % processes;

    uint64_t local_count =
        base + ((uint64_t)rank < remainder);

    uint64_t global_offset =
        base * (uint64_t)rank +
        (
            (uint64_t)rank < remainder
                ? (uint64_t)rank
                : remainder
        );

    int local_n = (int)local_count;

    int samples_per_rank =
        oversampling * (processes - 1);

    if (local_n <= samples_per_rank) {
        if (rank == 0) {
            fprintf(stderr, "Input too small.\n");
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 10: Generate local packed keys
    // ========================================

    uint64_t *local =
        allocate_keys((size_t)local_n, rank);

    uint32_t state =
        UINT32_C(0xA341316C) ^
        (
            (uint32_t)(rank + 1) *
            UINT32_C(0x9E3779B9)
        );

    uint64_t input_sum = 0;
    uint64_t input_xor = 0;

    for (int i = 0; i < local_n; i++) {
        uint32_t value;

        if (!duplicate_mode) {
            value = next_random(&state) %
                    UINT32_C(1000000);
        } else {
            uint32_t roll =
                next_random(&state) % 10;

            value = roll < 9
                ? 0
                : 1 + next_random(&state) %
                    UINT32_C(1000000);
        }

        uint32_t id =
            (uint32_t)(global_offset +
                       (uint64_t)i);

        uint64_t key = pack_key(value, id);

        local[i] = key;

        uint64_t h = fingerprint(key);

        input_sum += h;
        input_xor ^= h;
    }

    // Input generation is excluded
    // from sorting timings.

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    // ========================================
    // STEP 11: Local sorting
    // ========================================

    qsort(
        local,
        (size_t)local_n,
        sizeof(uint64_t),
        compare_keys
    );

    double t1 = MPI_Wtime();

    // ========================================
    // STEP 12: Regular sample selection
    // ========================================

    uint64_t samples[MAX_PROCESSES * 4];

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

    double t2 = MPI_Wtime();

    // ========================================
    // STEP 13: Calculate destination buckets
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
        send_counts[dest] =
            boundary - previous;

        previous = boundary;
    }

    send_displs[processes - 1] = previous;

    send_counts[processes - 1] =
        local_n - previous;

    // ========================================
    // STEP 14: Exchange destination counts
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

    int received_total = 0;

    for (int i = 0; i < processes; i++) {
        recv_displs[i] = received_total;
        received_total += recv_counts[i];
    }

    uint64_t *received = allocate_keys(
        (size_t)received_total,
        rank
    );

    double t3 = MPI_Wtime();

    // ========================================
    // STEP 15: Redistribute packed keys
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
    // STEP 16: Final local sorting
    // ========================================

    qsort(
        received,
        (size_t)received_total,
        sizeof(uint64_t),
        compare_keys
    );

    double t5 = MPI_Wtime();

    // ========================================
    // STEP 17: Verify local ordering
    // ========================================

    int local_sorted = 1;

    for (int i = 1; i < received_total; i++) {
        if (received[i - 1] >= received[i]) {
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
    // STEP 18: Verify global element count
    // ========================================

    uint64_t local_final_count =
        (uint64_t)received_total;

    uint64_t global_final_count = 0;

    MPI_Allreduce(
        &local_final_count,
        &global_final_count,
        1,
        MPI_UINT64_T,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 19: Verify all input keys preserved
    // ========================================

    uint64_t output_sum = 0;
    uint64_t output_xor = 0;

    for (int i = 0; i < received_total; i++) {
        uint64_t h = fingerprint(received[i]);

        output_sum += h;
        output_xor ^= h;
    }

    uint64_t global_input_sum = 0;
    uint64_t global_output_sum = 0;
    uint64_t global_input_xor = 0;
    uint64_t global_output_xor = 0;

    MPI_Allreduce(
        &input_sum,
        &global_input_sum,
        1,
        MPI_UINT64_T,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &output_sum,
        &global_output_sum,
        1,
        MPI_UINT64_T,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &input_xor,
        &global_input_xor,
        1,
        MPI_UINT64_T,
        MPI_BXOR,
        MPI_COMM_WORLD
    );

    MPI_Allreduce(
        &output_xor,
        &global_output_xor,
        1,
        MPI_UINT64_T,
        MPI_BXOR,
        MPI_COMM_WORLD
    );

    int preserved =
        global_input_sum == global_output_sum &&
        global_input_xor == global_output_xor;

    // ========================================
    // STEP 20: Verify global partition order
    // ========================================

    int has_values = received_total > 0;

    uint64_t local_min =
        has_values ? received[0] : UINT64_MAX;

    uint64_t local_max =
        has_values ?
        received[received_total - 1] : 0;

    int all_has[MAX_PROCESSES] = {0};
    int all_counts[MAX_PROCESSES] = {0};

    uint64_t all_mins[MAX_PROCESSES] = {0};
    uint64_t all_maxs[MAX_PROCESSES] = {0};

    MPI_Gather(
        &has_values, 1, MPI_INT,
        all_has, 1, MPI_INT,
        0, MPI_COMM_WORLD
    );

    MPI_Gather(
        &received_total, 1, MPI_INT,
        all_counts, 1, MPI_INT,
        0, MPI_COMM_WORLD
    );

    MPI_Gather(
        &local_min, 1, MPI_UINT64_T,
        all_mins, 1, MPI_UINT64_T,
        0, MPI_COMM_WORLD
    );

    MPI_Gather(
        &local_max, 1, MPI_UINT64_T,
        all_maxs, 1, MPI_UINT64_T,
        0, MPI_COMM_WORLD
    );

    // ========================================
    // STEP 21: Collect maximum phase timings
    // ========================================

    double local_times[6] = {
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3,
        t5 - t4,
        t5 - t0
    };

    double maximum_times[6] = {0};

    MPI_Reduce(
        local_times,
        maximum_times,
        6,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 22: Report final results
    // ========================================

    int success = 1;

    if (rank == 0) {
        int boundaries_ok = 1;

        uint64_t previous_max = 0;
        int previous_exists = 0;

        int maximum_count = 0;

        printf("Final counts:");

        for (int i = 0; i < processes; i++) {
            printf(" %d", all_counts[i]);

            if (all_counts[i] > maximum_count) {
                maximum_count = all_counts[i];
            }

            if (!all_has[i]) {
                continue;
            }

            if (
                previous_exists &&
                previous_max >= all_mins[i]
            ) {
                boundaries_ok = 0;
            }

            previous_max = all_maxs[i];
            previous_exists = 1;
        }

        printf("\n");

        success =
            all_sorted &&
            preserved &&
            boundaries_ok &&
            global_final_count == total;

        printf(
            "\n===== PACKED SAMPLE SORT =====\n"
        );

        printf(
            "Distribution: %s\n",
            argv[2]
        );

        printf(
            "Total integers: %llu\n",
            (unsigned long long)total
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

        printf(
            "Max/mean imbalance: %.3fx\n",
            (double)maximum_count /
                ((double)total / processes)
        );

        printf(
            "Local sorting: %s\n",
            all_sorted ? "PASS" : "FAIL"
        );

        printf(
            "Global ordering: %s\n",
            boundaries_ok ? "PASS" : "FAIL"
        );

        printf(
            "Elements preserved: %s\n",
            global_final_count == total
                ? "PASS" : "FAIL"
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
            maximum_times[0]
        );

        printf(
            "Sample selection: %.6f s\n",
            maximum_times[1]
        );

        printf(
            "Partition and counts: %.6f s\n",
            maximum_times[2]
        );

        printf(
            "Alltoallv redistribution: %.6f s\n",
            maximum_times[3]
        );

        printf(
            "Final sorting: %.6f s\n",
            maximum_times[4]
        );

        printf(
            "Total sorting time: %.6f s\n",
            maximum_times[5]
        );
    }

    MPI_Bcast(
        &success,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

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
