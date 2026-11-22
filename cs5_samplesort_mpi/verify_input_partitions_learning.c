
#include <mpi.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sample_input_learning.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    // ========================================
    // STEP 1: Validate arguments
    // ========================================

    uint64_t total = 0;

    int distribution = argc == 3
        ? learning_distribution(argv[2])
        : -1;

    if (
        argc != 3 ||
        !learning_parse_count(argv[1], &total) ||
        distribution < 0
    ) {
        if (rank == 0) {
            fprintf(
                stderr,
                "Usage: mpirun -np P "
                "./verify_input_partitions_learning "
                "TOTAL uniform|duplicates\n"
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 2: Divide global indices
    // ========================================

    uint64_t base = total / (uint64_t)processes;
    uint64_t remainder = total % (uint64_t)processes;

    uint64_t local_count =
        base + ((uint64_t)rank < remainder);

    uint64_t offset =
        base * (uint64_t)rank +
        (
            (uint64_t)rank < remainder
                ? (uint64_t)rank
                : remainder
        );

    // ========================================
    // STEP 3: Generate and fingerprint keys
    // ========================================

    uint64_t local_sum = 0;
    uint64_t local_xor = 0;

    for (uint64_t i = 0; i < local_count; i++) {

        uint64_t global_index = offset + i;

        uint64_t key = learning_key(
            global_index,
            distribution
        );

        uint64_t hash =
            learning_fingerprint(key);

        local_sum += hash;
        local_xor ^= hash;
    }

    // ========================================
    // STEP 4: Combine checksums
    // ========================================

    uint64_t global_sum = 0;
    uint64_t global_xor = 0;
    uint64_t global_count = 0;

    MPI_Reduce(
        &local_sum,
        &global_sum,
        1,
        MPI_UINT64_T,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_xor,
        &global_xor,
        1,
        MPI_UINT64_T,
        MPI_BXOR,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_count,
        &global_count,
        1,
        MPI_UINT64_T,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 5: Report consistent input identity
    // ========================================

    if (rank == 0) {
        printf(
            "===== INPUT CONSISTENCY =====\n"
        );

        printf(
            "Distribution: %s\n",
            argv[2]
        );

        printf(
            "Processes: %d\n",
            processes
        );

        printf(
            "Total integers: %" PRIu64 "\n",
            global_count
        );

        printf(
            "Global fingerprint sum: 0x%016"
            PRIx64 "\n",
            global_sum
        );

        printf(
            "Global fingerprint xor: 0x%016"
            PRIx64 "\n",
            global_xor
        );

        printf(
            "Count verification: %s\n",
            global_count == total
                ? "PASS"
                : "FAIL"
        );
    }

    MPI_Finalize();

    return global_count == total || rank != 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
