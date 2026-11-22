
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

#define LOCAL_COUNT 16
#define MAX_PROCESSES 8

// ============================================
// STEP 1: Deterministic per-rank RNG
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
// STEP 2: Integer comparison for qsort
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
// STEP 3: Display an integer array
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
// STEP 4: Initialize MPI
// ============================================

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank;
    int processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    if (processes < 2 ||
        processes > MAX_PROCESSES) {

        if (rank == 0) {
            fprintf(
                stderr,
                "Use between 2 and %d MPI processes.\n",
                MAX_PROCESSES
            );
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // ========================================
    // STEP 5: Generate each rank's own data
    // ========================================

    int local[LOCAL_COUNT];

    uint32_t state =
        UINT32_C(0xA341316C) ^
        ((uint32_t)(rank + 1) *
         UINT32_C(0x9E3779B9));

    for (int i = 0; i < LOCAL_COUNT; i++) {
        local[i] = (int)(
            next_random(&state) % 100
        );
    }

    // ========================================
    // STEP 6: Sort locally
    // ========================================

    qsort(
        local,
        LOCAL_COUNT,
        sizeof(int),
        compare_int
    );

    // ========================================
    // STEP 7: Choose regular samples
    // ========================================

    int samples[MAX_PROCESSES] = {0};

    for (int i = 1; i < processes; i++) {

        int index =
            (i * LOCAL_COUNT) / processes;

        samples[i - 1] = local[index];
    }

    // ========================================
    // STEP 8: Gather samples at rank 0
    // ========================================

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
    // STEP 9: Root chooses global splitters
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

        printf(
            "===== GLOBAL SAMPLE SELECTION =====\n"
        );

        print_array(
            "Sorted gathered samples:",
            gathered,
            sample_count
        );

        for (int i = 1; i < processes; i++) {

            int index =
                (i * sample_count) / processes;

            splitters[i - 1] = gathered[index];
        }

        print_array(
            "Chosen splitters:",
            splitters,
            processes - 1
        );
    }

    // ========================================
    // STEP 10: Broadcast the splitters
    // ========================================

    MPI_Bcast(
        splitters,
        processes - 1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // ========================================
    // STEP 11: Display each process's data
    // ========================================

    for (int current = 0;
         current < processes;
         current++) {

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == current) {

            printf("\n===== PROCESS %d =====\n", rank);

            print_array(
                "Sorted local values:",
                local,
                LOCAL_COUNT
            );

            print_array(
                "Selected samples:",
                samples,
                processes - 1
            );

            print_array(
                "Received splitters:",
                splitters,
                processes - 1
            );

            fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================
    // STEP 12: Verify local sorting
    // ========================================

    int local_sorted = 1;

    for (int i = 1; i < LOCAL_COUNT; i++) {

        if (local[i - 1] > local[i]) {
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

    if (rank == 0) {

        printf("\n===== VERIFICATION =====\n");

        printf(
            "Processes: %d\n",
            processes
        );

        printf(
            "Total distributed values: %d\n",
            processes * LOCAL_COUNT
        );

        printf(
            "Local sorting: %s\n",
            all_sorted ? "PASS" : "FAIL"
        );

        printf(
            "Sample selection: COMPLETE\n"
        );

        printf(
            "Global redistribution: NOT YET DONE\n"
        );
    }

    MPI_Finalize();

    return all_sorted
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
