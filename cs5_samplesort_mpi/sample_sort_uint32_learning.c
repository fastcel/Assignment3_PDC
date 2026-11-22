#include <mpi.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sample_input_learning.h"

#define MAX_PROCESSES 8
#define MAX_OVERSAMPLING 4

static void fail(int rank, const char *message) {
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

static uint32_t *allocate_values(size_t count, int rank) {
    if (count > SIZE_MAX / sizeof(uint32_t))
        fail(rank, "Allocation size overflow");

    uint32_t *ptr = malloc((count ? count : 1) * sizeof(uint32_t));
    if (!ptr) fail(rank, "Unable to allocate value array");
    return ptr;
}

static int compare_values(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int lower_bound_value(const uint32_t *a, int n, uint32_t x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static int upper_bound_value(const uint32_t *a, int n, uint32_t x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] <= x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static uint64_t fingerprint(uint32_t value) {
    return learning_mix64(
        (uint64_t)value ^ UINT64_C(0xD6E8FEB86659FD93)
    );
}

static uint64_t minimum_u64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, processes = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    uint64_t total = 0;
    int distribution = argc == 4 ? learning_distribution(argv[2]) : -1;
    int oversampling = -1;
    if (argc == 4) {
        if (!strcmp(argv[3], "1")) oversampling = 1;
        if (!strcmp(argv[3], "4")) oversampling = 4;
    }
    if (argc != 4 || processes < 1 || processes > MAX_PROCESSES ||
        !learning_parse_count(argv[1], &total) || distribution < 0 ||
        oversampling < 0 || total < (uint64_t)processes * 64) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np P %s TOTAL uniform|duplicates 1|4 (P=1..8)\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    uint64_t base = total / (uint64_t)processes;
    uint64_t rem = total % (uint64_t)processes;
    uint64_t local_count = base + ((uint64_t)rank < rem);
    uint64_t offset = base * (uint64_t)rank + minimum_u64((uint64_t)rank, rem);
    int sample_count = oversampling * (processes - 1);
    if (local_count > INT_MAX || local_count <= (uint64_t)sample_count)
        fail(rank, "Unsupported local count");
    int local_n = (int)local_count;

    uint32_t *local = allocate_values(local_n, rank);
    uint64_t input_sum = 0, input_xor = 0;
    for (int i = 0; i < local_n; i++) {
        uint32_t value = (uint32_t)(learning_key(offset + (uint64_t)i, distribution) >> 32);
        local[i] = value;
        uint64_t h = fingerprint(value);
        input_sum += h;
        input_xor ^= h;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    qsort(local, (size_t)local_n, sizeof(uint32_t), compare_values);
    double t1 = MPI_Wtime();

    uint32_t samples[MAX_PROCESSES * MAX_OVERSAMPLING] = {0};
    for (int i = 0; i < sample_count; i++) {
        uint64_t index = ((uint64_t)(i + 1) * local_count) / (uint64_t)(sample_count + 1);
        samples[i] = local[index];
    }

    uint32_t gathered[MAX_PROCESSES * MAX_PROCESSES * MAX_OVERSAMPLING] = {0};
    MPI_Gather(samples, sample_count, MPI_UINT32_T,
               gathered, sample_count, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    uint32_t splitters[MAX_PROCESSES] = {0};
    if (rank == 0 && processes > 1) {
        int sample_total = processes * sample_count;
        qsort(gathered, (size_t)sample_total, sizeof(uint32_t), compare_values);
        for (int j = 1; j < processes; j++)
            splitters[j - 1] = gathered[j * sample_total / processes];
    }
    MPI_Bcast(splitters, processes - 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    int send_counts[MAX_PROCESSES] = {0};
    int send_displs[MAX_PROCESSES] = {0};

    if (processes == 1) {
        send_counts[0] = local_n;
    } else {
        int local_less[MAX_PROCESSES] = {0};
        int local_equal[MAX_PROCESSES] = {0};
        for (int j = 0; j < processes - 1; j++) {
            int low = lower_bound_value(local, local_n, splitters[j]);
            int high = upper_bound_value(local, local_n, splitters[j]);
            local_less[j] = low;
            local_equal[j] = high - low;
        }

        int all_less[MAX_PROCESSES * MAX_PROCESSES] = {0};
        int all_equal[MAX_PROCESSES * MAX_PROCESSES] = {0};
        MPI_Allgather(local_less, processes - 1, MPI_INT,
                      all_less, processes - 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(local_equal, processes - 1, MPI_INT,
                      all_equal, processes - 1, MPI_INT, MPI_COMM_WORLD);

        int previous = 0;
        for (int j = 0; j < processes - 1; j++) {
            uint64_t global_less = 0, global_equal = 0;
            uint64_t preceding_equals = 0;
            for (int r = 0; r < processes; r++) {
                int index = r * (processes - 1) + j;
                global_less += (uint64_t)all_less[index];
                global_equal += (uint64_t)all_equal[index];
                if (r < rank) preceding_equals += (uint64_t)all_equal[index];
            }

            uint64_t target = (uint64_t)(j + 1) * total / (uint64_t)processes;
            uint64_t equal_to_left = 0;
            if (target > global_less)
                equal_to_left = minimum_u64(target - global_less, global_equal);

            uint64_t local_share = 0;
            if (equal_to_left > preceding_equals)
                local_share = minimum_u64(
                    equal_to_left - preceding_equals, (uint64_t)local_equal[j]
                );
            int boundary = local_less[j] + (int)local_share;
            if (boundary < previous || boundary > local_n)
                fail(rank, "Non-monotonic duplicate-aware partition boundary");
            send_displs[j] = previous;
            send_counts[j] = boundary - previous;
            previous = boundary;
        }
        send_displs[processes - 1] = previous;
        send_counts[processes - 1] = local_n - previous;
    }

    int recv_counts[MAX_PROCESSES] = {0};
    int recv_displs[MAX_PROCESSES] = {0};
    MPI_Alltoall(send_counts, 1, MPI_INT,
                 recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int received_total = 0;
    for (int r = 0; r < processes; r++) {
        if (recv_counts[r] < 0 || recv_counts[r] > INT_MAX - received_total)
            fail(rank, "Receive count overflow");
        recv_displs[r] = received_total;
        received_total += recv_counts[r];
    }
    uint32_t *received = allocate_values((size_t)received_total, rank);
    double t3 = MPI_Wtime();

    MPI_Alltoallv(local, send_counts, send_displs, MPI_UINT32_T,
                  received, recv_counts, recv_displs, MPI_UINT32_T,
                  MPI_COMM_WORLD);
    double t4 = MPI_Wtime();

    qsort(received, (size_t)received_total, sizeof(uint32_t), compare_values);
    double t5 = MPI_Wtime();

    int local_ordered = 1;
    for (int i = 1; i < received_total; i++) {
        if (received[i - 1] > received[i]) {
            local_ordered = 0;
            break;
        }
    }

    uint64_t output_sum = 0, output_xor = 0;
    for (int i = 0; i < received_total; i++) {
        uint64_t h = fingerprint(received[i]);
        output_sum += h;
        output_xor ^= h;
    }

    int all_ordered = 0;
    MPI_Reduce(&local_ordered, &all_ordered, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    uint64_t local_output_count = (uint64_t)received_total;
    uint64_t global_count = 0;
    MPI_Reduce(&local_output_count, &global_count, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t global_input_sum = 0, global_input_xor = 0;
    uint64_t global_output_sum = 0, global_output_xor = 0;
    MPI_Reduce(&input_sum, &global_input_sum, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&input_xor, &global_input_xor, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    MPI_Reduce(&output_sum, &global_output_sum, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&output_xor, &global_output_xor, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    int has_values = received_total > 0;
    uint32_t local_min = has_values ? received[0] : UINT32_MAX;
    uint32_t local_max = has_values ? received[received_total - 1] : 0;
    int all_has[MAX_PROCESSES] = {0}, all_counts[MAX_PROCESSES] = {0};
    uint32_t all_mins[MAX_PROCESSES] = {0}, all_maxs[MAX_PROCESSES] = {0};
    MPI_Gather(&has_values, 1, MPI_INT, all_has, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&received_total, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_min, 1, MPI_UINT32_T, all_mins, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_max, 1, MPI_UINT32_T, all_maxs, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    double local_times[6] = { t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t5-t0 };
    double max_times[6] = {0};
    double all_rank_times[MAX_PROCESSES][6] = {{0}};
    MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gather(local_times, 6, MPI_DOUBLE, all_rank_times, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int success = 1;
    if (rank == 0) {
        int global_ordered = 1, have_previous = 0;
        uint32_t previous_max = 0;
        int largest_partition = 0;
        for (int r = 0; r < processes; r++) {
            if (all_counts[r] > largest_partition) largest_partition = all_counts[r];
            if (!all_has[r]) continue;
            if (have_previous && previous_max > all_mins[r]) global_ordered = 0;
            previous_max = all_maxs[r];
            have_previous = 1;
        }
        int preserved = global_input_sum == global_output_sum &&
                        global_input_xor == global_output_xor;
        int count_ok = global_count == total;
        success = all_ordered && global_ordered && preserved && count_ok;

        printf("===== NATIVE 32-BIT MPI SAMPLE SORT =====\n");
        printf("Distribution: %s\n", argv[2]);
        printf("Total integers: %" PRIu64 "\n", total);
        printf("Processes: %d\n", processes);
        printf("Oversampling: %d\n", oversampling);
        printf("Element bytes: %zu\n", sizeof(uint32_t));
        printf("MPI payload datatype: MPI_UINT32_T\n");
        printf("Final counts:");
        for (int r = 0; r < processes; r++) printf(" %d", all_counts[r]);
        printf("\n");
        printf("Max/mean imbalance: %.3fx\n", (double)largest_partition / ((double)total / processes));
        printf("Global fingerprint sum: 0x%016" PRIx64 "\n", global_input_sum);
        printf("Global fingerprint xor: 0x%016" PRIx64 "\n", global_input_xor);
        printf("Local sorting: %s\n", all_ordered ? "PASS" : "FAIL");
        printf("Global ordering: %s\n", global_ordered ? "PASS" : "FAIL");
        printf("Elements preserved: %s\n", count_ok ? "PASS" : "FAIL");
        printf("Fingerprint preservation: %s\n", preserved ? "PASS" : "FAIL");
        printf("Correctness: %s\n", success ? "PASS" : "FAIL");

        printf("\n===== PHASE TIMINGS =====\n");
        printf("Local sorting: %.6f s\n", max_times[0]);
        printf("Sample selection: %.6f s\n", max_times[1]);
        printf("Partition and counts: %.6f s\n", max_times[2]);
        printf("Alltoallv redistribution: %.6f s\n", max_times[3]);
        printf("Final sorting: %.6f s\n", max_times[4]);
        printf("Total sorting time: %.6f s\n", max_times[5]);

        printf("\n===== PER-RANK PHASE TIMINGS =====\n");
        printf("rank,local_sort_s,sampling_s,partition_s,alltoallv_s,final_sort_s,phase_window_s\n");
        for (int r = 0; r < processes; r++)
            printf("%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n", r,
                   all_rank_times[r][0], all_rank_times[r][1], all_rank_times[r][2],
                   all_rank_times[r][3], all_rank_times[r][4], all_rank_times[r][5]);
        printf("===== END PER-RANK PHASE TIMINGS =====\n");
    }

    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    free(local);
    free(received);
    MPI_Finalize();
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
