
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 5
#define E 6

// =============================================
// STEP 1: Graph representation
// =============================================

// Outgoing edges:
// 0 -> 1, 2
// 1 -> 2
// 2 -> 0, 3
// 3 -> 2
// 4 -> none

static const int out_row[N + 1] = {
    0, 2, 3, 5, 6, 6
};

static const int out_col[E] = {
    1, 2, 2, 0, 3, 2
};

// Incoming edges:
// 0 <- 2
// 1 <- 0
// 2 <- 0, 1, 3
// 3 <- 2
// 4 <- none

static const int in_row[N + 1] = {
    0, 1, 2, 5, 6, 6
};

static const int in_src[E] = {
    2, 0, 0, 1, 3, 2
};

// =============================================
// STEP 2: Serial PUSH iteration
// =============================================

static void serial_push(
    const double old_rank[N],
    double result[N],
    double damping
) {
    double dangling = 0.0;

    for (int u = 0; u < N; u++) {
        int degree = out_row[u + 1] - out_row[u];

        if (degree == 0) {
            dangling += old_rank[u];
        }
    }

    double base =
        (1.0 - damping) / N
        + damping * dangling / N;

    for (int v = 0; v < N; v++) {
        result[v] = base;
    }

    // PUSH: visit each source and send rank
    // to every outgoing destination.
    for (int u = 0; u < N; u++) {
        int degree = out_row[u + 1] - out_row[u];

        if (degree == 0) {
            continue;
        }

        double contribution =
            damping * old_rank[u] / degree;

        for (int e = out_row[u];
             e < out_row[u + 1];
             e++) {

            int v = out_col[e];

            result[v] += contribution;
        }
    }
}

// =============================================
// STEP 3: Parallel PULL iteration
// =============================================

static void parallel_pull(
    const double old_rank[N],
    double result[N],
    double damping,
    int threads
) {
    double dangling = 0.0;

    // Dangling mass requires a reduction.
    #pragma omp parallel for num_threads(threads) \
        reduction(+:dangling) schedule(static)
    for (int u = 0; u < N; u++) {
        int degree = out_row[u + 1] - out_row[u];

        if (degree == 0) {
            dangling += old_rank[u];
        }
    }

    double base =
        (1.0 - damping) / N
        + damping * dangling / N;

    // PULL: each iteration owns one destination.
    // No two iterations write to the same result[v].
    #pragma omp parallel for num_threads(threads) \
        schedule(static)
    for (int v = 0; v < N; v++) {

        double incoming_sum = 0.0;

        for (int e = in_row[v];
             e < in_row[v + 1];
             e++) {

            int u = in_src[e];

            int degree =
                out_row[u + 1] - out_row[u];

            incoming_sum += old_rank[u] / degree;
        }

        result[v] = base + damping * incoming_sum;
    }
}

// =============================================
// STEP 4: Compare both implementations
// =============================================

int main(int argc, char **argv) {

    int threads = 4;

    if (argc > 2) {
        fprintf(stderr,
            "Usage: %s [threads]\n", argv[0]);
        return 1;
    }

    if (argc == 2) {
        char *end = NULL;
        long requested = strtol(argv[1], &end, 10);

        if (end == argv[1] || *end != '\0' ||
            requested < 1 || requested > 32) {

            fprintf(stderr, "Invalid thread count\n");
            return 1;
        }

        threads = (int)requested;
    }

    omp_set_dynamic(0);

    const double damping = 0.85;

    double old_rank[N];
    double push_result[N];
    double pull_result[N];

    for (int v = 0; v < N; v++) {
        old_rank[v] = 1.0 / N;
    }

    serial_push(
        old_rank, push_result, damping
    );

    parallel_pull(
        old_rank, pull_result, damping, threads
    );

    printf("===== PUSH VS PULL PAGERANK =====\n");
    printf("Nodes: %d\n", N);
    printf("Edges: %d\n", E);
    printf("Requested threads: %d\n", threads);

    printf("\n%-8s %-16s %-16s\n",
        "Node", "Serial Push", "Parallel Pull");

    double max_difference = 0.0;
    double push_total = 0.0;
    double pull_total = 0.0;

    for (int v = 0; v < N; v++) {

        double difference =
            fabs(push_result[v] - pull_result[v]);

        if (difference > max_difference) {
            max_difference = difference;
        }

        push_total += push_result[v];
        pull_total += pull_result[v];

        printf("%-8d %-16.12f %-16.12f\n",
            v, push_result[v], pull_result[v]);
    }

    printf("\n===== VERIFICATION =====\n");

    printf("Maximum difference: %.12e\n",
        max_difference);

    printf("Serial rank sum: %.12f\n",
        push_total);

    printf("Parallel rank sum: %.12f\n",
        pull_total);

    int passed =
        max_difference < 1e-12
        && fabs(push_total - 1.0) < 1e-12
        && fabs(pull_total - 1.0) < 1e-12;

    printf("Correctness: %s\n",
        passed ? "PASS" : "FAIL");

    return passed ? 0 : 1;
}
