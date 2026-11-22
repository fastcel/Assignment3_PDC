
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 5
#define E 6

// ========================================
// STEP 1: Outgoing and incoming CSR
// ========================================

static const int out_row[N + 1] = {
    0, 2, 3, 5, 6, 6
};

static const int out_col[E] = {
    1, 2, 2, 0, 3, 2
};

static const int in_row[N + 1] = {
    0, 1, 2, 5, 6, 6
};

static const int in_src[E] = {
    2, 0, 0, 1, 3, 2
};

// ========================================
// STEP 2: Serial reference
// ========================================

static void serial_push(
    const double old[N],
    double next[N],
    double damping
) {
    double dangling = 0.0;

    for (int u = 0; u < N; u++) {
        int degree = out_row[u + 1] - out_row[u];

        if (degree == 0)
            dangling += old[u];
    }

    double base =
        (1.0 - damping) / N
        + damping * dangling / N;

    for (int v = 0; v < N; v++)
        next[v] = base;

    for (int u = 0; u < N; u++) {
        int degree = out_row[u + 1] - out_row[u];

        if (degree == 0)
            continue;

        double contribution =
            damping * old[u] / degree;

        for (int e = out_row[u];
             e < out_row[u + 1]; e++) {

            int v = out_col[e];

            next[v] += contribution;
        }
    }
}

// ========================================
// STEP 3: Parallel PUSH with atomics
// ========================================

static void parallel_push_atomic(
    const double old[N],
    double next[N],
    double damping,
    int threads
) {
    double dangling = 0.0;

    // Find dangling mass safely.
    #pragma omp parallel for num_threads(threads) \
        reduction(+:dangling) schedule(static)
    for (int u = 0; u < N; u++) {

        int degree = out_row[u + 1] - out_row[u];

        if (degree == 0)
            dangling += old[u];
    }

    double base =
        (1.0 - damping) / N
        + damping * dangling / N;

    // Initialize all destinations.
    #pragma omp parallel for num_threads(threads)
    for (int v = 0; v < N; v++) {
        next[v] = base;
    }

    // PUSH: parallelize over source nodes.
    #pragma omp parallel for num_threads(threads) \
        schedule(static)
    for (int u = 0; u < N; u++) {

        int degree = out_row[u + 1] - out_row[u];

        if (degree == 0)
            continue;

        double contribution =
            damping * old[u] / degree;

        for (int e = out_row[u];
             e < out_row[u + 1]; e++) {

            int v = out_col[e];

            // Multiple threads may update
            // the same destination.
            #pragma omp atomic update
            next[v] += contribution;
        }
    }
}

// ========================================
// STEP 4: Parallel PULL
// ========================================

static void parallel_pull(
    const double old[N],
    double next[N],
    double damping,
    int threads
) {
    double dangling = 0.0;

    #pragma omp parallel for num_threads(threads) \
        reduction(+:dangling) schedule(static)
    for (int u = 0; u < N; u++) {

        int degree = out_row[u + 1] - out_row[u];

        if (degree == 0)
            dangling += old[u];
    }

    double base =
        (1.0 - damping) / N
        + damping * dangling / N;

    // Each destination is owned by one
    // iteration of the parallel loop.
    #pragma omp parallel for num_threads(threads) \
        schedule(static)
    for (int v = 0; v < N; v++) {

        double incoming = 0.0;

        for (int e = in_row[v];
             e < in_row[v + 1]; e++) {

            int u = in_src[e];

            int degree =
                out_row[u + 1] - out_row[u];

            incoming += old[u] / degree;
        }

        // No atomic needed.
        next[v] = base + damping * incoming;
    }
}

// ========================================
// STEP 5: Compare all three methods
// ========================================

int main(int argc, char **argv) {

    if (argc != 2) {
        fprintf(stderr,
            "Usage: %s threads\n", argv[0]);
        return 1;
    }

    char *end = NULL;
    long requested = strtol(argv[1], &end, 10);

    if (end == argv[1] || *end != '\0' ||
        requested < 1 || requested > 32) {

        fprintf(stderr, "Invalid thread count\n");
        return 1;
    }

    int threads = (int)requested;

    omp_set_dynamic(0);

    const double damping = 0.85;

    double old[N];
    double serial[N];
    double push[N];
    double pull[N];

    for (int v = 0; v < N; v++) {
        old[v] = 1.0 / N;
    }

    serial_push(old, serial, damping);

    parallel_push_atomic(
        old, push, damping, threads
    );

    parallel_pull(
        old, pull, damping, threads
    );

    printf("===== THREE-WAY PAGERANK =====\n");

    printf("Threads: %d\n", threads);

    printf("\n%-6s %-14s %-14s %-14s\n",
        "Node", "Serial", "Push Atomic", "Pull");

    double max_push_diff = 0.0;
    double max_pull_diff = 0.0;

    double push_sum = 0.0;
    double pull_sum = 0.0;

    for (int v = 0; v < N; v++) {

        double push_diff =
            fabs(serial[v] - push[v]);

        double pull_diff =
            fabs(serial[v] - pull[v]);

        if (push_diff > max_push_diff)
            max_push_diff = push_diff;

        if (pull_diff > max_pull_diff)
            max_pull_diff = pull_diff;

        push_sum += push[v];
        pull_sum += pull[v];

        printf(
            "%-6d %-14.10f %-14.10f %-14.10f\n",
            v, serial[v], push[v], pull[v]
        );
    }

    printf("\n===== VERIFICATION =====\n");

    printf("Push max difference: %.12e\n",
           max_push_diff);

    printf("Pull max difference: %.12e\n",
           max_pull_diff);

    printf("Push rank sum: %.12f\n", push_sum);
    printf("Pull rank sum: %.12f\n", pull_sum);

    int passed =
        max_push_diff < 1e-12 &&
        max_pull_diff < 1e-12 &&
        fabs(push_sum - 1.0) < 1e-12 &&
        fabs(pull_sum - 1.0) < 1e-12;

    printf("Correctness: %s\n",
           passed ? "PASS" : "FAIL");

    return passed ? 0 : 1;
}
