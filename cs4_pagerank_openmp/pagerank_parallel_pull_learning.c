
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define DAMPING 0.85
#define TOLERANCE 1e-10
#define MAX_ITERATIONS 1000

// ============================================
// STEP 1: Graph and results structures
// ============================================

typedef struct {
    int n;
    size_t m;

    size_t *out_row;
    int *out_col;

    size_t *in_row;
    int *in_src;
} Graph;

typedef struct {
    int iterations;
    double residual;
    double elapsed;
    double mass;
} Stats;

// ============================================
// STEP 2: Checked allocation
// ============================================

static void *allocate(size_t count, size_t size) {
    if (count == 0 || size == 0 ||
        count > SIZE_MAX / size) {
        fprintf(stderr, "Invalid allocation size\n");
        exit(EXIT_FAILURE);
    }

    void *ptr = calloc(count, size);

    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}

// ============================================
// STEP 3: Load outgoing and incoming CSR
// ============================================

static Graph load_graph(const char *filename) {
    Graph g = {0};

    FILE *f = fopen(filename, "r");

    if (!f) {
        perror(filename);
        exit(EXIT_FAILURE);
    }

    if (fscanf(f, "%d %zu", &g.n, &g.m) != 2 ||
        g.n <= 0 || g.m == 0) {
        fprintf(stderr, "Invalid graph header\n");
        exit(EXIT_FAILURE);
    }

    g.out_row = allocate(
        (size_t)g.n + 1, sizeof(size_t)
    );

    g.in_row = allocate(
        (size_t)g.n + 1, sizeof(size_t)
    );

    // First pass: count both types of edges.

    for (size_t e = 0; e < g.m; e++) {
        int u, v;

        if (fscanf(f, "%d %d", &u, &v) != 2 ||
            u < 0 || u >= g.n ||
            v < 0 || v >= g.n) {
            fprintf(stderr, "Invalid edge list\n");
            exit(EXIT_FAILURE);
        }

        g.out_row[u + 1]++;
        g.in_row[v + 1]++;
    }

    // Prefix sums form the CSR row pointers.

    for (int v = 0; v < g.n; v++) {
        g.out_row[v + 1] += g.out_row[v];
        g.in_row[v + 1] += g.in_row[v];
    }

    g.out_col = allocate(g.m, sizeof(int));
    g.in_src = allocate(g.m, sizeof(int));

    size_t *out_pos = allocate(
        (size_t)g.n, sizeof(size_t)
    );

    size_t *in_pos = allocate(
        (size_t)g.n, sizeof(size_t)
    );

    // Second pass: fill both adjacency arrays.

    rewind(f);

    int second_n;
    size_t second_m;

    if (fscanf(f, "%d %zu",
               &second_n, &second_m) != 2 ||
        second_n != g.n || second_m != g.m) {
        fprintf(stderr, "Graph header mismatch\n");
        exit(EXIT_FAILURE);
    }

    for (size_t e = 0; e < g.m; e++) {
        int u, v;

        if (fscanf(f, "%d %d", &u, &v) != 2 ||
            u < 0 || u >= g.n ||
            v < 0 || v >= g.n) {
            fprintf(stderr, "Invalid edge\n");
            exit(EXIT_FAILURE);
        }

        size_t oi = g.out_row[u] + out_pos[u]++;
        size_t ii = g.in_row[v] + in_pos[v]++;

        if (oi >= g.out_row[u + 1] ||
            ii >= g.in_row[v + 1]) {
            fprintf(stderr, "CSR row overflow\n");
            exit(EXIT_FAILURE);
        }

        g.out_col[oi] = v;
        g.in_src[ii] = u;
    }

    fclose(f);
    free(out_pos);
    free(in_pos);

    return g;
}

// ============================================
// STEP 4: Serial push PageRank reference
// ============================================

static Stats serial_pagerank(
    const Graph *g,
    double *result
) {
    int n = g->n;

    double *a = allocate((size_t)n, sizeof(double));
    double *b = allocate((size_t)n, sizeof(double));

    double *old = a;
    double *next = b;

    for (int v = 0; v < n; v++) {
        old[v] = 1.0 / n;
    }

    Stats stats = {0};
    double start = omp_get_wtime();

    for (int iter = 1;
         iter <= MAX_ITERATIONS; iter++) {

        double dangling = 0.0;

        for (int u = 0; u < n; u++) {
            size_t degree =
                g->out_row[u + 1] - g->out_row[u];

            if (degree == 0)
                dangling += old[u];
        }

        double base =
            (1.0 - DAMPING) / n
            + DAMPING * dangling / n;

        for (int v = 0; v < n; v++)
            next[v] = base;

        // SERIAL PUSH:
        // Each source distributes its rank.

        for (int u = 0; u < n; u++) {
            size_t degree =
                g->out_row[u + 1] - g->out_row[u];

            if (degree == 0)
                continue;

            double contribution =
                DAMPING * old[u] / degree;

            for (size_t e = g->out_row[u];
                 e < g->out_row[u + 1]; e++) {

                int v = g->out_col[e];
                next[v] += contribution;
            }
        }

        double l1 = 0.0;

        for (int v = 0; v < n; v++) {
            l1 += fabs(next[v] - old[v]);
        }

        stats.iterations = iter;
        stats.residual = l1;

        double *temp = old;
        old = next;
        next = temp;

        if (l1 < TOLERANCE)
            break;
    }

    stats.elapsed = omp_get_wtime() - start;

    for (int v = 0; v < n; v++) {
        result[v] = old[v];
        stats.mass += old[v];
    }

    free(a);
    free(b);

    return stats;
}

// ============================================
// STEP 5: Parallel pull PageRank
// ============================================

static Stats parallel_pull_pagerank(
    const Graph *g,
    double *result,
    int threads
) {
    int n = g->n;

    double *a = allocate((size_t)n, sizeof(double));
    double *b = allocate((size_t)n, sizeof(double));

    double *old = a;
    double *next = b;

    for (int v = 0; v < n; v++) {
        old[v] = 1.0 / n;
    }

    Stats stats = {0};
    double start = omp_get_wtime();

    for (int iter = 1;
         iter <= MAX_ITERATIONS; iter++) {

        // A. Parallel dangling-mass reduction.

        double dangling = 0.0;

        #pragma omp parallel for num_threads(threads) \
            schedule(static) reduction(+:dangling)
        for (int u = 0; u < n; u++) {
            size_t degree =
                g->out_row[u + 1] - g->out_row[u];

            if (degree == 0)
                dangling += old[u];
        }

        double base =
            (1.0 - DAMPING) / n
            + DAMPING * dangling / n;

        // B. Parallel PULL:
        // Each thread processes destinations.
        //
        // Only one iteration writes next[v].
        // Therefore, atomics are unnecessary.

        #pragma omp parallel for num_threads(threads) \
            schedule(static)
        for (int v = 0; v < n; v++) {

            double incoming = 0.0;

            for (size_t e = g->in_row[v];
                 e < g->in_row[v + 1]; e++) {

                int u = g->in_src[e];

                size_t degree =
                    g->out_row[u + 1] - g->out_row[u];

                incoming += old[u] / degree;
            }

            next[v] = base + DAMPING * incoming;
        }

        // C. Parallel L1 convergence reduction.

        double l1 = 0.0;

        #pragma omp parallel for num_threads(threads) \
            schedule(static) reduction(+:l1)
        for (int v = 0; v < n; v++) {
            l1 += fabs(next[v] - old[v]);
        }

        stats.iterations = iter;
        stats.residual = l1;

        double *temp = old;
        old = next;
        next = temp;

        if (l1 < TOLERANCE)
            break;
    }

    stats.elapsed = omp_get_wtime() - start;

    for (int v = 0; v < n; v++) {
        result[v] = old[v];
        stats.mass += old[v];
    }

    free(a);
    free(b);

    return stats;
}

// ============================================
// STEP 6: Print solver statistics
// ============================================

static void print_stats(
    const char *name,
    Stats s
) {
    printf("\n===== %s =====\n", name);
    printf("Iterations: %d\n", s.iterations);
    printf("Final L1 difference: %.12e\n",
           s.residual);
    printf("Sum of ranks: %.12f\n", s.mass);
    printf("Execution time: %.6f seconds\n",
           s.elapsed);
}

// ============================================
// STEP 7: Compare complete rank vectors
// ============================================

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr,
            "Usage: %s graph_file threads\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    char *end = NULL;
    long requested = strtol(argv[2], &end, 10);

    if (end == argv[2] || *end != '\0' ||
        requested < 1 || requested > 64) {
        fprintf(stderr, "Invalid thread count\n");
        return EXIT_FAILURE;
    }

    int threads = (int)requested;

    omp_set_dynamic(0);

    Graph g = load_graph(argv[1]);

    double *serial = allocate(
        (size_t)g.n, sizeof(double)
    );

    double *parallel = allocate(
        (size_t)g.n, sizeof(double)
    );

    printf("===== LARGE PAGERANK COMPARISON =====\n");
    printf("Vertices: %d\n", g.n);
    printf("Edges: %zu\n", g.m);
    printf("Requested threads: %d\n", threads);

    Stats s = serial_pagerank(&g, serial);

    Stats p = parallel_pull_pagerank(
        &g, parallel, threads
    );

    print_stats("SERIAL PUSH", s);
    print_stats("OPENMP PULL", p);

    double max_difference = 0.0;
    double l1_difference = 0.0;

    for (int v = 0; v < g.n; v++) {
        double difference =
            fabs(serial[v] - parallel[v]);

        if (difference > max_difference)
            max_difference = difference;

        l1_difference += difference;
    }

    printf("\n===== FULL VECTOR VERIFICATION =====\n");

    printf("Maximum rank difference: %.12e\n",
           max_difference);

    printf("L1 difference between vectors: %.12e\n",
           l1_difference);

    printf("\n===== SELECTED VERTICES =====\n");

    int samples[] = {0, 1, 1000, 99999};

    for (int i = 0; i < 4; i++) {
        int v = samples[i];

        if (v < g.n) {
            printf(
                "Node %d | Serial %.12f | Pull %.12f\n",
                v, serial[v], parallel[v]
            );
        }
    }

    printf("\n===== PERFORMANCE =====\n");

    if (p.elapsed > 0.0) {
        printf(
            "Speedup vs serial: %.3fx\n",
            s.elapsed / p.elapsed
        );

        printf(
            "Parallel efficiency: %.2f%%\n",
            100.0 * s.elapsed /
            (p.elapsed * threads)
        );
    }

    int passed =
        s.residual < TOLERANCE &&
        p.residual < TOLERANCE &&
        fabs(s.mass - 1.0) < 1e-8 &&
        fabs(p.mass - 1.0) < 1e-8 &&
        max_difference < 1e-10 &&
        l1_difference < 1e-8;

    printf("\nCorrectness: %s\n",
           passed ? "PASS" : "FAIL");

    free(serial);
    free(parallel);

    free(g.out_row);
    free(g.out_col);
    free(g.in_row);
    free(g.in_src);

    return passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
