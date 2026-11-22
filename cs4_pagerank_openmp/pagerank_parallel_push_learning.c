
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#define DAMPING 0.85
#define TOLERANCE 1e-10
#define MAX_ITERATIONS 1000

// ============================================
// STEP 1: Graph and solver structures
// ============================================

typedef struct {
    int n;
    size_t m;
    size_t *row;
    int *col;
} Graph;

typedef struct {
    int iterations;
    double residual;
    double mass;
    double elapsed;
} Stats;

// ============================================
// STEP 2: Memory allocation
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
// STEP 3: Load outgoing CSR
// ============================================

static Graph load_graph(const char *filename) {

    Graph g = {0};

    FILE *f = fopen(filename, "r");

    if (f == NULL) {
        perror(filename);
        exit(EXIT_FAILURE);
    }

    if (fscanf(f, "%d %zu", &g.n, &g.m) != 2 ||
        g.n <= 0 || g.m == 0) {
        fprintf(stderr, "Invalid graph header\n");
        exit(EXIT_FAILURE);
    }

    g.row = allocate(
        (size_t)g.n + 1, sizeof(size_t)
    );

    // Pass 1: Count outgoing edges.

    for (size_t e = 0; e < g.m; e++) {

        int u, v;

        if (fscanf(f, "%d %d", &u, &v) != 2 ||
            u < 0 || u >= g.n ||
            v < 0 || v >= g.n) {
            fprintf(stderr, "Invalid edge list\n");
            exit(EXIT_FAILURE);
        }

        g.row[u + 1]++;
    }

    for (int u = 0; u < g.n; u++) {
        g.row[u + 1] += g.row[u];
    }

    g.col = allocate(g.m, sizeof(int));

    size_t *cursor = allocate(
        (size_t)g.n, sizeof(size_t)
    );

    // Pass 2: Fill adjacency array.

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

        size_t index = g.row[u] + cursor[u]++;

        if (index >= g.row[u + 1]) {
            fprintf(stderr, "CSR row overflow\n");
            exit(EXIT_FAILURE);
        }

        g.col[index] = v;
    }

    fclose(f);
    free(cursor);

    return g;
}

// ============================================
// STEP 4: Serial and parallel PageRank solver
// ============================================

static Stats solve_pagerank(
    const Graph *g,
    double *result,
    int threads,
    int use_parallel
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

    double started = omp_get_wtime();

    // ========================================
    // STEP 5: Repeat until convergence
    // ========================================

    for (int iter = 1;
         iter <= MAX_ITERATIONS;
         iter++) {

        double dangling = 0.0;

        // ------------------------------------
        // A. Find dangling rank mass
        // ------------------------------------

        if (use_parallel) {

            #pragma omp parallel for num_threads(threads) \
                schedule(static) reduction(+:dangling)
            for (int u = 0; u < n; u++) {

                size_t degree =
                    g->row[u + 1] - g->row[u];

                if (degree == 0) {
                    dangling += old[u];
                }
            }

        } else {

            for (int u = 0; u < n; u++) {

                size_t degree =
                    g->row[u + 1] - g->row[u];

                if (degree == 0) {
                    dangling += old[u];
                }
            }
        }

        double base =
            (1.0 - DAMPING) / n
            + DAMPING * dangling / n;

        // ------------------------------------
        // B. Initialize next rank array
        // ------------------------------------

        if (use_parallel) {

            #pragma omp parallel for num_threads(threads) \
                schedule(static)
            for (int v = 0; v < n; v++) {
                next[v] = base;
            }

        } else {

            for (int v = 0; v < n; v++) {
                next[v] = base;
            }
        }

        // ------------------------------------
        // C. PUSH rank contributions
        // ------------------------------------

        if (use_parallel) {

            // Multiple source threads can
            // write to the same destination.

            #pragma omp parallel for num_threads(threads) \
                schedule(static)
            for (int u = 0; u < n; u++) {

                size_t degree =
                    g->row[u + 1] - g->row[u];

                if (degree == 0)
                    continue;

                double contribution =
                    DAMPING * old[u] / degree;

                for (size_t e = g->row[u];
                     e < g->row[u + 1]; e++) {

                    int v = g->col[e];

                    // Protect the shared update.
                    #pragma omp atomic update
                    next[v] += contribution;
                }
            }

        } else {

            // Pure serial PUSH baseline.

            for (int u = 0; u < n; u++) {

                size_t degree =
                    g->row[u + 1] - g->row[u];

                if (degree == 0)
                    continue;

                double contribution =
                    DAMPING * old[u] / degree;

                for (size_t e = g->row[u];
                     e < g->row[u + 1]; e++) {

                    int v = g->col[e];
                    next[v] += contribution;
                }
            }
        }

        // ------------------------------------
        // D. Calculate L1 convergence error
        // ------------------------------------

        double l1 = 0.0;

        if (use_parallel) {

            #pragma omp parallel for num_threads(threads) \
                schedule(static) reduction(+:l1)
            for (int v = 0; v < n; v++) {
                l1 += fabs(next[v] - old[v]);
            }

        } else {

            for (int v = 0; v < n; v++) {
                l1 += fabs(next[v] - old[v]);
            }
        }

        if (!isfinite(l1)) {
            fprintf(stderr, "Non-finite PageRank residual\n");
            exit(EXIT_FAILURE);
        }

        stats.iterations = iter;
        stats.residual = l1;

        // ------------------------------------
        // E. Swap old and next
        // ------------------------------------

        double *temp = old;
        old = next;
        next = temp;

        if (l1 < TOLERANCE) {
            break;
        }
    }

    stats.elapsed = omp_get_wtime() - started;

    // ========================================
    // STEP 6: Store final results
    // ========================================

    for (int v = 0; v < n; v++) {

        result[v] = old[v];
        stats.mass += old[v];
    }

    free(a);
    free(b);

    return stats;
}

// ============================================
// STEP 7: Print statistics
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
// STEP 8: Compare complete rank vectors
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

    printf("===== PARALLEL PUSH COMPARISON =====\n");
    printf("Vertices: %d\n", g.n);
    printf("Edges: %zu\n", g.m);
    printf("Requested threads: %d\n", threads);

    Stats s = solve_pagerank(
        &g, serial, 1, 0
    );

    Stats p = solve_pagerank(
        &g, parallel, threads, 1
    );

    print_stats("SERIAL PUSH", s);
    print_stats("OPENMP ATOMIC PUSH", p);

    double max_difference = 0.0;
    double l1_difference = 0.0;

    for (int v = 0; v < g.n; v++) {

        double difference =
            fabs(serial[v] - parallel[v]);

        if (difference > max_difference) {
            max_difference = difference;
        }

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
                "Node %d | Serial %.12f | Atomic %.12f\n",
                v, serial[v], parallel[v]
            );
        }
    }

    printf("\n===== PERFORMANCE =====\n");

    if (p.elapsed > 0.0) {
        double speedup = s.elapsed / p.elapsed;

        printf("Speedup vs serial: %.3fx\n",
               speedup);

        printf("Parallel efficiency: %.2f%%\n",
               100.0 * speedup / threads);
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

    free(g.row);
    free(g.col);

    return passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
