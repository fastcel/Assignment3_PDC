
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define DAMPING 0.85
#define TOLERANCE 1e-10
#define MAX_ITERATIONS 1000

// ========================================
// STEP 1: Graph and result structures
// ========================================

typedef struct {
    int n;
    size_t m;
    size_t *out_row;
    size_t *in_row;
    int *in_src;
} Graph;

typedef struct {
    int iterations;
    double residual;
    double mass;
    double seconds;
} Stats;

// ========================================
// STEP 2: Memory and error helpers
// ========================================

static void fail(const char *message) {
    fprintf(stderr, "ERROR: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *allocate(size_t count, size_t size) {
    if (count == 0 || size == 0 ||
        count > SIZE_MAX / size) {
        fail("Invalid allocation size");
    }

    void *ptr = calloc(count, size);

    if (!ptr)
        fail("Memory allocation failed");

    return ptr;
}

// ========================================
// STEP 3: Load graph into incoming CSR
// ========================================

static Graph load_graph(const char *filename) {
    Graph g = {0};

    FILE *f = fopen(filename, "r");

    if (!f) {
        perror(filename);
        exit(EXIT_FAILURE);
    }

    if (fscanf(f, "%d %zu", &g.n, &g.m) != 2 ||
        g.n <= 0 || g.m == 0) {
        fail("Invalid graph header");
    }

    g.out_row = allocate(
        (size_t)g.n + 1, sizeof(size_t)
    );

    g.in_row = allocate(
        (size_t)g.n + 1, sizeof(size_t)
    );

    // First pass: count degrees.

    for (size_t e = 0; e < g.m; e++) {
        int u, v;

        if (fscanf(f, "%d %d", &u, &v) != 2 ||
            u < 0 || u >= g.n ||
            v < 0 || v >= g.n) {
            fail("Invalid edge in first pass");
        }

        g.out_row[u + 1]++;
        g.in_row[v + 1]++;
    }

    // Convert counts to CSR offsets.

    for (int v = 0; v < g.n; v++) {
        g.out_row[v + 1] += g.out_row[v];
        g.in_row[v + 1] += g.in_row[v];
    }

    g.in_src = allocate(g.m, sizeof(int));

    size_t *cursor = allocate(
        (size_t)g.n, sizeof(size_t)
    );

    // Second pass: fill incoming CSR.

    rewind(f);

    int second_n;
    size_t second_m;

    if (fscanf(f, "%d %zu",
               &second_n, &second_m) != 2 ||
        second_n != g.n || second_m != g.m) {
        fail("Graph header mismatch");
    }

    for (size_t e = 0; e < g.m; e++) {
        int u, v;

        if (fscanf(f, "%d %d", &u, &v) != 2 ||
            u < 0 || u >= g.n ||
            v < 0 || v >= g.n) {
            fail("Invalid edge in second pass");
        }

        size_t position =
            g.in_row[v] + cursor[v]++;

        if (position >= g.in_row[v + 1])
            fail("Incoming CSR overflow");

        g.in_src[position] = u;
    }

    char extra;

    if (fscanf(f, " %c", &extra) == 1)
        fail("Unexpected content after edges");

    free(cursor);
    fclose(f);

    return g;
}

// ========================================
// STEP 4: Calculate one destination's rank
// ========================================

static inline double pull_vertex(
    const Graph *g,
    const double *old,
    int v,
    double base
) {
    double incoming = 0.0;

    for (size_t e = g->in_row[v];
         e < g->in_row[v + 1]; e++) {

        int u = g->in_src[e];

        size_t degree =
            g->out_row[u + 1] - g->out_row[u];

        incoming += old[u] / (double)degree;
    }

    return base + DAMPING * incoming;
}

// ========================================
// STEP 5: Serial or parallel pull solver
// ========================================

static Stats solve(
    const Graph *g,
    double *result,
    int threads,
    int parallel,
    size_t *first_iteration_work
) {
    int n = g->n;

    double *a = allocate(
        (size_t)n, sizeof(double)
    );

    double *b = allocate(
        (size_t)n, sizeof(double)
    );

    double *old = a;
    double *next = b;

    for (int v = 0; v < n; v++)
        old[v] = 1.0 / n;

    Stats stats = {0};

    double started = omp_get_wtime();

    for (int iter = 1;
         iter <= MAX_ITERATIONS; iter++) {

        // A. Compute dangling mass.
        // Keep this serial in both versions
        // to isolate scheduling of the pull loop.

        double dangling = 0.0;

        for (int u = 0; u < n; u++) {
            if (g->out_row[u] ==
                g->out_row[u + 1]) {
                dangling += old[u];
            }
        }

        double base =
            (1.0 - DAMPING) / n
            + DAMPING * dangling / n;

        // B. Compute new PageRanks.

        if (parallel) {

            #pragma omp parallel num_threads(threads)
            {
                int tid = omp_get_thread_num();
                size_t local_edges = 0;

                #pragma omp for schedule(runtime)
                for (int v = 0; v < n; v++) {

                    next[v] = pull_vertex(
                        g, old, v, base
                    );

                    // Record work distribution
                    // during the first iteration.

                    if (iter == 1) {
                        local_edges +=
                            g->in_row[v + 1]
                            - g->in_row[v];
                    }
                }

                if (iter == 1) {
                    first_iteration_work[tid] =
                        local_edges;
                }
            }

        } else {

            for (int v = 0; v < n; v++) {
                next[v] = pull_vertex(
                    g, old, v, base
                );
            }
        }

        // C. Same serial convergence check
        // for both implementations.

        double l1 = 0.0;

        for (int v = 0; v < n; v++) {
            l1 += fabs(next[v] - old[v]);
        }

        stats.iterations = iter;
        stats.residual = l1;

        // D. Swap the rank vectors.

        double *temp = old;
        old = next;
        next = temp;

        if (l1 < TOLERANCE)
            break;
    }

    stats.seconds = omp_get_wtime() - started;

    for (int v = 0; v < n; v++) {
        result[v] = old[v];
        stats.mass += old[v];
    }

    free(a);
    free(b);

    return stats;
}

// ========================================
// STEP 6: Select OpenMP schedule
// ========================================

static void configure_schedule(
    const char *name,
    int chunk
) {
    if (strcmp(name, "static") == 0) {
        omp_set_schedule(omp_sched_static, chunk);
    } else if (strcmp(name, "dynamic") == 0) {
        omp_set_schedule(omp_sched_dynamic, chunk);
    } else if (strcmp(name, "guided") == 0) {
        omp_set_schedule(omp_sched_guided, chunk);
    } else {
        fail("Use static, dynamic or guided");
    }
}

// ========================================
// STEP 7: Main and verification
// ========================================

int main(int argc, char **argv) {

    if (argc != 5) {
        fprintf(stderr,
            "Usage: %s graph_file threads "
            "static|dynamic|guided chunk\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    char *end = NULL;
    long requested = strtol(argv[2], &end, 10);

    if (end == argv[2] || *end != '\0' ||
        requested < 1 || requested > 64) {
        fail("Invalid thread count");
    }

    int threads = (int)requested;

    char *chunk_end = NULL;

    long requested_chunk = strtol(
        argv[4], &chunk_end, 10
    );

    if (
        chunk_end == argv[4] ||
        *chunk_end != '\0' ||
        requested_chunk < 1 ||
        requested_chunk > 1000000
    ) {
        fail("Invalid chunk size");
    }

    int chunk = (int)requested_chunk;


    omp_set_dynamic(0);

    Graph g = load_graph(argv[1]);

    configure_schedule(argv[3], chunk);

    double *reference = allocate(
        (size_t)g.n, sizeof(double)
    );

    double *parallel = allocate(
        (size_t)g.n, sizeof(double)
    );

    size_t *work = allocate(
        (size_t)threads, sizeof(size_t)
    );

    Stats serial_stats = solve(
        &g, reference, 1, 0, NULL
    );

    Stats parallel_stats = solve(
        &g, parallel, threads, 1, work
    );

    // Compare the complete final vectors.

    double max_difference = 0.0;
    double l1_difference = 0.0;

    for (int v = 0; v < g.n; v++) {

        double diff =
            fabs(reference[v] - parallel[v]);

        if (diff > max_difference)
            max_difference = diff;

        l1_difference += diff;
    }

    // Examine first-iteration work distribution.

    size_t minimum = SIZE_MAX;
    size_t maximum = 0;
    size_t total_work = 0;

    printf("===== PAGERANK SCHEDULING =====\n");
    printf("Vertices: %d\n", g.n);
    printf("Edges: %zu\n", g.m);
    printf("Threads: %d\n", threads);
    printf("Schedule: %s\n", argv[3]);

    printf("\n===== WORK DISTRIBUTION =====\n");

    for (int t = 0; t < threads; t++) {

        printf("Thread %d: %zu incoming edges\n",
               t, work[t]);

        if (work[t] < minimum)
            minimum = work[t];

        if (work[t] > maximum)
            maximum = work[t];

        total_work += work[t];
    }

    printf("Total assigned edges: %zu\n",
           total_work);

    if (minimum > 0) {
        printf("Max/min edge imbalance: %.3fx\n",
               (double)maximum / minimum);
    }

    printf("\n===== SOLVER RESULTS =====\n");

    printf("Serial pull iterations: %d\n",
           serial_stats.iterations);

    printf("OpenMP pull iterations: %d\n",
           parallel_stats.iterations);

    printf("Serial pull time: %.6f seconds\n",
           serial_stats.seconds);

    printf("OpenMP pull time: %.6f seconds\n",
           parallel_stats.seconds);

    printf("Speedup vs serial pull: %.3fx\n",
           serial_stats.seconds /
           parallel_stats.seconds);

    printf("Final residual: %.12e\n",
           parallel_stats.residual);

    printf("Sum of ranks: %.12f\n",
           parallel_stats.mass);

    printf("\n===== VERIFICATION =====\n");

    printf("Maximum rank difference: %.12e\n",
           max_difference);

    printf("L1 vector difference: %.12e\n",
           l1_difference);

    int passed =
        serial_stats.residual < TOLERANCE &&
        parallel_stats.residual < TOLERANCE &&
        fabs(serial_stats.mass - 1.0) < 1e-8 &&
        fabs(parallel_stats.mass - 1.0) < 1e-8 &&
        max_difference < 1e-10 &&
        l1_difference < 1e-8 &&
        total_work == g.m;

    printf("Correctness: %s\n",
           passed ? "PASS" : "FAIL");

    free(reference);
    free(parallel);
    free(work);

    free(g.out_row);
    free(g.in_row);
    free(g.in_src);

    return passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
