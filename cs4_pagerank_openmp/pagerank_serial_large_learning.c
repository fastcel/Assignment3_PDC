
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define DAMPING 0.85
#define TOLERANCE 1e-10
#define MAX_ITERATIONS 1000

// =============================================
// STEP 1: Outgoing CSR representation
// =============================================

typedef struct {
    int n;
    size_t m;
    size_t *row;
    int *col;
} Graph;


// =============================================
// STEP 2: Error handling and memory helpers
// =============================================

static void fail(const char *message) {
    fprintf(stderr, "ERROR: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *checked_calloc(
    size_t count,
    size_t item_size
) {
    if (count == 0 || item_size == 0 ||
        count > SIZE_MAX / item_size) {
        fail("Invalid allocation size");
    }

    void *p = calloc(count, item_size);

    if (p == NULL)
        fail("Memory allocation failed");

    return p;
}


// =============================================
// STEP 3: Wall-clock timer
// =============================================

static double seconds_now(void) {
    struct timespec t;

    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0)
        fail("clock_gettime failed");

    return (double)t.tv_sec
         + 1e-9 * (double)t.tv_nsec;
}


// =============================================
// STEP 4: Load the graph into outgoing CSR
// =============================================

static Graph load_outgoing_csr(
    const char *path
) {
    FILE *f = fopen(path, "r");

    if (f == NULL) {
        perror(path);
        exit(EXIT_FAILURE);
    }

    Graph g = {0};

    if (fscanf(f, "%d %zu", &g.n, &g.m) != 2 ||
        g.n <= 0 || g.m == 0) {

        fail("Invalid graph header");
    }

    g.row = checked_calloc(
        (size_t)g.n + 1,
        sizeof(*g.row)
    );

    // PASS 1: Count edges leaving each node.

    for (size_t e = 0; e < g.m; ++e) {
        int u, v;

        if (fscanf(f, "%d %d", &u, &v) != 2)
            fail("Truncated edge list");

        if (u < 0 || u >= g.n ||
            v < 0 || v >= g.n)
            fail("Invalid vertex ID");

        g.row[u + 1]++;
    }

    // Convert counts into CSR offsets.

    for (int u = 0; u < g.n; ++u)
        g.row[u + 1] += g.row[u];

    if (g.row[g.n] != g.m)
        fail("CSR edge count mismatch");

    g.col = checked_calloc(
        g.m, sizeof(*g.col)
    );

    size_t *cursor = checked_calloc(
        (size_t)g.n, sizeof(*cursor)
    );

    // PASS 2: Fill adjacency array.

    rewind(f);

    int header_n;
    size_t header_m;

    if (fscanf(f, "%d %zu",
               &header_n, &header_m) != 2 ||
        header_n != g.n ||
        header_m != g.m) {

        fail("Graph header mismatch");
    }

    for (size_t e = 0; e < g.m; ++e) {
        int u, v;

        if (fscanf(f, "%d %d", &u, &v) != 2)
            fail("Invalid edge on second pass");

        if (u < 0 || u >= g.n ||
            v < 0 || v >= g.n)
            fail("Invalid vertex on second pass");

        size_t pos =
            g.row[u] + cursor[u]++;

        if (pos >= g.row[u + 1])
            fail("CSR row overflow");

        g.col[pos] = v;
    }

    char extra;

    if (fscanf(f, " %c", &extra) == 1)
        fail("Unexpected content after edges");

    free(cursor);
    fclose(f);

    return g;
}


// =============================================
// STEP 5: Main PageRank program
// =============================================

int main(int argc, char **argv) {

    if (argc != 2) {
        fprintf(stderr,
            "Usage: %s graph_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    Graph graph =
        load_outgoing_csr(argv[1]);

    size_t dangling_count = 0;

    for (int u = 0; u < graph.n; ++u) {
        if (graph.row[u] == graph.row[u + 1])
            dangling_count++;
    }

    printf("===== LARGE SERIAL PAGERANK =====\n");
    printf("Vertices: %d\n", graph.n);
    printf("Edges: %zu\n", graph.m);
    printf("Dangling vertices: %zu\n",
           dangling_count);
    printf("Damping: %.2f\n", DAMPING);
    printf("Tolerance: %.1e\n", TOLERANCE);

    // -----------------------------------------
    // A. Allocate two rank vectors
    // -----------------------------------------

    double *a = checked_calloc(
        (size_t)graph.n, sizeof(double)
    );

    double *b = checked_calloc(
        (size_t)graph.n, sizeof(double)
    );

    double *rank = a;
    double *next = b;

    // Initialize each rank to 1/N.

    for (int v = 0; v < graph.n; ++v)
        rank[v] = 1.0 / (double)graph.n;

    int performed = 0;
    int converged = 0;

    double last_l1 = 0.0;

    // Timer excludes graph loading.
    double started = seconds_now();

    printf("\n===== ITERATION PROGRESS =====\n");

    // =========================================
    // STEP 6: Iterate until convergence
    // =========================================

    for (int iter = 1;
         iter <= MAX_ITERATIONS;
         ++iter) {

        // -------------------------------------
        // A. Sum the dangling-node mass
        // -------------------------------------

        double dangling_mass = 0.0;

        for (int u = 0; u < graph.n; ++u) {
            if (graph.row[u] ==
                graph.row[u + 1]) {

                dangling_mass += rank[u];
            }
        }

        // -------------------------------------
        // B. Initialize all destination ranks
        // -------------------------------------

        double base =
            (1.0 - DAMPING) / (double)graph.n
            + DAMPING * dangling_mass
              / (double)graph.n;

        for (int v = 0; v < graph.n; ++v)
            next[v] = base;

        // -------------------------------------
        // C. Serial PUSH: distribute rank
        // -------------------------------------

        for (int u = 0; u < graph.n; ++u) {

            size_t degree =
                graph.row[u + 1] - graph.row[u];

            if (degree == 0)
                continue;

            double contribution =
                DAMPING * rank[u] / (double)degree;

            for (size_t e = graph.row[u];
                 e < graph.row[u + 1];
                 ++e) {

                int v = graph.col[e];

                next[v] += contribution;
            }
        }

        // -------------------------------------
        // D. Calculate convergence difference
        // -------------------------------------

        double l1 = 0.0;
        double mass = 0.0;

        for (int v = 0; v < graph.n; ++v) {

            l1 += fabs(next[v] - rank[v]);
            mass += next[v];
        }

        // Rank mass must remain close to 1.

        if (!isfinite(l1) ||
            !isfinite(mass) ||
            fabs(mass - 1.0) > 1e-8) {

            fail("PageRank numerical check failed");
        }

        performed = iter;
        last_l1 = l1;

        if (iter <= 5 || iter % 10 == 0) {
            printf(
                "Iteration %3d | "
                "L1 difference: %.12e\n",
                iter, l1
            );
        }

        // -------------------------------------
        // E. Swap old/new rank arrays
        // -------------------------------------

        double *temp = rank;
        rank = next;
        next = temp;

        // -------------------------------------
        // F. Check stopping condition
        // -------------------------------------

        if (l1 < TOLERANCE) {
            converged = 1;
            break;
        }
    }

    double elapsed = seconds_now() - started;

    // =========================================
    // STEP 7: Final verification
    // =========================================

    double total = 0.0;

    for (int v = 0; v < graph.n; ++v)
        total += rank[v];

    printf("\n===== FINAL RESULTS =====\n");

    printf("Iterations: %d\n", performed);

    printf("Final L1 difference: %.12e\n",
           last_l1);

    printf("Sum of ranks: %.12f\n", total);

    printf("Execution time: %.6f seconds\n",
           elapsed);

    printf("Converged: %s\n",
           converged ? "YES" : "NO");

    // Display sample nodes for later comparisons.

    printf("\n===== SELECTED VERTICES =====\n");

    int sample[] = {0, 1, 1000, 99999};

    for (size_t i = 0;
         i < sizeof(sample) / sizeof(sample[0]);
         ++i) {

        if (sample[i] < graph.n) {
            printf(
                "Node %d: %.12f\n",
                sample[i],
                rank[sample[i]]
            );
        }
    }

    int passed =
        converged &&
        fabs(total - 1.0) < 1e-8;

    printf("\nCorrectness: %s\n",
           passed ? "PASS" : "FAIL");

    // =========================================
    // STEP 8: Free allocated memory
    // =========================================

    free(a);
    free(b);

    free(graph.row);
    free(graph.col);

    return passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
