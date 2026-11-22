
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>

// =============================================
// STEP 1: Define the CSR graph structure
// =============================================

typedef struct {
    int n;
    size_t m;

    // Outgoing CSR for PUSH PageRank
    size_t *out_row;
    int *out_col;

    // Incoming CSR for PULL PageRank
    size_t *in_row;
    int *in_src;
} Graph;


// =============================================
// STEP 2: Memory allocation helper
// =============================================

static void *allocate(size_t count, size_t size) {

    void *ptr = calloc(count, size);

    if (ptr == NULL) {
        fprintf(stderr,
                "ERROR: Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}


// =============================================
// STEP 3: Read the graph header
// =============================================

static void read_header(
    FILE *file,
    int *n,
    size_t *m
) {
    if (fscanf(file, "%d %zu", n, m) != 2) {
        fprintf(stderr, "ERROR: Invalid graph header.\n");
        exit(EXIT_FAILURE);
    }

    if (*n <= 0 || *m == 0 ||
        *m > (size_t)-1 / sizeof(int)) {

        fprintf(stderr,
                "ERROR: Invalid graph dimensions.\n");
        exit(EXIT_FAILURE);
    }
}


// =============================================
// STEP 4: Load both CSR representations
// =============================================

static Graph load_graph(const char *filename) {

    Graph g = {0};

    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        perror("Could not open graph file");
        exit(EXIT_FAILURE);
    }

    read_header(file, &g.n, &g.m);

    printf("===== LOADING GRAPH =====\n");
    printf("Vertices: %d\n", g.n);
    printf("Edges: %zu\n", g.m);

    // Allocate row pointers.
    // We need N+1 entries to mark row ends.

    g.out_row = allocate(
        (size_t)g.n + 1, sizeof(size_t)
    );

    g.in_row = allocate(
        (size_t)g.n + 1, sizeof(size_t)
    );

    // -----------------------------------------
    // PASS 1: Count outgoing/incoming edges
    // -----------------------------------------

    for (size_t e = 0; e < g.m; e++) {

        int u, v;

        if (fscanf(file, "%d %d", &u, &v) != 2) {
            fprintf(stderr,
                    "ERROR: Invalid edge at index %zu.\n",
                    e);
            exit(EXIT_FAILURE);
        }

        if (u < 0 || u >= g.n ||
            v < 0 || v >= g.n) {

            fprintf(stderr,
                    "ERROR: Invalid vertex ID.\n");
            exit(EXIT_FAILURE);
        }

        g.out_row[u + 1]++;
        g.in_row[v + 1]++;
    }

    // Check for unexpected additional edges.

    int extra_u, extra_v;

    if (fscanf(file, "%d %d",
               &extra_u, &extra_v) == 2) {

        fprintf(stderr,
                "ERROR: More edges than header specifies.\n");
        exit(EXIT_FAILURE);
    }

    // -----------------------------------------
    // PASS 2: Convert counts to prefix sums
    // -----------------------------------------

    for (int v = 0; v < g.n; v++) {

        g.out_row[v + 1] += g.out_row[v];
        g.in_row[v + 1] += g.in_row[v];
    }

    // -----------------------------------------
    // PASS 3: Allocate adjacency arrays
    // -----------------------------------------

    g.out_col = allocate(g.m, sizeof(int));
    g.in_src = allocate(g.m, sizeof(int));

    size_t *out_pos = allocate(
        (size_t)g.n, sizeof(size_t)
    );

    size_t *in_pos = allocate(
        (size_t)g.n, sizeof(size_t)
    );

    // -----------------------------------------
    // PASS 4: Reread edge list and fill CSR
    // -----------------------------------------

    rewind(file);

    int second_n;
    size_t second_m;

    read_header(file, &second_n, &second_m);

    if (second_n != g.n || second_m != g.m) {
        fprintf(stderr,
                "ERROR: Graph header changed.\n");
        exit(EXIT_FAILURE);
    }

    for (size_t e = 0; e < g.m; e++) {

        int u, v;

        if (fscanf(file, "%d %d", &u, &v) != 2) {
            fprintf(stderr,
                    "ERROR: Invalid edge on second pass.\n");
            exit(EXIT_FAILURE);
        }

        if (u < 0 || u >= g.n ||
            v < 0 || v >= g.n) {

            fprintf(stderr,
                    "ERROR: Invalid vertex on second pass.\n");
            exit(EXIT_FAILURE);
        }

        // Store destination in source row.
        size_t out_index =
            g.out_row[u] + out_pos[u]++;

        // Store source in destination row.
        size_t in_index =
            g.in_row[v] + in_pos[v]++;

        if (out_index >= g.out_row[u + 1] ||
            in_index >= g.in_row[v + 1]) {

            fprintf(stderr,
                    "ERROR: CSR row overflow.\n");
            exit(EXIT_FAILURE);
        }

        g.out_col[out_index] = v;
        g.in_src[in_index] = u;
    }

    fclose(file);

    free(out_pos);
    free(in_pos);

    return g;
}


// =============================================
// STEP 5: Print outgoing/incoming neighbours
// =============================================

static void print_vertex(
    const Graph *g,
    int v,
    int max_neighbors
) {
    size_t outdegree =
        g->out_row[v + 1] - g->out_row[v];

    size_t indegree =
        g->in_row[v + 1] - g->in_row[v];

    printf("\nNode %d\n", v);

    printf("Outdegree: %zu\n", outdegree);
    printf("Indegree: %zu\n", indegree);

    printf("First outgoing neighbours:");

    for (size_t e = g->out_row[v];
         e < g->out_row[v + 1] &&
         e < g->out_row[v] + (size_t)max_neighbors;
         e++) {

        printf(" %d", g->out_col[e]);
    }

    if (outdegree == 0) {
        printf(" NONE");
    }

    printf("\n");

    printf("First incoming neighbours:");

    for (size_t e = g->in_row[v];
         e < g->in_row[v + 1] &&
         e < g->in_row[v] + (size_t)max_neighbors;
         e++) {

        printf(" %d", g->in_src[e]);
    }

    if (indegree == 0) {
        printf(" NONE");
    }

    printf("\n");
}


// =============================================
// STEP 6: Verify CSR consistency
// =============================================

static int verify_graph(const Graph *g) {

    size_t total_out = 0;
    size_t total_in = 0;

    size_t dangling = 0;
    size_t max_outdegree = 0;
    size_t max_indegree = 0;

    for (int v = 0; v < g->n; v++) {

        size_t outdegree =
            g->out_row[v + 1] - g->out_row[v];

        size_t indegree =
            g->in_row[v + 1] - g->in_row[v];

        total_out += outdegree;
        total_in += indegree;

        if (outdegree == 0)
            dangling++;

        if (outdegree > max_outdegree)
            max_outdegree = outdegree;

        if (indegree > max_indegree)
            max_indegree = indegree;
    }

    printf("\n===== GRAPH STATISTICS =====\n");

    printf("Total outgoing entries: %zu\n",
           total_out);

    printf("Total incoming entries: %zu\n",
           total_in);

    printf("Dangling vertices: %zu\n",
           dangling);

    printf("Maximum outdegree: %zu\n",
           max_outdegree);

    printf("Maximum indegree: %zu\n",
           max_indegree);

    int passed =
        total_out == g->m &&
        total_in == g->m &&
        g->out_row[0] == 0 &&
        g->in_row[0] == 0 &&
        g->out_row[g->n] == g->m &&
        g->in_row[g->n] == g->m;

    printf("\nCSR verification: %s\n",
           passed ? "PASS" : "FAIL");

    return passed;
}


// =============================================
// STEP 7: Release graph memory
// =============================================

static void free_graph(Graph *g) {

    free(g->out_row);
    free(g->out_col);

    free(g->in_row);
    free(g->in_src);
}


// =============================================
// STEP 8: Main
// =============================================

int main(int argc, char **argv) {

    if (argc != 2) {
        fprintf(stderr,
                "Usage: %s graph_file\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    Graph graph = load_graph(argv[1]);

    print_vertex(&graph, 0, 5);
    print_vertex(&graph, 1000, 5);
    print_vertex(&graph, 99999, 5);

    int passed = verify_graph(&graph);

    free_graph(&graph);

    return passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
