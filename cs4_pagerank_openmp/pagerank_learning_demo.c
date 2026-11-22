
#include <stdio.h>
#include <math.h>

#define N 5
#define E 6

int main(void) {

    // ====================================
    // STEP 1: Small directed graph in CSR
    // ====================================

    const int row_ptr[N + 1] = {
        0, 2, 3, 5, 6, 6
    };

    const int col_idx[E] = {
        1, 2,
        2,
        0, 3,
        2
    };

    const double damping = 0.85;

    double rank[N];
    double next_rank[N];

    printf("===== CSR GRAPH =====\n");

    for (int u = 0; u < N; u++) {

        int degree = row_ptr[u + 1] - row_ptr[u];

        printf("Node %d | Outdegree %d | Neighbors:",
               u, degree);

        if (degree == 0) {
            printf(" NONE (dangling node)");
        }

        for (int e = row_ptr[u];
             e < row_ptr[u + 1]; e++) {

            printf(" %d", col_idx[e]);
        }

        printf("\n");
    }

    // ====================================
    // STEP 2: Initial uniform ranks
    // ====================================

    for (int u = 0; u < N; u++) {
        rank[u] = 1.0 / N;
    }

    // ====================================
    // STEP 3: Calculate dangling mass
    // ====================================

    double dangling_mass = 0.0;

    for (int u = 0; u < N; u++) {

        int degree = row_ptr[u + 1] - row_ptr[u];

        if (degree == 0) {
            dangling_mass += rank[u];
        }
    }

    // ====================================
    // STEP 4: Initialize next ranks
    // ====================================

    double base = (1.0 - damping) / N;

    double dangling_share =
        damping * dangling_mass / N;

    for (int v = 0; v < N; v++) {
        next_rank[v] = base + dangling_share;
    }

    // ====================================
    // STEP 5: Distribute rank along edges
    // ====================================

    for (int u = 0; u < N; u++) {

        int degree = row_ptr[u + 1] - row_ptr[u];

        if (degree == 0) {
            continue;
        }

        double contribution =
            damping * rank[u] / degree;

        for (int e = row_ptr[u];
             e < row_ptr[u + 1]; e++) {

            int v = col_idx[e];

            next_rank[v] += contribution;
        }
    }

    // ====================================
    // STEP 6: Calculate L1 difference
    // ====================================

    double difference = 0.0;
    double total = 0.0;

    printf("\n===== PAGERANK RESULTS =====\n");

    for (int v = 0; v < N; v++) {

        difference += fabs(next_rank[v] - rank[v]);
        total += next_rank[v];

        printf(
            "Node %d | Old: %.6f | New: %.6f\n",
            v, rank[v], next_rank[v]
        );
    }

    printf("\nDangling mass: %.6f\n",
           dangling_mass);

    printf("L1 difference: %.6f\n",
           difference);

    printf("Sum of new ranks: %.6f\n",
           total);

    int passed = fabs(total - 1.0) < 1e-12;

    printf("Correctness: %s\n",
           passed ? "PASS" : "FAIL");

    return passed ? 0 : 1;
}
