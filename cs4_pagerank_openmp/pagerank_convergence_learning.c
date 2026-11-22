
#include <stdio.h>
#include <math.h>

#define N 5
#define E 6
#define MAX_ITERATIONS 1000

int main(void) {

    // =====================================
    // STEP 1: Directed graph stored in CSR
    // =====================================

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
    const double epsilon = 1e-10;

    // Two separate arrays:
    // old ranks and newly calculated ranks.
    double rank_a[N];
    double rank_b[N];

    double *rank = rank_a;
    double *next_rank = rank_b;

    // =====================================
    // STEP 2: Initialize uniform ranks
    // =====================================

    for (int v = 0; v < N; v++) {
        rank[v] = 1.0 / N;
    }

    int iterations = 0;
    double final_difference = 0.0;
    int converged = 0;

    printf("===== SERIAL PAGERANK CONVERGENCE =====\n");

    printf("Nodes: %d\n", N);
    printf("Edges: %d\n", E);
    printf("Damping: %.2f\n", damping);
    printf("Tolerance: %.2e\n", epsilon);

    printf("\n===== ITERATION PROGRESS =====\n");

    // =====================================
    // STEP 3: Repeated PageRank iterations
    // =====================================

    for (int iter = 1;
         iter <= MAX_ITERATIONS;
         iter++) {

        // ---------------------------------
        // A. Find total dangling rank
        // ---------------------------------

        double dangling_mass = 0.0;

        for (int u = 0; u < N; u++) {

            int degree =
                row_ptr[u + 1] - row_ptr[u];

            if (degree == 0) {
                dangling_mass += rank[u];
            }
        }

        // ---------------------------------
        // B. Initialize new ranks
        // ---------------------------------

        double base =
            (1.0 - damping) / N
            + damping * dangling_mass / N;

        for (int v = 0; v < N; v++) {
            next_rank[v] = base;
        }

        // ---------------------------------
        // C. Distribute contributions
        // ---------------------------------

        for (int u = 0; u < N; u++) {

            int degree =
                row_ptr[u + 1] - row_ptr[u];

            if (degree == 0) {
                continue;
            }

            double contribution =
                damping * rank[u] / degree;

            for (int e = row_ptr[u];
                 e < row_ptr[u + 1];
                 e++) {

                int v = col_idx[e];

                next_rank[v] += contribution;
            }
        }

        // ---------------------------------
        // D. Calculate L1 difference
        // ---------------------------------

        double difference = 0.0;

        for (int v = 0; v < N; v++) {

            difference += fabs(
                next_rank[v] - rank[v]
            );
        }

        iterations = iter;
        final_difference = difference;

        if (iter <= 5 || iter % 5 == 0) {

            printf(
                "Iteration %3d | L1 difference: %.12e\n",
                iter, difference
            );
        }

        // ---------------------------------
        // E. Swap the rank arrays
        // ---------------------------------

        double *temp = rank;
        rank = next_rank;
        next_rank = temp;

        // ---------------------------------
        // F. Check convergence
        // ---------------------------------

        if (difference < epsilon) {

            converged = 1;
            break;
        }
    }

    // =====================================
    // STEP 4: Display final node ranks
    // =====================================

    printf("\n===== FINAL PAGE RANKS =====\n");

    double total = 0.0;

    for (int v = 0; v < N; v++) {

        printf(
            "Node %d: %.12f\n",
            v, rank[v]
        );

        total += rank[v];
    }

    printf("\n===== VERIFICATION =====\n");

    printf("Iterations performed: %d\n",
           iterations);

    printf("Final L1 difference: %.12e\n",
           final_difference);

    printf("Sum of final ranks: %.12f\n",
           total);

    printf("Converged: %s\n",
           converged ? "YES" : "NO");

    int passed =
        converged &&
        fabs(total - 1.0) < 1e-10;

    printf("Correctness: %s\n",
           passed ? "PASS" : "FAIL");

    return passed ? 0 : 1;
}
