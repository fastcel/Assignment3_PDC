
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define PATHS 8

int main(void) {

    // Illustrative discounted option payoffs.
    const double payoffs[PATHS] = {
        0.0, 0.0, 4.0, 10.0,
        15.0, 0.0, 6.0, 20.0
    };

    double serial_sum = 0.0;
    double serial_squares = 0.0;

    double parallel_sum = 0.0;
    double parallel_squares = 0.0;

    omp_set_dynamic(0);

    // ====================================
    // STEP 1: Serial reference
    // ====================================

    for (int i = 0; i < PATHS; i++) {

        double x = payoffs[i];

        serial_sum += x;
        serial_squares += x * x;
    }

    // ====================================
    // STEP 2: Parallel reduction
    // ====================================

    #pragma omp parallel for \
        schedule(static) \
        reduction(+:parallel_sum, parallel_squares)

    for (int i = 0; i < PATHS; i++) {

        double x = payoffs[i];

        // OpenMP provides thread-local
        // accumulators and combines them
        // at the end of the parallel loop.

        parallel_sum += x;
        parallel_squares += x * x;
    }

    // ====================================
    // STEP 3: Calculate statistics
    // ====================================

    double M = (double)PATHS;

    double serial_price = serial_sum / M;

    double parallel_price = parallel_sum / M;

    double variance = (
        parallel_squares
        - parallel_sum * parallel_sum / M
    ) / (M - 1.0);

    double standard_error = sqrt(variance / M);

    // ====================================
    // STEP 4: Verify results
    // ====================================

    printf("===== OPENMP REDUCTION DEMO =====\n");

    printf("Threads requested: %d\n",
           omp_get_max_threads());

    printf("Number of payoffs: %d\n", PATHS);

    printf("\n===== SERIAL =====\n");

    printf("Sum: %.6f\n", serial_sum);
    printf("Sum of squares: %.6f\n",
           serial_squares);
    printf("Average payoff: %.6f\n",
           serial_price);

    printf("\n===== PARALLEL =====\n");

    printf("Sum: %.6f\n", parallel_sum);
    printf("Sum of squares: %.6f\n",
           parallel_squares);
    printf("Average payoff: %.6f\n",
           parallel_price);

    printf("\nIllustrative standard error: %.6f\n",
           standard_error);

    int passed =
        fabs(serial_sum - parallel_sum) < 1e-10 &&
        fabs(serial_squares - parallel_squares) < 1e-10;

    printf("\nCorrectness: %s\n",
           passed ? "PASS" : "FAIL");
        
    return passed ? 0 : 1;
}
