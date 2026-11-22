
#include <stdio.h>
#include <stdlib.h>

// Original approach: scan every cell
void old_boundary(int rows, int cols, int *visited) {
    int pitch = cols + 2;

    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            if (i == 1 || i == rows ||
                j == 1 || j == cols) {
                visited[i * pitch + j]++;
            }
        }
    }
}

// Improved approach: visit only perimeter cells
void new_boundary(int rows, int cols, int *visited) {
    int pitch = cols + 2;

    // Top and bottom rows
    for (int j = 1; j <= cols; j++) {
        visited[1 * pitch + j]++;

        if (rows > 1)
            visited[rows * pitch + j]++;
    }

    // Left and right columns
    // Exclude corners already visited above
    for (int i = 2; i < rows; i++) {
        visited[i * pitch + 1]++;

        if (cols > 1)
            visited[i * pitch + cols]++;
    }
}

int main(void) {
    int tests[][2] = {
        {1, 1},
        {1, 5},
        {5, 1},
        {2, 2},
        {3, 7},
        {7, 3},
        {512, 512},
        {342, 512}
    };

    int total = sizeof(tests) / sizeof(tests[0]);

    for (int t = 0; t < total; t++) {
        int rows = tests[t][0];
        int cols = tests[t][1];

        size_t cells = (size_t)(rows + 2) * (cols + 2);

        int *old = calloc(cells, sizeof(int));
        int *new = calloc(cells, sizeof(int));

        if (!old || !new) {
            fprintf(stderr, "Allocation failed\n");
            free(old);
            free(new);
            return 1;
        }

        old_boundary(rows, cols, old);
        new_boundary(rows, cols, new);

        int passed = 1;

        for (size_t i = 0; i < cells; i++) {
            if (old[i] != new[i]) {
                passed = 0;
                break;
            }
        }

        printf("%d x %d: %s\n",
               rows, cols, passed ? "PASS" : "FAIL");

        free(old);
        free(new);

        if (!passed)
            return 1;
    }

    printf("\nAll boundary tests passed!\n");
    return 0;
}
