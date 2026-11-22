
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Convert DNA base into 2-bit integer
static int encode_base(char base) {
    switch (base) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default:  return -1;
    }
}

// Convert integer index back into DNA letters
static void decode_kmer(
    uint32_t index,
    int k,
    char *result
) {
    const char bases[] = "ACGT";

    for (int i = k - 1; i >= 0; i--) {
        result[i] = bases[index & 3u];
        index >>= 2;
    }

    result[k] = '\0';
}

int main(void) {

    const char *dna = "ACGTACGT";
    const int k = 3;

    size_t length = strlen(dna);

    uint64_t histogram[64] = {0};

    uint32_t mask = (1u << (2 * k)) - 1u;
    uint32_t code = 0;

    printf("DNA: %s\n", dna);
    printf("k: %d\n\n", k);

    // Slide through sequence once
    for (size_t i = 0; i < length; i++) {

        int base = encode_base(dna[i]);

        if (base < 0) {
            fprintf(stderr, "Invalid DNA base\n");
            return 1;
        }

        // Shift existing bits and append new base
        code = ((code << 2) | (uint32_t)base) & mask;

        // Only count once a full window exists
        if (i + 1 >= (size_t)k) {

            histogram[code]++;

            printf(
                "Window %zu: %.*s -> index %u\n",
                i + 1 - (size_t)k,
                k,
                dna + i + 1 - (size_t)k,
                code
            );
        }
    }

    printf("\n===== HISTOGRAM =====\n");

    uint64_t total = 0;

    for (uint32_t index = 0; index < 64; index++) {

        if (histogram[index] > 0) {

            char decoded[4];

            decode_kmer(index, k, decoded);

            printf(
                "%s: %llu\n",
                decoded,
                (unsigned long long)histogram[index]
            );

            total += histogram[index];
        }
    }

    uint64_t expected = length - (size_t)k + 1;

    printf("\nTotal k-mers: %llu\n",
           (unsigned long long)total);

    printf("Expected: %llu\n",
           (unsigned long long)expected);

    if (total != expected) {
        printf("FAIL: Incorrect count\n");
        return 1;
    }

    printf("PASS: Counts match!\n");

    return 0;
}
