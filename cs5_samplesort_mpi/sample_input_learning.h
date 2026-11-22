
#ifndef SAMPLE_INPUT_LEARNING_H
#define SAMPLE_INPUT_LEARNING_H

#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#define LEARNING_MAX_N UINT64_C(100000000)

/*
 * STEP 1: A deterministic 64-bit mixing function.
 *
 * Every global index generates the same random
 * value regardless of which MPI rank owns it.
 */

static inline uint64_t learning_mix64(uint64_t x) {
    x += UINT64_C(0x9E3779B97F4A7C15);

    x = (x ^ (x >> 30)) *
        UINT64_C(0xBF58476D1CE4E5B9);

    x = (x ^ (x >> 27)) *
        UINT64_C(0x94D049BB133111EB);

    return x ^ (x >> 31);
}

/*
 * STEP 2: Parse the distribution name.
 *
 * uniform    -> mode 0
 * duplicates -> mode 1
 */

static inline int learning_distribution(
    const char *name
) {
    if (strcmp(name, "uniform") == 0) {
        return 0;
    }

    if (strcmp(name, "duplicates") == 0) {
        return 1;
    }

    return -1;
}

/*
 * STEP 3: Parse the total element count.
 */

static inline int learning_parse_count(
    const char *text,
    uint64_t *result
) {
    if (text == NULL || text[0] == '-') {
        return 0;
    }

    errno = 0;

    char *end = NULL;

    unsigned long long value =
        strtoull(text, &end, 10);

    if (
        errno != 0 ||
        end == text ||
        *end != '\0' ||
        value == 0 ||
        value > LEARNING_MAX_N
    ) {
        return 0;
    }

    *result = (uint64_t)value;
    return 1;
}

/*
 * STEP 4: Generate a packed sorting key.
 *
 * Upper 32 bits: original integer
 * Lower 32 bits: unique global index
 *
 * The maximum input is 100 million, so
 * every global index fits in 32 bits.
 */

static inline uint64_t learning_key(
    uint64_t global_index,
    int distribution
) {
    uint64_t random_bits = learning_mix64(
        global_index ^
        UINT64_C(0xA341316C9E3779B9)
    );

    uint32_t value;

    if (distribution == 0) {

        value = (uint32_t)(
            random_bits % UINT64_C(1000000)
        );

    } else {

        if (random_bits % UINT64_C(10) < 9) {
            value = 0;
        } else {
            value = 1u + (uint32_t)(
                (random_bits >> 8) %
                UINT64_C(1000000)
            );
        }
    }

    return (
        ((uint64_t)value << 32) |
        (uint32_t)global_index
    );
}

/*
 * STEP 5: Fingerprint one packed key.
 *
 * Used to compare input datasets across
 * different MPI process counts.
 */

static inline uint64_t learning_fingerprint(
    uint64_t key
) {
    return learning_mix64(
        key ^ UINT64_C(0xD6E8FEB86659FD93)
    );
}

#endif
