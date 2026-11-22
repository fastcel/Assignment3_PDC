
#!/usr/bin/env bash
set -euo pipefail

echo "===== FULL-GRID NUMERICAL VERIFICATION ====="

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

programs=(
    mpi_1d_demo
    mpi_2d_demo
    mpi_1d_nonblocking_demo
    mpi_2d_nonblocking_demo
)

# Step 1: Create instrumented source copies.
# Enable full-grid printing for N <= 64.
# Increase printed floating-point precision.

python3 - "$tmpdir" <<'PY'
import sys
import re
from pathlib import Path

destination = Path(sys.argv[1])

sources = [
    "serial.c",
    "mpi_1d_demo.c",
    "mpi_2d_demo.c",
    "mpi_1d_nonblocking_demo.c",
    "mpi_2d_nonblocking_demo.c",
]

for filename in sources:
    source = Path(filename)

    if not source.exists():
        raise FileNotFoundError(filename)

    code = source.read_text()

    code, guards = re.subn(
        r"\bN\s*<=\s*8\b",
        "N <= 64",
        code
    )

    code, formats = re.subn(
        r"%6\.1f",
        "%23.17g",
        code
    )

    if guards == 0 or formats == 0:
        raise ValueError(
            f"Could not instrument {filename}; "
            "check its grid-printing code."
        )

    (destination / filename).write_text(code)

print("Temporary high-precision sources prepared.")
PY

# Step 2: Compile the temporary versions

gcc -O2 -Wall -Wextra -std=c11 \
    "$tmpdir/serial.c" \
    -o "$tmpdir/serial"

for program in "${programs[@]}"; do

    mpicc -O2 -Wall -Wextra -std=c11 \
        "$tmpdir/${program}.c" \
        -lm -o "$tmpdir/$program"

done

# Step 3: Test two representative grid sizes

total=0

for N in 32 64; do

    T=500
    K=50

    echo ""
    echo "================================"
    echo "Grid: ${N}x${N}, T=$T, K=$K"
    echo "================================"

    "$tmpdir/serial" "$N" "$T" \
        > "$tmpdir/reference.out"

    for program in "${programs[@]}"; do

        echo ""
        echo "Testing $program"

        for p in 1 2 4 6; do

            mpirun -np "$p" \
                "$tmpdir/$program" "$N" "$T" "$K" \
                > "$tmpdir/parallel.out"

            python3 - \
                "$N" \
                "$tmpdir/reference.out" \
                "$tmpdir/parallel.out" \
                "$program" \
                "$p" <<'PY'
import sys
import math

N = int(sys.argv[1])

serial_file = sys.argv[2]
parallel_file = sys.argv[3]

program = sys.argv[4]
processes = sys.argv[5]

def load_grid(filename, is_serial):

    with open(filename) as f:
        lines = f.read().splitlines()

    marker = (
        "Temperature After "
        if is_serial
        else "Final Temperature Grid:"
    )

    for index, line in enumerate(lines):

        if line.startswith(marker):

            rows = lines[index + 1:index + N + 1]

            if len(rows) != N:
                raise ValueError("Incomplete grid")

            grid = []

            for row in rows:

                values = [
                    float(value)
                    for value in row.split()
                ]

                if len(values) != N:
                    raise ValueError(
                        "Incorrect number of grid columns"
                    )

                grid.extend(values)

            return grid

    raise ValueError(
        f"Could not find grid in {filename}"
    )

reference = load_grid(serial_file, True)
parallel = load_grid(parallel_file, False)

if len(reference) != N * N:
    raise ValueError("Invalid serial grid size")

if len(parallel) != N * N:
    raise ValueError("Invalid parallel grid size")

max_error = 0.0
worst_cell = (0, 0)

for index, (a, b) in enumerate(
    zip(reference, parallel)
):
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Non-finite grid value")

    error = abs(a - b)

    if error > max_error:
        max_error = error
        worst_cell = (
            index // N,
            index % N
        )

tolerance = 1e-10

if max_error > tolerance:

    print(
        f"FAIL | {program} | "
        f"p={processes} | "
        f"error={max_error:.12e} | "
        f"cell={worst_cell}"
    )

    sys.exit(1)

print(
    f"PASS | p={processes} | "
    f"Max error={max_error:.12e}"
)
PY

            total=$((total + 1))

        done
    done
done

echo ""
echo "================================"
echo "ALL $total FULL-GRID TESTS PASSED"
echo "================================"
