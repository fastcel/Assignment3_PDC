
#!/usr/bin/env bash
set -euo pipefail

echo "===== OPTIMIZED 2D MPI VERIFICATION ====="

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

# Create temporary copies with 17-digit precision
sed 's/%6\.1f/%23.17g/g' \
    serial.c > "$tmpdir/serial.c"

sed 's/%6\.1f/%23.17g/g' \
    mpi_2d_nonblocking_demo.c \
    > "$tmpdir/mpi.c"

# Compile both versions with identical optimization
gcc -O2 -Wall -Wextra -std=c11 \
    "$tmpdir/serial.c" \
    -o "$tmpdir/serial"

mpicc -O2 -Wall -Wextra -std=c11 \
    "$tmpdir/mpi.c" -lm \
    -o "$tmpdir/mpi"

for N in 7 8; do

    if [ "$N" -eq 7 ]; then
        T=20
    else
        T=40
    fi

    echo ""
    echo "Grid: ${N}x${N}, T=$T"

    "$tmpdir/serial" "$N" "$T" \
        > "$tmpdir/serial.out"

    for p in 1 2 4 6; do

        mpirun -np "$p" \
            "$tmpdir/mpi" "$N" "$T" 1 \
            > "$tmpdir/mpi.out"

        python3 - "$N" \
            "$tmpdir/serial.out" \
            "$tmpdir/mpi.out" <<'PY'
import sys
import math

N = int(sys.argv[1])

def read_grid(filename, serial):
    with open(filename) as f:
        lines = f.read().splitlines()

    marker = (
        "Temperature After "
        if serial else "Final Temperature Grid:"
    )

    for idx, line in enumerate(lines):
        if line.startswith(marker):
            rows = lines[idx + 1:idx + N + 1]
            if len(rows) != N:
                raise ValueError("Incomplete grid")

            grid = []
            for row in rows:
                values = [float(x) for x in row.split()]
                if len(values) != N:
                    raise ValueError("Invalid row size")
                grid.extend(values)
            return grid

    raise ValueError("Grid not found")

reference = read_grid(sys.argv[2], True)
parallel = read_grid(sys.argv[3], False)

if not all(math.isfinite(x) for x in reference + parallel):
    raise ValueError("Non-finite value")

max_error = max(
    abs(a - b)
    for a, b in zip(reference, parallel)
)

if max_error > 1e-10:
    print(f"FAIL | Error = {max_error:.12e}")
    sys.exit(1)

print(f"PASS | Max error = {max_error:.12e}")
PY

        echo "Completed: $p processes"
    done
done

echo ""
echo "===== ALL OPTIMIZED TESTS PASSED ====="
