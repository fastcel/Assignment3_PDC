
#!/usr/bin/env bash
set -euo pipefail

N=4096
T=500
RUNS=3

mkdir -p results_padding

echo "===== PADDED SERIAL BASELINE ====="
echo "Grid: ${N}x${N}"
echo "Iterations: $T"
echo "Runs: $RUNS"
echo ""

# Compile both implementations consistently
gcc -O2 -Wall -Wextra -std=c11 \
    stride_probe.c -o stride_probe

mpicc -O2 -Wall -Wextra -std=c11 \
    mpi_1d_demo.c -lm -o mpi_1d_demo

CSV="results_padding/timings.csv"

echo "version,run,seconds,checksum,sum_squares,center" > "$CSV"

for version in padded_serial mpi_one_rank; do

    echo ""
    echo "===== $version ====="

    for run in 1 2 3; do

        echo "Run $run"

        if [[ "$version" == "padded_serial" ]]; then
            output=$(./stride_probe "$N" "$T" 1)
        else
            output=$(mpirun -np 1 \
                ./mpi_1d_demo "$N" "$T" 50)
        fi

        printf '%s\n' "$output" |
            grep -E \
            "Execution Time|Checksum|Sum of Squares|Centre Temperature"

        seconds=$(printf '%s\n' "$output" |
            awk '/^Execution Time:/ {print $3}')

        checksum=$(printf '%s\n' "$output" |
            awk '/^Checksum:/ {print $2}')

        squares=$(printf '%s\n' "$output" |
            awk '/^Sum of Squares:/ {print $4}')

        center=$(printf '%s\n' "$output" |
            awk '/^Centre Temperature:/ {print $3}')

        echo "$version,$run,$seconds,$checksum,$squares,$center" \
            >> "$CSV"

    done
done

python3 - "$CSV" <<'PY'
import csv
import sys
import math
from statistics import median
from collections import defaultdict

groups = defaultdict(list)
rows = []

with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)
        groups[row["version"]].append(float(row["seconds"]))

print("\n===== MEDIAN EXECUTION TIMES =====")

for version, times in groups.items():
    print(f"{version:20s}: {median(times):.6f} s")

print("\n===== NUMERICAL VERIFICATION =====")

reference = rows[0]

for row in rows:
    errors = [
        abs(float(row[key]) - float(reference[key]))
        for key in ("checksum", "sum_squares", "center")
    ]

    if not all(math.isfinite(error) for error in errors):
        raise ValueError("Non-finite result")

    if max(errors) > 1e-8:
        raise ValueError(
            f"Numerical mismatch: {row['version']}, "
            f"run {row['run']}"
        )

print("All summary statistics match within 1e-8.")
print("\n===== COMPLETE =====")
PY
