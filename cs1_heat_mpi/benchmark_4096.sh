
#!/usr/bin/env bash
set -euo pipefail

N=4096
T=500
K=50
P=4
RUNS=3

mkdir -p results_4096

echo "===== LARGE GRID BENCHMARK ====="
echo "Grid: ${N}x${N}"
echo "Steps: $T"
echo "MPI processes: $P"
echo ""

# Rebuild with consistent optimization flags
gcc -O2 -Wall -Wextra -std=c11 \
    serial.c -o serial

for program in \
    mpi_1d_demo \
    mpi_2d_demo \
    mpi_1d_nonblocking_demo \
    mpi_2d_nonblocking_demo
do
    mpicc -O2 -Wall -Wextra -std=c11 \
        "${program}.c" -lm -o "$program"
done

CSV="results_4096/timings.csv"

echo "version,processes,run,seconds,checksum,sum_squares,center" \
    > "$CSV"

programs=(
    serial
    mpi_1d_demo
    mpi_2d_demo
    mpi_1d_nonblocking_demo
    mpi_2d_nonblocking_demo
)

for program in "${programs[@]}"; do
    echo ""
    echo "===== $program ====="

    for run in 1 2 3; do
        echo "Run $run"

        if [[ "$program" == "serial" ]]; then
            output=$(./serial "$N" "$T")
            processes=1
        else
            output=$(mpirun -np "$P" \
                "./$program" "$N" "$T" "$K")
            processes=$P
        fi

        logfile="results_4096/${program}_run${run}.txt"
        printf '%s\n' "$output" > "$logfile"

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

        echo "$program,$processes,$run,$seconds,$checksum,$squares,$center" \
            >> "$CSV"
    done
done

echo ""
echo "===== MEDIAN EXECUTION TIMES ====="

python3 - "$CSV" <<'PY'
import csv
import sys
from statistics import median
from collections import defaultdict

groups = defaultdict(list)
results = defaultdict(list)

with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        name = row["version"]
        groups[name].append(float(row["seconds"]))
        results[name].append(row)

for name, times in groups.items():
    print(f"{name:28s} {median(times):.6f} seconds")

reference = results["serial"][0]

print("\n===== SUMMARY CORRECTNESS CHECK =====")

for name, rows in results.items():
    for row in rows:
        errors = [
            abs(float(row[key]) - float(reference[key]))
            for key in ("checksum", "sum_squares", "center")
        ]

        if max(errors) > 1e-8:
            print(f"FAIL: {name}, run {row['run']}")
            sys.exit(1)

    print(f"{name}: PASS")

print("\nAll summary checks passed.")
PY

echo ""
echo "Results saved in: $CSV"
echo "===== BENCHMARK COMPLETE ====="
