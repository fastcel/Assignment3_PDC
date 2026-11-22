
#!/usr/bin/env bash
set -euo pipefail

N=4096
T=500
K=50
RUNS=3

PROGRAMS=(
    mpi_1d_demo
    mpi_2d_demo
    mpi_1d_nonblocking_demo
    mpi_2d_nonblocking_demo
)

PROCESSES=(1 2 4 6)

mkdir -p results_4096/strong_logs

CSV="results_4096/strong_scaling.csv"

echo "===== 4096 STRONG SCALING ====="
echo "Grid: ${N}x${N}"
echo "Iterations: $T"
echo "Convergence interval: $K"
echo "Runs per configuration: $RUNS"
echo ""

# Ensure all versions are compiled
make

printf 'program,processes,run,seconds,checksum,sum_squares,center\n' > "$CSV"

for program in "${PROGRAMS[@]}"; do

    echo ""
    echo "===== $program ====="

    for p in "${PROCESSES[@]}"; do

        echo ""
        echo "MPI processes: $p"

        for run in $(seq 1 "$RUNS"); do

            LOG="results_4096/strong_logs/${program}_p${p}_r${run}.txt"

            mpirun -np "$p" "./$program" \
                "$N" "$T" "$K" > "$LOG"

            seconds=$(awk '/^Execution Time:/ {print $3}' "$LOG")
            checksum=$(awk '/^Checksum:/ {print $2}' "$LOG")
            squares=$(awk '/^Sum of Squares:/ {print $4}' "$LOG")
            center=$(awk '/^Centre Temperature:/ {print $3}' "$LOG")

            # Do not silently record incomplete output
            if [[ -z "$seconds" || -z "$checksum" ||
                  -z "$squares" || -z "$center" ]]; then
                echo "ERROR: Missing output in $LOG"
                exit 1
            fi

            printf '%s,%s,%s,%s,%s,%s,%s\n' \
                "$program" "$p" "$run" \
                "$seconds" "$checksum" \
                "$squares" "$center" >> "$CSV"

            printf '  Run %d: %s seconds\n' \
                "$run" "$seconds"

        done
    done
done

echo ""
echo "===== RESULTS AND CORRECTNESS ====="

python3 - "$CSV" <<'PY'
import csv
import sys
import math

from collections import defaultdict
from statistics import median

path = sys.argv[1]

timings = defaultdict(list)
rows = []

with open(path, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

        key = (
            row["program"],
            int(row["processes"])
        )

        timings[key].append(float(row["seconds"]))

# Compare numerical summary values with the
# first successful configuration.
reference = rows[0]

for row in rows:
    for field in (
        "checksum",
        "sum_squares",
        "center"
    ):
        actual = float(row[field])
        expected = float(reference[field])

        if not math.isfinite(actual):
            raise ValueError(
                f"Non-finite {field}: {row}"
            )

        if abs(actual - expected) > 1e-8:
            raise ValueError(
                f"Mismatch in {field}: {row}"
            )

print("All summary correctness checks passed.")
print()

programs = [
    "mpi_1d_demo",
    "mpi_2d_demo",
    "mpi_1d_nonblocking_demo",
    "mpi_2d_nonblocking_demo"
]

for program in programs:
    baseline = median(timings[(program, 1)])

    print(f"\n===== {program} =====")
    print(
        f"{'Processes':>10} "
        f"{'Median (s)':>13} "
        f"{'Speedup':>10} "
        f"{'Efficiency':>12}"
    )

    for p in (1, 2, 4, 6):
        t = median(timings[(program, p)])

        speedup = baseline / t
        efficiency = 100.0 * speedup / p

        print(
            f"{p:>10d} "
            f"{t:>13.6f} "
            f"{speedup:>10.3f} "
            f"{efficiency:>11.2f}%"
        )

print("\nAll 4096 strong-scaling results calculated.")
PY

echo ""
echo "CSV saved to: $CSV"
echo "===== EXPERIMENT COMPLETE ====="
