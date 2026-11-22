
#!/usr/bin/env bash
set -euo pipefail

L=100000000
RUNS=3

THREAD_COUNTS=(1 2 4 6)
K_VALUES=(8 12)

mkdir -p results_cs2/logs

CSV="results_cs2/benchmark.csv"

echo "k,threads,run,serial,atomic,private" > "$CSV"

echo "===== CASE STUDY 2 BENCHMARK ====="
echo "DNA length: $L"
echo "Runs per configuration: $RUNS"
echo ""

# Ensure the program exists
if [[ ! -x ./omp_fullsize_learning ]]; then
    echo "ERROR: Compile omp_fullsize_learning first."
    exit 1
fi

for k in "${K_VALUES[@]}"; do

    if [[ "$k" -eq 8 ]]; then
        expected=99999993
        expected_hash="605ab0a06ea0f33e"
    else
        expected=99999989
        expected_hash="fdafb38ee6cbd6e0"
    fi

    for p in "${THREAD_COUNTS[@]}"; do

        echo ""
        echo "================================"
        echo "k=$k | threads=$p"
        echo "================================"

        for run in $(seq 1 "$RUNS"); do

            LOG="results_cs2/logs/k${k}_p${p}_r${run}.txt"

            OMP_NUM_THREADS="$p" \
            OMP_DYNAMIC=FALSE \
            OMP_PROC_BIND=close \
            OMP_PLACES=cores \
                ./omp_fullsize_learning \
                "$L" "$k" "$p" > "$LOG"

            # Verify complete histogram equality
            if ! grep -q \
                "Exact bin-by-bin correctness: PASS" \
                "$LOG"; then

                echo "FAIL: Correctness failed"
                echo "See $LOG"
                exit 1
            fi

            serial=$(awk '/^Serial:/ {print $2}' "$LOG")
            atomic=$(awk '/^Atomic:/ {print $2}' "$LOG")
            private=$(awk '/^Private:/ {print $2}' "$LOG")

            windows=$(awk '/^Expected windows:/ {print $3}' "$LOG")
            actual=$(awk '/^Serial total:/ {print $3}' "$LOG")
            hash=$(awk '/^Serial fingerprint:/ {print $3}' "$LOG")

            if [[ -z "$serial" || -z "$atomic" ||
                  -z "$private" ]]; then
                echo "ERROR: Incomplete timing results"
                exit 1
            fi

            if [[ "$windows" != "$expected" ||
                  "$actual" != "$expected" ||
                  "$hash" != "$expected_hash" ]]; then

                echo "ERROR: Unexpected serial results"
                echo "See $LOG"
                exit 1
            fi

            echo "$k,$p,$run,$serial,$atomic,$private" \
                >> "$CSV"

            printf "Run %d | Serial: %s | Atomic: %s | Private: %s\n" \
                "$run" "$serial" "$atomic" "$private"

        done
    done
done

echo ""
echo "===== MEDIAN PERFORMANCE RESULTS ====="

python3 - "$CSV" <<'PY'
import csv
import sys
from collections import defaultdict
from statistics import median

data = defaultdict(list)

with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        key = (
            int(row["k"]),
            int(row["threads"])
        )
        data[key].append(row)

for k in (8, 12):

    print(f"\n===== k = {k} =====")

    # The serial reference is measured with one
    # configured OpenMP thread, using three runs.
    baseline = median(
        float(row["serial"])
        for row in data[(k, 1)]
    )

    print(f"Serial baseline: {baseline:.6f} s")

    print(
        f"{'Threads':>8} "
        f"{'Atomic (s)':>12} "
        f"{'Private (s)':>12} "
        f"{'Atomic S':>10} "
        f"{'Private S':>10}"
    )

    for p in (1, 2, 4, 6):

        rows = data[(k, p)]

        if len(rows) != 3:
            raise ValueError(
                f"Expected 3 runs for k={k}, p={p}"
            )

        atomic = median(
            float(row["atomic"]) for row in rows
        )

        private = median(
            float(row["private"]) for row in rows
        )

        atomic_speedup = baseline / atomic
        private_speedup = baseline / private

        print(
            f"{p:>8} "
            f"{atomic:>12.6f} "
            f"{private:>12.6f} "
            f"{atomic_speedup:>10.3f} "
            f"{private_speedup:>10.3f}"
        )

print("\nAll benchmark medians calculated.")
PY

echo ""
echo "Results saved: $CSV"
echo "===== BENCHMARK COMPLETE ====="
