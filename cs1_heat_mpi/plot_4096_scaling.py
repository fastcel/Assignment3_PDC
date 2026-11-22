
import csv
import math
from pathlib import Path
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt

# ==========================================
# STEP 1: Load actual benchmark measurements
# ==========================================

csv_path = Path("results_4096/strong_scaling.csv")
output_dir = Path("figures_cs1")
output_dir.mkdir(exist_ok=True)

processes = [1, 2, 4, 6]

programs = {
    "mpi_1d_demo": "1D Blocking",
    "mpi_2d_demo": "2D Blocking",
    "mpi_1d_nonblocking_demo": "1D Non-blocking",
    "mpi_2d_nonblocking_demo": "2D Non-blocking",
}

raw = defaultdict(list)

with csv_path.open(newline="") as f:
    for row in csv.DictReader(f):
        name = row["program"]
        p = int(row["processes"])
        seconds = float(row["seconds"])

        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid execution time")

        raw[(name, p)].append(seconds)

# Validate the benchmark coverage
medians = {}

for name in programs:
    for p in processes:
        times = raw[(name, p)]

        if len(times) != 3:
            raise ValueError(
                f"Expected 3 runs for {name}, p={p}; "
                f"found {len(times)}"
            )

        medians[(name, p)] = median(times)

# ==========================================
# STEP 2: Plot execution times
# ==========================================

plt.figure(figsize=(9, 5))

for name, label in programs.items():
    times = [medians[(name, p)] for p in processes]

    plt.plot(processes, times, marker="o", label=label)

plt.xlabel("MPI Processes")
plt.ylabel("Median Execution Time (seconds)")
plt.title("Strong Scaling Runtime: N=4096, T=500")
plt.xticks(processes)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "07_large_strong_runtime.png",
    dpi=200
)
plt.close()

# ==========================================
# STEP 3: Calculate and plot speedup
# ==========================================

plt.figure(figsize=(9, 5))

for name, label in programs.items():
    baseline = medians[(name, 1)]

    speedups = [
        baseline / medians[(name, p)]
        for p in processes
    ]

    plt.plot(
        processes, speedups,
        marker="o", label=label
    )

plt.plot(
    processes, processes,
    linestyle="--", label="Ideal speedup"
)

plt.xlabel("MPI Processes")
plt.ylabel("Speedup relative to MPI p=1")
plt.title("Strong Scaling Speedup: N=4096")
plt.xticks(processes)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "08_large_strong_speedup.png",
    dpi=200
)
plt.close()

# ==========================================
# STEP 4: Calculate and plot efficiency
# ==========================================

plt.figure(figsize=(9, 5))

for name, label in programs.items():
    baseline = medians[(name, 1)]

    efficiencies = [
        100 * baseline / (p * medians[(name, p)])
        for p in processes
    ]

    plt.plot(
        processes, efficiencies,
        marker="o", label=label
    )

plt.axhline(
    70, linestyle="--",
    label="70% efficiency threshold"
)

plt.xlabel("MPI Processes")
plt.ylabel("Parallel Efficiency (%)")
plt.title("Strong Scaling Efficiency: N=4096")
plt.xticks(processes)
plt.ylim(0, 110)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "09_large_strong_efficiency.png",
    dpi=200
)
plt.close()

# ==========================================
# STEP 5: Print output confirmation
# ==========================================

print("===== NEW FIGURES GENERATED =====")

for filename in [
    "07_large_strong_runtime.png",
    "08_large_strong_speedup.png",
    "09_large_strong_efficiency.png",
]:
    print("Saved:", output_dir / filename)

print("\nAll 4096 scaling graphs completed.")
