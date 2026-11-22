
import csv
from pathlib import Path
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt

# ========================================
# STEP 1: Prepare output directory
# ========================================

output_dir = Path("figures_cs1")
output_dir.mkdir(exist_ok=True)

processes = [1, 2, 4, 6]

# ========================================
# STEP 2: Recorded N=1024 measurements
# ========================================

# Strong scaling:
# N=1024, T=500, K=50, GCC -O2
# Medians of three runs.
#
# These are the medians recorded in our
# earlier strong-scaling experiments.

serial_time = 0.853725

strong_1d = [
    0.995426,
    0.560887,
    0.344891,
    0.299430,
]

strong_2d = [
    0.912531,
    0.581568,
    0.338960,
    0.326244,
]

# Weak scaling:
# Keep approximately N^2 / p constant.

weak_grid_sizes = [1024, 1448, 2048, 2508]

weak_1d = [
    0.878722,
    1.160118,
    1.599630,
    2.171520,
]

weak_2d = [
    0.906932,
    1.140949,
    1.774705,
    2.345174,
]

# Optimized 2D communication experiment:
# N=1024, T=500, K=50
# These numbers come from the latest benchmark.

blocking_2d = [
    0.937148,
    0.572285,
    0.427185,
    0.320396,
]

nonblocking_2d = [
    1.062005,
    0.641219,
    0.467654,
    0.331105,
]

# ========================================
# STEP 3: Helper to save figures
# ========================================

def save_figure(filename):
    plt.tight_layout()
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


# ========================================
# GRAPH 1: Strong scaling speedup
# ========================================

speedup_1d = [
    serial_time / t for t in strong_1d
]

speedup_2d = [
    serial_time / t for t in strong_2d
]

plt.figure(figsize=(8, 5))

plt.plot(
    processes, speedup_1d,
    marker="o", label="MPI 1D Blocking"
)

plt.plot(
    processes, speedup_2d,
    marker="s", label="MPI 2D Blocking"
)

plt.plot(
    processes, processes,
    linestyle="--", label="Ideal Speedup"
)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Speedup (Serial Time / MPI Time)")
plt.title("Strong Scaling: N=1024, T=500")
plt.xticks(processes)
plt.grid(True, alpha=0.3)
plt.legend()

save_figure("01_strong_speedup.png")


# ========================================
# GRAPH 2: Strong scaling efficiency
# ========================================

efficiency_1d = [
    100 * speedup / p
    for speedup, p in zip(speedup_1d, processes)
]

efficiency_2d = [
    100 * speedup / p
    for speedup, p in zip(speedup_2d, processes)
]

plt.figure(figsize=(8, 5))

plt.plot(
    processes, efficiency_1d,
    marker="o", label="MPI 1D Blocking"
)

plt.plot(
    processes, efficiency_2d,
    marker="s", label="MPI 2D Blocking"
)

plt.axhline(
    70, linestyle="--",
    label="70% Efficiency Threshold"
)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Parallel Efficiency (%)")
plt.title("Strong Scaling Efficiency: N=1024")
plt.xticks(processes)
plt.ylim(0, 110)
plt.grid(True, alpha=0.3)
plt.legend()

save_figure("02_strong_efficiency.png")


# ========================================
# GRAPH 3: Weak scaling execution time
# ========================================

plt.figure(figsize=(8, 5))

plt.plot(
    processes, weak_1d,
    marker="o", label="MPI 1D Blocking"
)

plt.plot(
    processes, weak_2d,
    marker="s", label="MPI 2D Blocking"
)

plt.axhline(
    weak_1d[0], linestyle="--",
    label="Ideal Reference (1D, p=1)"
)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Execution Time (seconds)")
plt.title("Weak Scaling: Approximately Constant Work per Rank")
plt.xticks(processes)
plt.grid(True, alpha=0.3)
plt.legend()

save_figure("03_weak_runtime.png")


# ========================================
# GRAPH 4: Weak scaling efficiency
# ========================================

weak_efficiency_1d = [
    100 * weak_1d[0] / t
    for t in weak_1d
]

weak_efficiency_2d = [
    100 * weak_2d[0] / t
    for t in weak_2d
]

plt.figure(figsize=(8, 5))

plt.plot(
    processes, weak_efficiency_1d,
    marker="o", label="MPI 1D Blocking"
)

plt.plot(
    processes, weak_efficiency_2d,
    marker="s", label="MPI 2D Blocking"
)

plt.axhline(
    100, linestyle="--",
    label="Ideal Weak Scaling"
)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Weak Scaling Efficiency (%)")
plt.title("Weak Scaling Efficiency")
plt.xticks(processes)
plt.ylim(0, 110)
plt.grid(True, alpha=0.3)
plt.legend()

save_figure("04_weak_efficiency.png")


# ========================================
# GRAPH 5: Optimized 2D communication
# ========================================

plt.figure(figsize=(8, 5))

plt.plot(
    processes, blocking_2d,
    marker="o", label="2D Blocking"
)

plt.plot(
    processes, nonblocking_2d,
    marker="s", label="2D Non-blocking (Optimized)"
)

plt.xlabel("Number of MPI Processes")
plt.ylabel("Execution Time (seconds)")
plt.title("2D Communication Comparison: N=1024")
plt.xticks(processes)
plt.grid(True, alpha=0.3)
plt.legend()

save_figure("05_blocking_vs_nonblocking.png")


# ========================================
# GRAPH 6: N=4096 MPI comparison
# ========================================

csv_file = Path("results_4096/timings.csv")

if not csv_file.exists():
    raise FileNotFoundError(
        f"Missing benchmark data: {csv_file}"
    )

times = defaultdict(list)

with csv_file.open(newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["version"] != "serial":
            times[row["version"]].append(
                float(row["seconds"])
            )

versions = [
    "mpi_1d_demo",
    "mpi_2d_demo",
    "mpi_1d_nonblocking_demo",
    "mpi_2d_nonblocking_demo",
]

labels = [
    "1D Blocking",
    "2D Blocking",
    "1D Non-blocking",
    "2D Non-blocking",
]

medians = [
    median(times[version])
    for version in versions
]

plt.figure(figsize=(9, 5))

bars = plt.bar(labels, medians)

for bar, value in zip(bars, medians):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.05,
        f"{value:.3f}",
        ha="center"
    )

plt.ylabel("Median Execution Time (seconds)")
plt.title("MPI Comparison: N=4096, T=500, p=4")
plt.ylim(0, max(medians) * 1.15)
plt.xticks(rotation=12)

save_figure("06_large_grid_comparison.png")


# ========================================
# STEP 4: Print numerical summary
# ========================================

print("\n===== STRONG SCALING =====")

for i, p in enumerate(processes):
    print(
        f"p={p}: "
        f"1D speedup={speedup_1d[i]:.3f}, "
        f"1D efficiency={efficiency_1d[i]:.2f}%, "
        f"2D speedup={speedup_2d[i]:.3f}, "
        f"2D efficiency={efficiency_2d[i]:.2f}%"
    )

print("\n===== WEAK SCALING =====")

for i, p in enumerate(processes):
    print(
        f"p={p}, N={weak_grid_sizes[i]}: "
        f"1D efficiency={weak_efficiency_1d[i]:.2f}%, "
        f"2D efficiency={weak_efficiency_2d[i]:.2f}%"
    )

print("\n===== ALL SIX FIGURES GENERATED =====")
