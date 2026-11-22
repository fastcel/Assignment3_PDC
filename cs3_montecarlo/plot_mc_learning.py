
import csv
import re
from pathlib import Path
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt

# ========================================
# STEP 1: Paths and configuration
# ========================================

RESULTS = Path("results_cs3")
FIGURES = Path("figures_cs3")

FIGURES.mkdir(exist_ok=True)

THREADS = [1, 2, 4, 6, 8]

# ========================================
# STEP 2: Load strong-scaling measurements
# ========================================

measurements = defaultdict(list)

with (RESULTS / "strong_scaling.csv").open(
    newline=""
) as file:

    for row in csv.DictReader(file):
        p = int(row["threads"])

        measurements[p].append({
            "serial": float(row["serial_time"]),
            "parallel": float(row["parallel_time"])
        })

times = {}

for p in THREADS:

    runs = measurements[p]

    if len(runs) != 3:
        raise ValueError(
            f"Expected 3 benchmark runs for p={p}"
        )

    times[p] = median(
        run["parallel"] for run in runs
    )

baseline = median(
    run["serial"] for run in measurements[1]
)

speedups = {
    p: baseline / times[p]
    for p in THREADS
}

efficiencies = {
    p: speedups[p] / p * 100
    for p in THREADS
}

# ========================================
# STEP 3: Figure helper
# ========================================

def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=200)
    plt.close()

    print("Saved:", FIGURES / name)

# ========================================
# FIGURE 1: Execution time
# ========================================

plt.figure(figsize=(8, 5))

plt.plot(
    THREADS,
    [times[p] for p in THREADS],
    marker="o",
    label="OpenMP"
)

plt.axhline(
    baseline,
    linestyle="--",
    label="Serial baseline"
)

plt.xlabel("OpenMP Threads")
plt.ylabel("Median Execution Time (seconds)")
plt.title("Monte Carlo: Strong Scaling Runtime")
plt.xticks(THREADS)
plt.grid(alpha=0.3)
plt.legend()

save("01_runtime.png")

# ========================================
# FIGURE 2: Speedup
# ========================================

plt.figure(figsize=(8, 5))

plt.plot(
    THREADS,
    [speedups[p] for p in THREADS],
    marker="o",
    label="Measured speedup"
)

plt.plot(
    THREADS,
    THREADS,
    linestyle="--",
    label="Ideal linear speedup"
)

plt.xlabel("OpenMP Threads")
plt.ylabel("Speedup vs Serial")
plt.title("Monte Carlo: Strong Scaling Speedup")
plt.xticks(THREADS)
plt.grid(alpha=0.3)
plt.legend()

save("02_speedup.png")

# ========================================
# FIGURE 3: Parallel efficiency
# ========================================

plt.figure(figsize=(8, 5))

plt.plot(
    THREADS,
    [efficiencies[p] for p in THREADS],
    marker="o",
    label="Measured efficiency"
)

plt.axhline(
    100,
    linestyle="--",
    label="Ideal efficiency"
)

plt.xlabel("OpenMP Threads")
plt.ylabel("Efficiency (%)")
plt.title("Monte Carlo: Parallel Efficiency")
plt.xticks(THREADS)
plt.grid(alpha=0.3)
plt.legend()

save("03_efficiency.png")

# ========================================
# STEP 4: Read fixed-time experiment logs
# ========================================

def read_field(text, label):
    match = re.search(
        rf"{re.escape(label)}:\s*\$?([0-9.]+)",
        text
    )

    if not match:
        raise ValueError(
            f"Cannot find {label}"
        )

    return float(match.group(1))


def openmp_section(text):
    match = re.search(
        r"===== OPENMP ====="
        r"(.*?)"
        r"===== PERFORMANCE =====",
        text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            "Missing OPENMP section"
        )

    return match.group(1)


fixed = {}

for p in (1, 8):

    run_data = []

    for run in (1, 2, 3):

        filename = (
            RESULTS / f"fixed_p{p}_r{run}.txt"
        )

        content = filename.read_text()
        section = openmp_section(content)

        run_data.append({
            "time": read_field(
                section, "Execution time"
            ),
            "se": read_field(
                section, "Standard error"
            )
        })

    fixed[p] = {
        "time": median(
            value["time"] for value in run_data
        ),
        "se": median(
            value["se"] for value in run_data
        )
    }

# ========================================
# FIGURE 4: Fixed-time precision
# ========================================

plt.figure(figsize=(7, 5))

labels = ["1 thread\n18M paths",
          "8 threads\n97M paths"]

values = [fixed[1]["se"], fixed[8]["se"]]

plt.bar(labels, values)

plt.ylabel("Monte Carlo Standard Error")
plt.title("Near-Equal-Time Statistical Precision")
plt.grid(axis="y", alpha=0.3)

save("04_fixed_time_precision.png")

# ========================================
# STEP 5: Read RNG correlation logs
# ========================================

rng = {}

for p in (4, 8):

    filename = RESULTS / f"rng_corr_p{p}.txt"
    content = filename.read_text()

    rng[p] = {
        "naive": read_field(
            content, "Naive standard error"
        ),
        "corrected": read_field(
            content, "Effective standard error"
        ),
        "separate": read_field(
            content, "Estimated standard error"
        )
    }

# ========================================
# FIGURE 5: RNG correlation effects
# ========================================

positions = [0, 1]
width = 0.25

plt.figure(figsize=(8, 5))

plt.bar(
    [x - width for x in positions],
    [rng[p]["naive"] for p in (4, 8)],
    width,
    label="Identical seeds: naive SE"
)

plt.bar(
    positions,
    [rng[p]["corrected"] for p in (4, 8)],
    width,
    label="Identical seeds: corrected SE"
)

plt.bar(
    [x + width for x in positions],
    [rng[p]["separate"] for p in (4, 8)],
    width,
    label="Separately seeded: estimated SE"
)

plt.xticks(positions, ["4 threads", "8 threads"])
plt.ylabel("Standard Error")
plt.title("RNG Stream Duplication and False Precision")
plt.grid(axis="y", alpha=0.3)
plt.legend()

save("05_rng_correlation.png")

# ========================================
# STEP 6: Generate summary CSV
# ========================================

summary = RESULTS / "performance_summary.csv"

with summary.open("w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "threads",
        "median_parallel_seconds",
        "speedup",
        "efficiency_percent",
        "million_paths_per_second",
        "amdahl_effective_serial_fraction"
    ])

    for p in THREADS:

        throughput = (
            100_000_000 / times[p] / 1e6
        )

        # Amdahl's law does not identify a
        # serial fraction for p=1.
        if p == 1:
            fraction = ""
        else:
            fraction = (
                p / speedups[p] - 1
            ) / (p - 1)

        writer.writerow([
            p,
            f"{times[p]:.6f}",
            f"{speedups[p]:.4f}",
            f"{efficiencies[p]:.2f}",
            f"{throughput:.3f}",
            (
                f"{fraction:.5f}"
                if p > 1 else ""
            )
        ])

# ========================================
# STEP 7: Print results
# ========================================

print("\n===== STRONG SCALING =====")
print(f"Serial baseline: {baseline:.6f} s")

for p in THREADS:

    print(
        f"p={p}: "
        f"time={times[p]:.6f}s, "
        f"speedup={speedups[p]:.3f}x, "
        f"efficiency={efficiencies[p]:.2f}%"
    )

print("\n===== FIXED-TIME COMPARISON =====")

for p in (1, 8):
    print(
        f"p={p}: "
        f"median time={fixed[p]['time']:.6f}s, "
        f"SE={fixed[p]['se']:.8f}"
    )

improvement = (
    fixed[1]["se"] / fixed[8]["se"]
)

print(f"SE improvement: {improvement:.3f}x")

print("\n===== RNG CORRELATION =====")

for p in (4, 8):

    factor = (
        rng[p]["corrected"] /
        rng[p]["naive"]
    )

    print(
        f"p={p}: "
        f"SE underestimation={factor:.3f}x"
    )

print("\nSummary saved:", summary)
print("===== ALL FIVE FIGURES GENERATED =====")
