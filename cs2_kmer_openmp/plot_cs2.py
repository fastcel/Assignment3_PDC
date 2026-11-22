
import csv
import re
import math
from pathlib import Path
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt

# ============================================
# STEP 1: Load and validate benchmark data
# ============================================

source = Path("results_cs2/benchmark.csv")
output = Path("figures_cs2")
output.mkdir(exist_ok=True)

processes = [1, 2, 4, 6]
k_values = [8, 12]
length = 100_000_000

measurements = defaultdict(list)

with source.open(newline="") as f:
    for row in csv.DictReader(f):
        k = int(row["k"])
        p = int(row["threads"])

        measurements[(k, p)].append({
            method: float(row[method])
            for method in ("serial", "atomic", "private")
        })

times = {}
baselines = {}

for k in k_values:
    for p in processes:
        runs = measurements[(k, p)]

        if len(runs) != 3:
            raise ValueError(
                f"Expected 3 runs for k={k}, p={p}"
            )

        for method in ("serial", "atomic", "private"):
            values = [r[method] for r in runs]

            if not all(
                math.isfinite(v) and v > 0
                for v in values
            ):
                raise ValueError("Invalid timing")

            times[(k, p, method)] = median(values)

    baselines[k] = times[(k, 1, "serial")]

def save(name):
    plt.tight_layout()
    plt.savefig(output / name, dpi=200)
    plt.close()
    print("Saved:", output / name)

# ============================================
# STEP 2: Runtime and speedup graphs
# ============================================

for k in k_values:

    # Atomic vs private execution time
    plt.figure(figsize=(8, 5))

    for method, label in (
        ("atomic", "Atomic"),
        ("private", "Private histogram")
    ):
        values = [
            times[(k, p, method)]
            for p in processes
        ]

        plt.plot(
            processes, values,
            marker="o", label=label
        )

    plt.xlabel("OpenMP Threads")
    plt.ylabel("Median Time (seconds)")
    plt.title(f"k={k}: Atomic vs Private Runtime")
    plt.xticks(processes)
    plt.grid(alpha=0.3)
    plt.legend()

    save(f"0{1 if k == 8 else 3}_runtime_k{k}.png")

    # Speedup against the serial baseline
    plt.figure(figsize=(8, 5))

    for method, label in (
        ("atomic", "Atomic"),
        ("private", "Private histogram")
    ):
        speedups = [
            baselines[k] / times[(k, p, method)]
            for p in processes
        ]

        plt.plot(
            processes, speedups,
            marker="o", label=label
        )

    plt.plot(
        processes, processes,
        linestyle="--", label="Ideal"
    )

    plt.xlabel("OpenMP Threads")
    plt.ylabel("Speedup vs Serial")
    plt.title(f"k={k}: Strong Scaling Speedup")
    plt.xticks(processes)
    plt.grid(alpha=0.3)
    plt.legend()

    save(f"0{2 if k == 8 else 4}_speedup_k{k}.png")

# ============================================
# STEP 3: False-sharing comparison
# ============================================

adjacent = []
padded = []

for p in processes:
    path = Path(
        f"results_cs2/false_sharing_p{p}.txt"
    )

    content = path.read_text()

    a = re.search(
        r"Adjacent counters:\s*([0-9.]+)",
        content
    )

    b = re.search(
        r"Padded counters:\s*([0-9.]+)",
        content
    )

    if not a or not b:
        raise ValueError(
            f"Missing false-sharing results: {path}"
        )

    adjacent.append(float(a.group(1)))
    padded.append(float(b.group(1)))

plt.figure(figsize=(8, 5))

plt.plot(
    processes, adjacent,
    marker="o", label="Adjacent counters"
)

plt.plot(
    processes, padded,
    marker="s", label="64-byte padded counters"
)

plt.xlabel("OpenMP Threads")
plt.ylabel("Median Time (seconds)")
plt.title("False Sharing: Adjacent vs Padded")
plt.xticks(processes)
plt.grid(alpha=0.3)
plt.legend()

save("05_false_sharing.png")

# ============================================
# STEP 4: Throughput graph
# ============================================

plt.figure(figsize=(9, 5))

for k in k_values:
    windows = length - k + 1

    for method in ("atomic", "private"):
        throughput = [
            windows /
            (times[(k, p, method)] * 1e6)
            for p in processes
        ]

        plt.plot(
            processes, throughput,
            marker="o",
            label=f"k={k}, {method}"
        )

plt.xlabel("OpenMP Threads")
plt.ylabel("Million k-mers per second")
plt.title("k-mer Processing Throughput")
plt.xticks(processes)
plt.grid(alpha=0.3)
plt.legend()

save("06_throughput.png")

# ============================================
# STEP 5: Save numerical performance table
# ============================================

summary_path = Path(
    "results_cs2/performance_summary.csv"
)

with summary_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "k", "threads", "method",
        "median_seconds", "speedup",
        "efficiency_percent",
        "throughput_million_per_second",
        "private_histogram_mib"
    ])

    for k in k_values:
        windows = length - k + 1
        bins = 1 << (2 * k)

        for p in processes:

            private_mib = (
                bins * p * 8 / (1024 ** 2)
            )

            for method in ("atomic", "private"):

                t = times[(k, p, method)]
                speedup = baselines[k] / t
                efficiency = 100 * speedup / p
                throughput = windows / (t * 1e6)

                writer.writerow([
                    k, p, method,
                    f"{t:.6f}",
                    f"{speedup:.4f}",
                    f"{efficiency:.2f}",
                    f"{throughput:.3f}",
                    f"{private_mib:.2f}"
                ])

# ============================================
# STEP 6: Print summary
# ============================================

print("\n===== PERFORMANCE SUMMARY =====")

for k in k_values:
    print(f"\nk = {k}")
    print(f"Serial baseline: {baselines[k]:.6f} s")

    for p in processes:
        atomic = times[(k, p, "atomic")]
        private = times[(k, p, "private")]

        winner = (
            "Atomic" if atomic < private
            else "Private"
        )

        print(
            f"p={p} | "
            f"Atomic={atomic:.6f}s | "
            f"Private={private:.6f}s | "
            f"Winner={winner}"
        )

print("\n===== FALSE SHARING =====")

for p, a, b in zip(processes, adjacent, padded):
    print(
        f"p={p}: padding advantage = {a/b:.3f}x"
    )

print("\nSummary CSV:", summary_path)
print("===== ALL SIX FIGURES GENERATED =====")
