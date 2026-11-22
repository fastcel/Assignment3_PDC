import csv
import os
import random
import re
import subprocess
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

# ============================================
# STEP 1: Experiment configuration
# ============================================

GRAPH = "graphs_cs4/preferential_attachment_edges.txt"
PROGRAM = "./pagerank_thread_timing_learning"

CONFIGS = [
    ("static", 25000),
    ("static", 128),
    ("dynamic", 2048),
    ("guided", 2048),
]

THREADS = 4
REPEATS = 3

RESULTS = Path("results_cs4/thread_imbalance")
LOGS = RESULTS / "logs"
FIGURES = Path("figures_cs4")

LOGS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(exist_ok=True)

env = os.environ.copy()
env["OMP_DYNAMIC"] = "FALSE"
env["OMP_PROC_BIND"] = "close"
env["OMP_PLACES"] = "cores"

# ============================================
# STEP 2: Parse verified output fields
# ============================================

def number(output, label):
    pattern = (
        rf"^{re.escape(label)}:\s*"
        rf"([0-9]+(?:\.[0-9]+)?)"
    )

    match = re.search(pattern, output, re.MULTILINE)

    if match is None:
        raise ValueError(f"Missing field: {label}")

    return float(match.group(1))


def run_case(schedule, chunk, repetition):

    command = [
        PROGRAM,
        GRAPH,
        str(THREADS),
        schedule,
        str(chunk),
    ]

    completed = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
    )

    output = completed.stdout + "\n" + completed.stderr

    filename = (
        f"{schedule}_chunk{chunk}_r{repetition}.txt"
    )

    (LOGS / filename).write_text(output)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Execution failed: {filename}"
        )

    if "Correctness: PASS" not in output:
        raise RuntimeError(
            f"Correctness failed: {filename}"
        )

    if (
        int(number(output, "Serial pull iterations")) != 69
        or int(number(output, "OpenMP pull iterations")) != 69
    ):
        raise RuntimeError(
            f"Unexpected iteration count: {filename}"
        )

    pattern = (
        r"^Thread (\d+): "
        r"first=([0-9.]+) ms "
        r"total=([0-9.]+) ms$"
    )

    matches = re.findall(
        pattern, output, re.MULTILINE
    )

    if len(matches) != THREADS:
        raise RuntimeError(
            f"Missing per-thread timings: {filename}"
        )

    first_times = [
        float(item[1]) for item in matches
    ]

    cumulative_times = [
        float(item[2]) for item in matches
    ]

    first_R = (
        max(first_times) /
        (sum(first_times) / THREADS)
    )

    cumulative_R = (
        max(cumulative_times) /
        (sum(cumulative_times) / THREADS)
    )

    return {
        "schedule": schedule,
        "chunk": chunk,
        "repetition": repetition,
        "first_R": first_R,
        "cumulative_R": cumulative_R,
        "runtime_seconds": number(
            output, "OpenMP pull time"
        ),
        "serial_seconds": number(
            output, "Serial pull time"
        ),
    }


# ============================================
# STEP 3: Run all configurations
# ============================================

rows = []

print("===== THREAD IMBALANCE BENCHMARK =====")
print("Configurations:", len(CONFIGS))
print("Repetitions:", REPEATS)
print("Total experiments:", len(CONFIGS) * REPEATS)

for repetition in range(1, REPEATS + 1):

    order = CONFIGS.copy()

    random.Random(
        20261122 + repetition
    ).shuffle(order)

    print(f"\n===== ROUND {repetition} =====")

    for schedule, chunk in order:

        result = run_case(
            schedule, chunk, repetition
        )

        rows.append(result)

        print(
            f"{schedule:<8} "
            f"chunk={chunk:<5} "
            f"time={result['runtime_seconds']:.6f}s "
            f"R={result['cumulative_R']:.3f}x "
            "PASS"
        )


# ============================================
# STEP 4: Save raw measurements
# ============================================

raw_file = RESULTS / "thread_imbalance_raw.csv"

fields = list(rows[0].keys())

with raw_file.open("w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================
# STEP 5: Calculate median results
# ============================================

summary = []

for schedule, chunk in CONFIGS:

    selected = [
        row for row in rows
        if row["schedule"] == schedule
        and row["chunk"] == chunk
    ]

    summary.append({
        "schedule": schedule,
        "chunk": chunk,
        "median_first_R": median(
            r["first_R"] for r in selected
        ),
        "median_cumulative_R": median(
            r["cumulative_R"] for r in selected
        ),
        "median_runtime_seconds": median(
            r["runtime_seconds"] for r in selected
        ),
    })

summary_file = (
    RESULTS / "thread_imbalance_summary.csv"
)

with summary_file.open("w", newline="") as file:

    writer = csv.DictWriter(
        file, fieldnames=list(summary[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary)


# ============================================
# STEP 6: Print the final comparison
# ============================================

print("\n===== MEDIAN THREAD IMBALANCE =====")

print(
    f"{'Schedule':<10} {'Chunk':>7} "
    f"{'First R':>10} {'Total R':>10} "
    f"{'Runtime':>11}"
)

for row in summary:

    print(
        f"{row['schedule']:<10} "
        f"{row['chunk']:>7} "
        f"{row['median_first_R']:>9.3f}x "
        f"{row['median_cumulative_R']:>9.3f}x "
        f"{row['median_runtime_seconds']:>10.6f}s"
    )


# ============================================
# STEP 7: Plot median cumulative imbalance
# ============================================

labels = [
    f"{row['schedule']}\n"
    f"chunk {row['chunk']}"
    for row in summary
]

values = [
    row["median_cumulative_R"]
    for row in summary
]

plt.figure(figsize=(9, 5.5))

plt.bar(labels, values)

plt.axhline(
    1.0,
    linestyle="--",
    label="Perfect balance"
)

plt.title(
    "PageRank — Median Thread-Time Imbalance"
)

plt.xlabel("OpenMP scheduling configuration")

plt.ylabel(
    "Slowest thread time / mean thread time"
)

plt.ylim(bottom=0)
plt.grid(axis="y", alpha=0.25)
plt.legend()

figure_file = (
    FIGURES / "07_thread_timing_imbalance.png"
)

plt.savefig(
    figure_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:", raw_file)
print("Saved:", summary_file)
print("Saved:", figure_file)

print("\n===== ALL 12 RUNS VERIFIED =====")
