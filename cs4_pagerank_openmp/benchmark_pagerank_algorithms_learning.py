
import csv
import os
import random
import re
import subprocess

from pathlib import Path
from statistics import median

# ==========================================
# STEP 1: Configure the experiments
# ==========================================

GRAPH = Path("graphs_cs4/synthetic_edges.txt")

PUSH = Path("./pagerank_parallel_push_learning")
PULL = Path("./pagerank_scheduling_learning")

RESULTS = Path("results_cs4")
LOGS = RESULTS / "algorithm_comparison_logs"

LOGS.mkdir(parents=True, exist_ok=True)

THREADS = [1, 2, 4, 6, 8]
REPEATS = 3

ALGORITHMS = ["atomic_push", "dynamic_pull"]

NODES = 100_000
EDGES = 1_260_000
ITERATIONS = 19
TOLERANCE = 1e-10

for path in [GRAPH, PUSH, PULL]:
    if not path.is_file():
        raise SystemExit(f"Missing required file: {path}")


# ==========================================
# STEP 2: Output parsing helpers
# ==========================================

def field(output, label):
    pattern = (
        rf"^{re.escape(label)}:\s*"
        rf"([-+]?\d+(?:\.\d+)?"
        rf"(?:[eE][-+]?\d+)?)"
    )

    match = re.search(
        pattern, output, re.MULTILINE
    )

    if match is None:
        raise ValueError(f"Missing field: {label}")

    return float(match.group(1))


def section(output, name):
    heading = f"===== {name} ====="

    if heading not in output:
        raise ValueError(
            f"Missing section: {name}"
        )

    return output.split(heading, 1)[1].split(
        "===== ", 1
    )[0]


# ==========================================
# STEP 3: Validate and extract measurements
# ==========================================

def parse_push(output, threads):

    if "Correctness: PASS" not in output:
        raise ValueError("Atomic push failed correctness")

    if int(field(output, "Requested threads")) != threads:
        raise ValueError("Incorrect push thread count")

    serial = section(output, "SERIAL PUSH")
    parallel = section(output, "OPENMP ATOMIC PUSH")

    if int(field(serial, "Iterations")) != ITERATIONS:
        raise ValueError("Serial iteration mismatch")

    if int(field(parallel, "Iterations")) != ITERATIONS:
        raise ValueError("Atomic iteration mismatch")

    residual = field(
        parallel, "Final L1 difference"
    )

    mass = field(parallel, "Sum of ranks")

    max_diff = field(
        output, "Maximum rank difference"
    )

    l1_diff = field(
        output, "L1 difference between vectors"
    )

    if (
        residual >= TOLERANCE
        or abs(mass - 1.0) >= 1e-8
        or max_diff >= 1e-10
        or l1_diff >= 1e-8
    ):
        raise ValueError("Atomic push numerical mismatch")

    return {
        "time": field(parallel, "Execution time"),
        "reference": field(serial, "Execution time"),
        "max_diff": max_diff
    }


def parse_pull(output, threads):

    if "Correctness: PASS" not in output:
        raise ValueError("Dynamic pull failed correctness")

    if int(field(output, "Threads")) != threads:
        raise ValueError("Incorrect pull thread count")

    if "Schedule: dynamic" not in output:
        raise ValueError("Incorrect pull schedule")

    if int(field(output, "Total assigned edges")) != EDGES:
        raise ValueError("Missing edge work")

    if int(field(output, "Serial pull iterations")) != ITERATIONS:
        raise ValueError("Serial pull iteration mismatch")

    if int(field(output, "OpenMP pull iterations")) != ITERATIONS:
        raise ValueError("OpenMP pull iteration mismatch")

    residual = field(output, "Final residual")
    mass = field(output, "Sum of ranks")

    max_diff = field(
        output, "Maximum rank difference"
    )

    l1_diff = field(output, "L1 vector difference")

    if (
        residual >= TOLERANCE
        or abs(mass - 1.0) >= 1e-8
        or max_diff >= 1e-10
        or l1_diff >= 1e-8
    ):
        raise ValueError("Dynamic pull numerical mismatch")

    return {
        "time": field(output, "OpenMP pull time"),
        "reference": field(output, "Serial pull time"),
        "max_diff": max_diff
    }


# ==========================================
# STEP 4: Run three benchmark rounds
# ==========================================

rows = []

configs = [
    (algorithm, p)
    for algorithm in ALGORITHMS
    for p in THREADS
]

print("===== ATOMIC PUSH VS DYNAMIC PULL =====")
print("Graph:", GRAPH)
print(f"Total runs: {len(configs) * REPEATS}")

for run in range(1, REPEATS + 1):

    order = configs.copy()
    random.Random(20261121 + run).shuffle(order)

    print(f"\n===== ROUND {run} =====", flush=True)

    for algorithm, p in order:

        env = os.environ.copy()

        env["OMP_DYNAMIC"] = "FALSE"
        env["OMP_PROC_BIND"] = "close"
        env["OMP_PLACES"] = "cores"

        if algorithm == "atomic_push":
            command = [
                str(PUSH.resolve()),
                str(GRAPH),
                str(p)
            ]
        else:
            command = [
                str(PULL.resolve()),
                str(GRAPH),
                str(p),
                "dynamic",
                "64"
            ]

        completed = subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            check=True
        )

        output = completed.stdout

        if (
            int(field(output, "Vertices")) != NODES
            or int(field(output, "Edges")) != EDGES
        ):
            raise ValueError("Graph dimensions mismatch")

        if algorithm == "atomic_push":
            stats = parse_push(output, p)
        else:
            stats = parse_pull(output, p)

        if stats["time"] <= 0 or stats["reference"] <= 0:
            raise ValueError("Invalid measured execution time")

        log = LOGS / f"{algorithm}_p{p}_r{run}.txt"
        log.write_text(output)

        rows.append({
            "algorithm": algorithm,
            "threads": p,
            "run": run,
            "parallel_seconds": stats["time"],
            "reference_seconds": stats["reference"],
            "maximum_rank_difference": stats["max_diff"]
        })

        print(
            f"{algorithm:13s} "
            f"p={p:<2d} "
            f"time={stats['time']:.6f}s "
            f"PASS",
            flush=True
        )


# ==========================================
# STEP 5: Save raw measurements
# ==========================================

raw_file = RESULTS / "algorithm_comparison_raw.csv"

with raw_file.open("w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)


# ==========================================
# STEP 6: Establish one common baseline
# ==========================================

# Use serial PUSH from the three atomic-push
# experiments at p=1.
#
# Both parallel algorithms solve the same
# PageRank problem, so this gives us a common
# reference when comparing total runtimes.

baseline = median(
    row["reference_seconds"]
    for row in rows
    if row["algorithm"] == "atomic_push"
    and row["threads"] == 1
)


# ==========================================
# STEP 7: Calculate median performance
# ==========================================

summary = []

for algorithm in ALGORITHMS:

    for p in THREADS:

        subset = [
            row for row in rows
            if row["algorithm"] == algorithm
            and row["threads"] == p
        ]

        if len(subset) != REPEATS:
            raise ValueError("Incomplete experiment")

        runtime = median(
            row["parallel_seconds"]
            for row in subset
        )

        summary.append({
            "algorithm": algorithm,
            "threads": p,
            "median_seconds": runtime,
            "speedup_vs_serial_push": baseline / runtime,
            "efficiency_percent": (
                100.0 * baseline / (p * runtime)
            )
        })


# ==========================================
# STEP 8: Save summary and display table
# ==========================================

summary_file = (
    RESULTS / "algorithm_comparison_summary.csv"
)

with summary_file.open("w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=list(summary[0])
    )
    writer.writeheader()
    writer.writerows(summary)

print("\n===== MEDIAN PERFORMANCE =====")
print(f"Serial push baseline: {baseline:.6f}s")

print(
    f"{'Algorithm':<15}"
    f"{'Threads':>8}"
    f"{'Time(s)':>12}"
    f"{'Speedup':>11}"
)

for row in summary:
    print(
        f"{row['algorithm']:<15}"
        f"{row['threads']:>8}"
        f"{row['median_seconds']:>12.6f}"
        f"{row['speedup_vs_serial_push']:>11.3f}"
    )


# ==========================================
# STEP 9: Compare algorithms at each p
# ==========================================

print("\n===== DIRECT COMPARISON =====")
print("Values above 1 mean dynamic pull was faster.")

for p in THREADS:

    push_time = next(
        row["median_seconds"]
        for row in summary
        if row["algorithm"] == "atomic_push"
        and row["threads"] == p
    )

    pull_time = next(
        row["median_seconds"]
        for row in summary
        if row["algorithm"] == "dynamic_pull"
        and row["threads"] == p
    )

    ratio = push_time / pull_time

    print(
        f"p={p}: dynamic pull vs atomic push "
        f"= {ratio:.3f}x"
    )

print("\nRaw data:", raw_file)
print("Summary:", summary_file)
print("Logs:", LOGS)

print("\n===== ALL 30 RUNS VERIFIED =====")
