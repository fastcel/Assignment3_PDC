
import csv
import os
import random
import re
import subprocess

from pathlib import Path
from statistics import median

# ============================================
# STEP 1: Benchmark configuration
# ============================================

GRAPH = Path("graphs_cs4/synthetic_edges.txt")

PROGRAM = Path("./pagerank_scheduling_learning")

RESULTS = Path("results_cs4")
LOGS = RESULTS / "scheduling_benchmark_logs"

LOGS.mkdir(parents=True, exist_ok=True)

THREADS = [1, 2, 4, 6, 8]

SCHEDULES = ["static", "dynamic", "guided"]

REPEATS = 3

EXPECTED_NODES = 100_000
EXPECTED_EDGES = 1_260_000
EXPECTED_ITERATIONS = 19

if not GRAPH.is_file():
    raise SystemExit("Graph file is missing.")

if not PROGRAM.is_file():
    raise SystemExit(
        "Compile pagerank_scheduling_learning first."
    )


# ============================================
# STEP 2: Extract output fields
# ============================================

def field(text, label):

    pattern = (
        rf"^{re.escape(label)}:\s*"
        rf"([-+]?[0-9]+(?:\.[0-9]+)?"
        rf"(?:[eE][-+]?[0-9]+)?)"
    )

    match = re.search(
        pattern, text, re.MULTILINE
    )

    if match is None:
        raise ValueError(
            f"Missing output field: {label}"
        )

    return float(match.group(1))


# ============================================
# STEP 3: Build experiment combinations
# ============================================

configurations = [
    (schedule, threads)
    for schedule in SCHEDULES
    for threads in THREADS
]

raw_rows = []

print("===== PAGERANK SCHEDULING BENCHMARK =====")
print(f"Configurations: {len(configurations)}")
print(f"Repeats per configuration: {REPEATS}")
print(f"Total runs: {len(configurations) * REPEATS}")


# ============================================
# STEP 4: Execute all experiments
# ============================================

for run in range(1, REPEATS + 1):

    # Shuffle configurations each round to
    # reduce systematic ordering effects.
    order = configurations.copy()

    random.Random(20261121 + run).shuffle(order)

    print(f"\n===== ROUND {run} =====", flush=True)

    for schedule, threads in order:

        env = os.environ.copy()

        env["OMP_DYNAMIC"] = "FALSE"
        env["OMP_PROC_BIND"] = "close"
        env["OMP_PLACES"] = "cores"

        command = [
            str(PROGRAM.resolve()),
            str(GRAPH),
            str(threads),
            schedule,
            str(
                (100000 + threads - 1) // threads
                if schedule == "static"
                else 64
            )
        ]

        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )

        output = completed.stdout

        log = LOGS / (
            f"{schedule}_p{threads}_r{run}.txt"
        )

        log.write_text(output)

        # ------------------------------------
        # A. Verify configuration
        # ------------------------------------

        nodes = int(field(output, "Vertices"))
        edges = int(field(output, "Edges"))

        actual_threads = int(
            field(output, "Threads")
        )

        actual_schedule = re.search(
            r"^Schedule:\s*(\w+)",
            output,
            re.MULTILINE
        )

        if actual_schedule is None:
            raise ValueError("Missing schedule field")

        if (
            nodes != EXPECTED_NODES
            or edges != EXPECTED_EDGES
            or actual_threads != threads
            or actual_schedule.group(1) != schedule
        ):
            raise ValueError(
                f"Configuration mismatch: {log}"
            )

        # ------------------------------------
        # B. Verify complete correctness
        # ------------------------------------

        if "Correctness: PASS" not in output:
            raise ValueError(
                f"PageRank correctness failed: {log}"
            )

        serial_iters = int(
            field(output, "Serial pull iterations")
        )

        parallel_iters = int(
            field(output, "OpenMP pull iterations")
        )

        if (
            serial_iters != EXPECTED_ITERATIONS
            or parallel_iters != EXPECTED_ITERATIONS
        ):
            raise ValueError(
                f"Unexpected iteration count: {log}"
            )

        assigned_edges = int(
            field(output, "Total assigned edges")
        )

        if assigned_edges != EXPECTED_EDGES:
            raise ValueError(
                f"Workload accounting failed: {log}"
            )

        max_error = field(
            output, "Maximum rank difference"
        )

        l1_error = field(
            output, "L1 vector difference"
        )

        if max_error >= 1e-10 or l1_error >= 1e-8:
            raise ValueError(
                f"Rank vectors disagree: {log}"
            )

        # ------------------------------------
        # C. Parse performance measurements
        # ------------------------------------

        serial_time = field(
            output, "Serial pull time"
        )

        parallel_time = field(
            output, "OpenMP pull time"
        )

        if serial_time <= 0 or parallel_time <= 0:
            raise ValueError(
                f"Invalid timing in {log}"
            )

        # ------------------------------------
        # D. Extract per-thread edge work
        # ------------------------------------

        work_values = [
            int(value)
            for value in re.findall(
                r"^Thread \d+: (\d+) incoming edges$",
                output,
                re.MULTILINE
            )
        ]

        if len(work_values) != threads:
            raise ValueError(
                f"Missing worker statistics: {log}"
            )

        zero_workers = sum(
            work == 0 for work in work_values
        )

        minimum = min(work_values)
        maximum = max(work_values)

        imbalance = (
            maximum / minimum
            if minimum > 0 else None
        )

        raw_rows.append({
            "schedule": schedule,
            "threads": threads,
            "run": run,
            "serial_pull_time": serial_time,
            "parallel_pull_time": parallel_time,
            "max_rank_difference": max_error,
            "l1_vector_difference": l1_error,
            "zero_work_workers": zero_workers,
            "first_iteration_imbalance": (
                imbalance if imbalance is not None else ""
            )
        })

        print(
            f"{schedule:8s} "
            f"p={threads:<2d} "
            f"run={run} "
            f"time={parallel_time:.6f}s "
            f"zero_workers={zero_workers} "
            f"PASS",
            flush=True
        )


# ============================================
# STEP 5: Save raw results
# ============================================

raw_path = RESULTS / "scheduling_benchmark_raw.csv"

with raw_path.open("w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(raw_rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(raw_rows)


# ============================================
# STEP 6: Choose serial pull baseline
# ============================================

# Use the three serial pull runs from the
# 1-thread static configuration as baseline.

baseline_rows = [
    row for row in raw_rows
    if row["schedule"] == "static"
    and row["threads"] == 1
]

baseline = median(
    row["serial_pull_time"]
    for row in baseline_rows
)


# ============================================
# STEP 7: Calculate median results
# ============================================

summary_rows = []

for schedule in SCHEDULES:

    for threads in THREADS:

        subset = [
            row for row in raw_rows
            if row["schedule"] == schedule
            and row["threads"] == threads
        ]

        if len(subset) != REPEATS:
            raise ValueError(
                "Incomplete benchmark configuration"
            )

        median_time = median(
            row["parallel_pull_time"]
            for row in subset
        )

        speedup = baseline / median_time

        efficiency = speedup / threads * 100

        summary_rows.append({
            "schedule": schedule,
            "threads": threads,
            "median_time_seconds": median_time,
            "speedup_vs_serial_pull": speedup,
            "efficiency_percent": efficiency
        })


# ============================================
# STEP 8: Save summary CSV
# ============================================

summary_path = (
    RESULTS / "scheduling_benchmark_summary.csv"
)

with summary_path.open("w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(summary_rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================
# STEP 9: Print performance table
# ============================================

print("\n===== MEDIAN PERFORMANCE RESULTS =====")

print(f"Serial pull baseline: {baseline:.6f}s")

print(
    f"{'Schedule':<10}"
    f"{'Threads':>8}"
    f"{'Median(s)':>12}"
    f"{'Speedup':>12}"
    f"{'Efficiency':>13}"
)

for row in summary_rows:

    print(
        f"{row['schedule']:<10}"
        f"{row['threads']:>8}"
        f"{row['median_time_seconds']:>12.6f}"
        f"{row['speedup_vs_serial_pull']:>12.3f}"
        f"{row['efficiency_percent']:>12.2f}%"
    )


# ============================================
# STEP 10: Identify fastest configuration
# ============================================

best = min(
    summary_rows,
    key=lambda row: row["median_time_seconds"]
)

print("\n===== FASTEST MEASURED CONFIGURATION =====")

print("Schedule:", best["schedule"])
print("Threads:", best["threads"])

print(
    "Median time:",
    f"{best['median_time_seconds']:.6f}s"
)

print(
    "Speedup:",
    f"{best['speedup_vs_serial_pull']:.3f}x"
)

print("\nSaved raw data:", raw_path)
print("Saved summary:", summary_path)
print("Individual logs:", LOGS)

print("\n===== ALL 45 RUNS VERIFIED =====")
