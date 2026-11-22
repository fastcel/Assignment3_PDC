
import csv
import math
import re

from pathlib import Path
from statistics import median

# ============================================
# STEP 1: Experiment configuration
# ============================================

GRAPH = Path("graphs_cs4/synthetic_edges.txt")
RESULTS = Path("results_cs4")
FIGURES = Path("figures_cs4")

THREADS = [1, 2, 4, 6, 8]
RUNS = [1, 2, 3]

SCHEDULES = ["static", "dynamic", "guided"]
ALGORITHMS = ["atomic_push", "dynamic_pull"]

EXPECTED_NODES = 100_000
EXPECTED_EDGES = 1_260_000
EXPECTED_ITERATIONS = 19

# ============================================
# STEP 2: Verification helpers
# ============================================

def require(condition, message):
    if not condition:
        raise AssertionError(message)


def field(text, label):
    pattern = (
        rf"^{re.escape(label)}:\s*"
        rf"\$?([-+]?(?:\d+\.?\d*|\.\d+)"
        rf"(?:[eE][-+]?\d+)?)"
    )

    match = re.search(pattern, text, re.MULTILINE)

    require(match is not None, f"Missing field: {label}")

    value = float(match.group(1))

    require(math.isfinite(value), f"Invalid {label}")

    return value


def section(text, name):
    heading = f"===== {name} ====="

    require(heading in text, f"Missing section: {name}")

    return text.split(heading, 1)[1].split(
        "\n=====", 1
    )[0]


def read_csv(path):
    require(path.is_file(), f"Missing CSV: {path}")

    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def close(a, b, tolerance=1e-6):
    return abs(a - b) <= tolerance


def check_numerics(output, residual_label,
                   mass_label, max_label, l1_label):

    require(
        "Correctness: PASS" in output,
        "Correctness test failed"
    )

    require(
        field(output, residual_label) < 1e-10,
        "Convergence residual too large"
    )

    require(
        abs(field(output, mass_label) - 1.0) < 1e-8,
        "PageRank mass not conserved"
    )

    require(
        field(output, max_label) < 1e-10,
        "Maximum rank difference too large"
    )

    require(
        field(output, l1_label) < 1e-8,
        "Full-vector L1 error too large"
    )


# ============================================
# STEP 3: Verify graph and serial reference
# ============================================

print("===== CS4 RESULTS AUDIT =====")

require(GRAPH.is_file(), "Graph file missing")

with GRAPH.open() as file:
    header = file.readline().strip()

require(
    header == "100000 1260000",
    "Unexpected graph dimensions"
)

reference_path = RESULTS / "serial_reference.txt"

require(
    reference_path.is_file(),
    "Serial reference missing"
)

reference = reference_path.read_text()

require(
    "Correctness: PASS" in reference,
    "Serial PageRank failed"
)

require(
    int(field(reference, "Iterations"))
    == EXPECTED_ITERATIONS,
    "Serial iteration count mismatch"
)

require(
    field(reference, "Final L1 difference") < 1e-10,
    "Serial reference did not converge"
)

require(
    close(field(reference, "Sum of ranks"), 1.0),
    "Serial rank mass mismatch"
)

print("Graph header and serial reference: PASS")


# ============================================
# STEP 4: Verify scheduling benchmark logs
# ============================================

schedule_raw = read_csv(
    RESULTS / "scheduling_benchmark_raw.csv"
)

require(
    len(schedule_raw) == 45,
    "Expected 45 scheduling measurements"
)

schedule_index = {
    (row["schedule"],
     int(row["threads"]),
     int(row["run"])): row
    for row in schedule_raw
}

require(
    len(schedule_index) == 45,
    "Duplicate scheduling measurements"
)

for schedule in SCHEDULES:
    for p in THREADS:
        for run in RUNS:

            key = (schedule, p, run)

            require(
                key in schedule_index,
                f"Missing scheduling measurement: {key}"
            )

            path = (
                RESULTS / "scheduling_benchmark_logs"
                / f"{schedule}_p{p}_r{run}.txt"
            )

            require(path.is_file(), f"Missing log: {path}")

            output = path.read_text()
            row = schedule_index[key]

            require(
                int(field(output, "Vertices"))
                == EXPECTED_NODES,
                f"Wrong vertex count: {path}"
            )

            require(
                int(field(output, "Edges"))
                == EXPECTED_EDGES,
                f"Wrong edge count: {path}"
            )

            require(
                int(field(output, "Threads")) == p,
                f"Wrong thread count: {path}"
            )

            require(
                f"Schedule: {schedule}" in output,
                f"Wrong schedule: {path}"
            )

            require(
                int(field(output, "Serial pull iterations"))
                == EXPECTED_ITERATIONS,
                f"Serial iterations mismatch: {path}"
            )

            require(
                int(field(output, "OpenMP pull iterations"))
                == EXPECTED_ITERATIONS,
                f"OpenMP iterations mismatch: {path}"
            )

            check_numerics(
                output,
                "Final residual",
                "Sum of ranks",
                "Maximum rank difference",
                "L1 vector difference"
            )

            require(
                int(field(output, "Total assigned edges"))
                == EXPECTED_EDGES,
                f"Work accounting failed: {path}"
            )

            work = [
                int(x) for x in re.findall(
                    r"^Thread \d+: (\d+) incoming edges$",
                    output,
                    re.MULTILINE
                )
            ]

            require(
                len(work) == p and sum(work) == EXPECTED_EDGES,
                f"Incorrect worker statistics: {path}"
            )

            require(
                close(
                    float(row["serial_pull_time"]),
                    field(output, "Serial pull time")
                ),
                f"Serial timing mismatch: {path}"
            )

            require(
                close(
                    float(row["parallel_pull_time"]),
                    field(output, "OpenMP pull time")
                ),
                f"Parallel timing mismatch: {path}"
            )

print("Scheduling benchmark logs: 45/45 PASS")


# ============================================
# STEP 5: Verify scheduling medians
# ============================================

schedule_summary = read_csv(
    RESULTS / "scheduling_benchmark_summary.csv"
)

require(
    len(schedule_summary) == 15,
    "Expected 15 scheduling summary rows"
)

baseline = median(
    float(row["serial_pull_time"])
    for row in schedule_raw
    if row["schedule"] == "static"
    and int(row["threads"]) == 1
)

for row in schedule_summary:

    schedule = row["schedule"]
    p = int(row["threads"])

    times = [
        float(r["parallel_pull_time"])
        for r in schedule_raw
        if r["schedule"] == schedule
        and int(r["threads"]) == p
    ]

    require(
        len(times) == 3,
        "Incomplete scheduling configuration"
    )

    measured = median(times)
    speedup = baseline / measured
    efficiency = 100.0 * speedup / p

    require(
        close(
            float(row["median_time_seconds"]),
            measured
        ),
        "Incorrect scheduling median"
    )

    require(
        close(
            float(row["speedup_vs_serial_pull"]),
            speedup
        ),
        "Incorrect scheduling speedup"
    )

    require(
        close(
            float(row["efficiency_percent"]),
            efficiency, 0.01
        ),
        "Incorrect scheduling efficiency"
    )

print("Scheduling medians and speedups: PASS")


# ============================================
# STEP 6: Verify algorithm comparison logs
# ============================================

algorithm_raw = read_csv(
    RESULTS / "algorithm_comparison_raw.csv"
)

require(
    len(algorithm_raw) == 30,
    "Expected 30 algorithm measurements"
)

algorithm_index = {
    (row["algorithm"],
     int(row["threads"]),
     int(row["run"])): row
    for row in algorithm_raw
}

require(
    len(algorithm_index) == 30,
    "Duplicate algorithm measurements"
)

for algorithm in ALGORITHMS:
    for p in THREADS:
        for run in RUNS:

            key = (algorithm, p, run)

            require(
                key in algorithm_index,
                f"Missing algorithm measurement: {key}"
            )

            path = (
                RESULTS / "algorithm_comparison_logs"
                / f"{algorithm}_p{p}_r{run}.txt"
            )

            require(path.is_file(), f"Missing log: {path}")

            output = path.read_text()
            row = algorithm_index[key]

            require(
                int(field(output, "Vertices"))
                == EXPECTED_NODES,
                f"Wrong graph: {path}"
            )

            require(
                int(field(output, "Edges"))
                == EXPECTED_EDGES,
                f"Wrong edge count: {path}"
            )

            if algorithm == "atomic_push":

                require(
                    int(field(output, "Requested threads"))
                    == p,
                    f"Wrong thread count: {path}"
                )

                serial = section(output, "SERIAL PUSH")
                parallel = section(
                    output, "OPENMP ATOMIC PUSH"
                )

                require(
                    int(field(serial, "Iterations"))
                    == EXPECTED_ITERATIONS,
                    "Serial push iteration mismatch"
                )

                require(
                    int(field(parallel, "Iterations"))
                    == EXPECTED_ITERATIONS,
                    "Atomic push iteration mismatch"
                )

                check_numerics(
                    output,
                    "Final L1 difference",
                    "Sum of ranks",
                    "Maximum rank difference",
                    "L1 difference between vectors"
                ) if False else None

                require(
                    "Correctness: PASS" in output,
                    "Atomic push correctness failed"
                )

                require(
                    field(parallel, "Final L1 difference")
                    < 1e-10,
                    "Atomic push did not converge"
                )

                require(
                    close(
                        field(parallel, "Sum of ranks"),
                        1.0
                    ),
                    "Atomic push rank mass mismatch"
                )

                require(
                    field(output, "Maximum rank difference")
                    < 1e-10,
                    "Atomic push rank error"
                )

                require(
                    field(
                        output,
                        "L1 difference between vectors"
                    ) < 1e-8,
                    "Atomic push vector error"
                )

                measured = field(
                    parallel, "Execution time"
                )

                reference_time = field(
                    serial, "Execution time"
                )

            else:

                require(
                    int(field(output, "Threads")) == p,
                    f"Wrong thread count: {path}"
                )

                require(
                    "Schedule: dynamic" in output,
                    f"Wrong scheduling policy: {path}"
                )

                check_numerics(
                    output,
                    "Final residual",
                    "Sum of ranks",
                    "Maximum rank difference",
                    "L1 vector difference"
                )

                measured = field(
                    output, "OpenMP pull time"
                )

                reference_time = field(
                    output, "Serial pull time"
                )

            require(
                close(
                    measured,
                    float(row["parallel_seconds"])
                ),
                f"Parallel timing mismatch: {path}"
            )

            require(
                close(
                    reference_time,
                    float(row["reference_seconds"])
                ),
                f"Reference timing mismatch: {path}"
            )

print("Algorithm comparison logs: 30/30 PASS")


# ============================================
# STEP 7: Verify algorithm summary
# ============================================

algorithm_summary = read_csv(
    RESULTS / "algorithm_comparison_summary.csv"
)

require(
    len(algorithm_summary) == 10,
    "Expected 10 algorithm summary rows"
)

common_baseline = median(
    float(row["reference_seconds"])
    for row in algorithm_raw
    if row["algorithm"] == "atomic_push"
    and int(row["threads"]) == 1
)

for row in algorithm_summary:

    algorithm = row["algorithm"]
    p = int(row["threads"])

    times = [
        float(r["parallel_seconds"])
        for r in algorithm_raw
        if r["algorithm"] == algorithm
        and int(r["threads"]) == p
    ]

    require(
        len(times) == 3,
        "Incomplete algorithm configuration"
    )

    measured = median(times)
    speedup = common_baseline / measured

    require(
        close(
            float(row["median_seconds"]),
            measured
        ),
        "Algorithm median mismatch"
    )

    require(
        close(
            float(row["speedup_vs_serial_push"]),
            speedup
        ),
        "Algorithm speedup mismatch"
    )

print("Algorithm medians and speedups: PASS")


# ============================================
# STEP 8: Verify the five figures
# ============================================

figure_names = [
    "01_scheduling_runtime.png",
    "02_scheduling_speedup.png",
    "03_push_vs_pull_runtime.png",
    "04_algorithm_speedup.png",
    "05_pull_advantage.png"
]

for filename in figure_names:

    path = FIGURES / filename

    require(
        path.is_file() and path.stat().st_size > 1000,
        f"Missing or empty figure: {path}"
    )

    with path.open("rb") as file:
        signature = file.read(8)

    require(
        signature == b"\x89PNG\r\n\x1a\n",
        f"Invalid PNG file: {path}"
    )

print("Performance figures: 5/5 PASS")


# ============================================
# STEP 9: Final results
# ============================================

print("\n===== FINAL CS4 AUDIT =====")

print("Large graph and serial reference: PASS")
print("Scheduling benchmark logs: PASS")
print("Scheduling statistics: PASS")
print("Algorithm comparison logs: PASS")
print("Algorithm statistics: PASS")
print("Five performance figures: PASS")

print("\nCS4 EXPERIMENTAL RESULTS AUDIT: PASS")
