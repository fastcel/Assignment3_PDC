
import csv
import math
import re
from pathlib import Path
from collections import Counter
from statistics import median

RESULTS = Path("results_cs3")
LOGS = RESULTS / "logs"
FIGURES = Path("figures_cs3")

THREADS = (1, 2, 4, 6, 8)
RUNS = (1, 2, 3)
PATHS = 100_000_000

errors = []

def check(condition, message):
    if not condition:
        errors.append(message)
        print("FAIL:", message)

def number(text, label):
    match = re.search(
        rf"^{re.escape(label)}:\s*\$?([0-9.]+)",
        text,
        re.MULTILINE
    )
    if not match:
        raise ValueError(f"Missing field: {label}")
    return float(match.group(1))

def section(text, name):
    start = f"===== {name} ====="
    if start not in text:
        raise ValueError(f"Missing section: {name}")

    return text.split(start, 1)[1].split(
        "===== ", 1
    )[0]

print("===== CS3 RESULTS AUDIT =====")

# =====================================
# 1. Verify strong-scaling CSV
# =====================================

csv_path = RESULTS / "strong_scaling.csv"

with csv_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

check(len(rows) == 15, "Expected 15 benchmark rows")

counts = Counter(int(r["threads"]) for r in rows)

for p in THREADS:
    check(counts[p] == 3, f"Expected 3 runs at p={p}")

    subset = [r for r in rows if int(r["threads"]) == p]

    check(
        sorted(int(r["run"]) for r in subset) == [1, 2, 3],
        f"Run numbers incorrect at p={p}"
    )

    for r in subset:
        check(
            float(r["serial_time"]) > 0 and
            float(r["parallel_time"]) > 0,
            f"Invalid timing at p={p}"
        )

print("Strong-scaling CSV: checked")

# =====================================
# 2. Verify all 15 benchmark logs
# =====================================

checked_logs = 0

for p in THREADS:

    prices = []
    standard_errors = []

    for run in RUNS:

        path = LOGS / f"mc_p{p}_r{run}.txt"

        check(path.is_file(), f"Missing {path}")

        if not path.is_file():
            continue

        content = path.read_text()
        parallel = section(content, "OPENMP")

        checked_logs += 1

        check(
            int(number(content, "Paths")) == PATHS,
            f"Incorrect path count: {path}"
        )

        check(
            int(number(content, "Actual threads")) == p,
            f"Incorrect worker count: {path}"
        )

        price = number(parallel, "Monte Carlo price")
        se = number(parallel, "Standard error")

        check(
            math.isfinite(price) and
            math.isfinite(se) and se > 0,
            f"Invalid Monte Carlo statistics: {path}"
        )

        prices.append(price)
        standard_errors.append(se)

    check(
        len(set(prices)) == 1 and len(prices) == 3,
        f"Price reproducibility failed at p={p}"
    )

    check(
        len(set(standard_errors)) == 1 and
        len(standard_errors) == 3,
        f"SE reproducibility failed at p={p}"
    )

print(f"Benchmark logs: {checked_logs}/15 checked")
print("Fixed-worker reproducibility: checked")

# =====================================
# 3. Verify near-equal-time experiment
# =====================================

fixed_times = {}
fixed_se = {}

for p, expected_paths in ((1, 18_000_000),
                          (8, 97_000_000)):

    times = []
    errors_se = []

    for run in RUNS:

        path = RESULTS / f"fixed_p{p}_r{run}.txt"

        check(path.is_file(), f"Missing {path}")

        if not path.is_file():
            continue

        content = path.read_text()
        parallel = section(content, "OPENMP")

        check(
            int(number(content, "Paths")) ==
            expected_paths,
            f"Incorrect fixed-time workload: {path}"
        )

        times.append(
            number(parallel, "Execution time")
        )

        errors_se.append(
            number(parallel, "Standard error")
        )

    if len(times) == 3:
        fixed_times[p] = median(times)
        fixed_se[p] = median(errors_se)

check(
    len(fixed_times) == 2,
    "Incomplete fixed-time experiments"
)

if len(fixed_times) == 2:
    print(
        "Fixed-time median runtimes:",
        f"p=1 {fixed_times[1]:.6f}s;",
        f"p=8 {fixed_times[8]:.6f}s"
    )

    print(
        "SE improvement:",
        f"{fixed_se[1] / fixed_se[8]:.3f}x"
    )

# =====================================
# 4. Verify RNG correlation experiments
# =====================================

for p in (4, 8):

    path = RESULTS / f"rng_corr_p{p}.txt"

    check(path.is_file(), f"Missing {path}")

    if not path.is_file():
        continue

    content = path.read_text()

    check(
        "Identical worker results: YES" in content,
        f"Duplicate-stream demonstration failed: p={p}"
    )

    factor = number(
        content, "SE underestimation factor"
    )

    check(
        abs(factor - math.sqrt(p)) < 0.02,
        f"Unexpected SE underestimation at p={p}"
    )

print("RNG correlation experiments: checked")

# =====================================
# 5. Verify figures
# =====================================

figures = [
    "01_runtime.png",
    "02_speedup.png",
    "03_efficiency.png",
    "04_fixed_time_precision.png",
    "05_rng_correlation.png"
]

for name in figures:

    path = FIGURES / name

    check(
        path.is_file() and
        path.stat().st_size > 0,
        f"Missing or empty figure: {path}"
    )

print("Performance figures: checked")

# =====================================
# 6. Verify performance summary
# =====================================

summary = RESULTS / "performance_summary.csv"

with summary.open(newline="") as f:
    summary_rows = list(csv.DictReader(f))

check(
    len(summary_rows) == 5,
    "Expected 5 performance-summary rows"
)

check(
    {int(r["threads"]) for r in summary_rows}
    == set(THREADS),
    "Missing thread configurations in summary"
)

print("Performance summary: checked")

# =====================================
# 7. Final audit result
# =====================================

print("\n===== FINAL CS3 AUDIT =====")

if errors:
    print(f"FAIL: {len(errors)} problem(s) found")
    raise SystemExit(1)

print("Strong-scaling benchmarks: PASS")
print("Reproducibility: PASS")
print("Fixed-time experiment: PASS")
print("RNG correlation: PASS")
print("Performance figures: PASS")
print("Summary CSV: PASS")

print("\nCS3 EXPERIMENTAL RESULTS AUDIT: PASS")
