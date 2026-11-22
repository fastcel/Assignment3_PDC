
import csv
import os
import re
import subprocess
from pathlib import Path
from statistics import median

# ============================================
# STEP 1: Experiment configuration
# ============================================

PATHS = 100_000_000
THREADS = [1, 2, 4, 6, 8]
REPEATS = 3

EXE = Path("./mc_fullsize_learning")
RESULTS = Path("results_cs3")
LOGS = RESULTS / "logs"

LOGS.mkdir(parents=True, exist_ok=True)

if not EXE.is_file():
    raise SystemExit(
        "ERROR: Compile mc_fullsize_learning first."
    )

# ============================================
# STEP 2: Extract a value from output
# ============================================

def get_value(text, pattern):
    match = re.search(pattern, text)

    if match is None:
        raise ValueError(
            f"Missing output field: {pattern}"
        )

    return match.group(1)


def get_section(output, name):
    pattern = (
        rf"===== {name} =====\n"
        rf"(.*?)(?=\n===== |\Z)"
    )

    return get_value_with_flags(
        output, pattern, re.DOTALL
    )


def get_value_with_flags(text, pattern, flags):
    match = re.search(pattern, text, flags)

    if match is None:
        raise ValueError(
            f"Missing section: {pattern}"
        )

    return match.group(1)


# ============================================
# STEP 3: Run the benchmark
# ============================================

rows = []

print("===== CS3 MONTE CARLO BENCHMARK =====")
print(f"Paths: {PATHS:,}")
print(f"Repeats: {REPEATS}")

for p in THREADS:

    print(f"\n===== {p} THREADS =====", flush=True)

    prices = []
    standard_errors = []

    for run in range(1, REPEATS + 1):

        env = os.environ.copy()

        env["OMP_NUM_THREADS"] = str(p)
        env["OMP_DYNAMIC"] = "FALSE"
        env["OMP_PROC_BIND"] = "close"
        env["OMP_PLACES"] = "cores"

        command = [
            str(EXE.resolve()),
            str(PATHS),
            str(p)
        ]

        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )

        output = completed.stdout

        log = LOGS / f"mc_p{p}_r{run}.txt"
        log.write_text(output)

        actual = int(get_value(
            output, r"Actual threads:\s*(\d+)"
        ))

        paths = int(get_value(
            output, r"Paths:\s*(\d+)"
        ))

        if actual != p or paths != PATHS:
            raise ValueError(
                f"Incorrect configuration: {log}"
            )

        serial = get_section(output, "SERIAL")
        parallel = get_section(output, "OPENMP")

        def parse_stats(section):
            return {
                "time": float(get_value(
                    section,
                    r"Execution time:\s*([\d.]+)"
                )),
                "price": get_value(
                    section,
                    r"Monte Carlo price:\s*\$([\d.]+)"
                ),
                "se": get_value(
                    section,
                    r"Standard error:\s*([\d.]+)"
                )
            }

        s = parse_stats(serial)
        q = parse_stats(parallel)

        if s["time"] <= 0 or q["time"] <= 0:
            raise ValueError(
                f"Invalid execution time: {log}"
            )

        prices.append(q["price"])
        standard_errors.append(q["se"])

        rows.append({
            "threads": p,
            "run": run,
            "serial_time": s["time"],
            "parallel_time": q["time"],
            "serial_price": s["price"],
            "parallel_price": q["price"],
            "parallel_se": q["se"]
        })

        print(
            f"Run {run} | "
            f"Serial: {s['time']:.6f}s | "
            f"OpenMP: {q['time']:.6f}s | "
            f"Price: ${q['price']}",
            flush=True
        )

    # The fixed-order reduction is designed to
    # reproduce the numerical result at fixed p.
    if len(set(prices)) != 1:
        raise ValueError(
            f"Price reproducibility failed at p={p}"
        )

    if len(set(standard_errors)) != 1:
        raise ValueError(
            f"SE reproducibility failed at p={p}"
        )

    print("Reproducibility: PASS", flush=True)

# ============================================
# STEP 4: Save raw CSV
# ============================================

csv_path = RESULTS / "strong_scaling.csv"

with csv_path.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "threads",
            "run",
            "serial_time",
            "parallel_time",
            "serial_price",
            "parallel_price",
            "parallel_se"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

# ============================================
# STEP 5: Calculate median speedups
# ============================================

baseline = median(
    row["serial_time"]
    for row in rows
    if row["threads"] == 1
)

print("\n===== STRONG SCALING RESULTS =====")

print(f"Serial baseline: {baseline:.6f}s")

print(
    f"{'Threads':>8} "
    f"{'Median (s)':>12} "
    f"{'Speedup':>10} "
    f"{'Efficiency':>12}"
)

for p in THREADS:

    subset = [
        row for row in rows
        if row["threads"] == p
    ]

    parallel_time = median(
        row["parallel_time"]
        for row in subset
    )

    speedup = baseline / parallel_time

    efficiency = speedup / p * 100

    print(
        f"{p:>8} "
        f"{parallel_time:>12.6f} "
        f"{speedup:>10.3f} "
        f"{efficiency:>11.2f}%"
    )

print("\nResults saved:", csv_path)
print("===== BENCHMARK COMPLETE =====")
