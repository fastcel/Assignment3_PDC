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

TOTAL = 100_000_000
REPEATS = 3

DISTRIBUTIONS = ["uniform", "duplicates"]
PROCESSES = [1, 2, 4, 6, 8]

SERIAL = "./sample_serial_uint32_learning"
MPI = "./sample_sort_uint32_learning"

RESULTS = Path("results_cs5")
LOGS = RESULTS / "uint32_scaling_logs"

LOGS.mkdir(parents=True, exist_ok=True)

ENV = os.environ.copy()
ENV["OMP_NUM_THREADS"] = "1"

raw_rows = []


# ============================================
# STEP 2: Read numerical output fields
# ============================================

def number(output, label):

    match = re.search(
        rf"^{re.escape(label)}:\s*"
        rf"([-+]?\d+(?:\.\d+)?)",
        output,
        re.MULTILINE
    )

    if match is None:
        raise ValueError(
            f"Missing output field: {label}"
        )

    return float(match.group(1))


def text_field(output, label):

    match = re.search(
        rf"^{re.escape(label)}:\s*(.+)$",
        output,
        re.MULTILINE
    )

    if match is None:
        raise ValueError(
            f"Missing output field: {label}"
        )

    return match.group(1).strip()


# ============================================
# STEP 3: Run a program and save its log
# ============================================

def execute(command, log_name):

    result = subprocess.run(
        command,
        env=ENV,
        capture_output=True,
        text=True
    )

    output = result.stdout + "\n" + result.stderr

    path = LOGS / log_name
    path.write_text(output)

    if result.returncode != 0:
        raise RuntimeError(
            f"Program failed: {path}\n"
            + output[-1500:]
        )

    if "Correctness: PASS" not in output:
        raise RuntimeError(
            f"Correctness failed: {path}"
        )

    if int(number(output, "Total integers")) != TOTAL:
        raise RuntimeError(
            f"Wrong input size: {path}"
        )

    if int(number(output, "Element bytes")) != 4:
        raise RuntimeError(
            f"Not native 32-bit: {path}"
        )

    return output


# ============================================
# STEP 4: Save each completed measurement
# ============================================

RAW = RESULTS / "uint32_scaling_raw.csv"

FIELDS = [
    "distribution",
    "mode",
    "processes",
    "run",
    "seconds",
    "imbalance"
]


def save_raw():

    with RAW.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS
        )

        writer.writeheader()
        writer.writerows(raw_rows)


# ============================================
# STEP 5: Benchmark execution
# ============================================

print("===== NATIVE 32-BIT SCALING BENCHMARK =====")
print("Elements:", TOTAL)
print("Repetitions:", REPEATS)
print("MPI process counts:", PROCESSES)
print("Total expected executions: 36")

for repetition in range(1, REPEATS + 1):

    print(f"\n===== ROUND {repetition} =====", flush=True)

    references = {}

    # A. Execute matching serial references.

    for distribution in DISTRIBUTIONS:

        output = execute(
            [
                SERIAL,
                str(TOTAL),
                distribution
            ],
            f"{distribution}_serial_r{repetition}.txt"
        )

        references[distribution] = (
            text_field(
                output, "Input fingerprint sum"
            ),
            text_field(
                output, "Input fingerprint xor"
            )
        )

        seconds = number(
            output, "Sorting time"
        )

        raw_rows.append({
            "distribution": distribution,
            "mode": "serial",
            "processes": 1,
            "run": repetition,
            "seconds": seconds,
            "imbalance": 1.0
        })

        save_raw()

        print(
            f"serial   {distribution:<11} "
            f"time={seconds:.6f}s PASS",
            flush=True
        )

    # B. Randomize MPI configuration order.

    cases = [
        (distribution, p)
        for distribution in DISTRIBUTIONS
        for p in PROCESSES
    ]

    random.Random(
        20261122 + repetition
    ).shuffle(cases)

    for distribution, p in cases:

        output = execute(
            [
                "mpirun",
                "--use-hwthread-cpus",
                "-np",
                str(p),
                MPI,
                str(TOTAL),
                distribution,
                "4"
            ],
            f"{distribution}_p{p}_r{repetition}.txt"
        )

        if (
            text_field(
                output, "MPI payload datatype"
            ) != "MPI_UINT32_T"
        ):
            raise RuntimeError(
                "MPI datatype is not MPI_UINT32_T"
            )

        actual = (
            text_field(
                output, "Global fingerprint sum"
            ),
            text_field(
                output, "Global fingerprint xor"
            )
        )

        if actual != references[distribution]:
            raise RuntimeError(
                f"Serial/MPI fingerprint mismatch: "
                f"{distribution}, p={p}"
            )

        seconds = number(
            output, "Total sorting time"
        )

        imbalance = number(
            output, "Max/mean imbalance"
        )

        raw_rows.append({
            "distribution": distribution,
            "mode": "mpi",
            "processes": p,
            "run": repetition,
            "seconds": seconds,
            "imbalance": imbalance
        })

        save_raw()

        print(
            f"mpi      {distribution:<11} "
            f"p={p:<2} "
            f"time={seconds:.6f}s "
            f"imbalance={imbalance:.3f}x "
            "PASS",
            flush=True
        )


# ============================================
# STEP 6: Calculate median scaling results
# ============================================

summary = []

for distribution in DISTRIBUTIONS:

    serial_times = [
        row["seconds"]
        for row in raw_rows
        if row["distribution"] == distribution
        and row["mode"] == "serial"
    ]

    serial_median = median(serial_times)

    for p in PROCESSES:

        selected = [
            row for row in raw_rows
            if row["distribution"] == distribution
            and row["mode"] == "mpi"
            and row["processes"] == p
        ]

        mpi_median = median(
            row["seconds"] for row in selected
        )

        imbalance_median = median(
            row["imbalance"] for row in selected
        )

        speedup = serial_median / mpi_median

        efficiency = 100.0 * speedup / p

        summary.append({
            "distribution": distribution,
            "processes": p,
            "serial_median_seconds": serial_median,
            "mpi_median_seconds": mpi_median,
            "speedup": speedup,
            "efficiency_percent": efficiency,
            "median_imbalance": imbalance_median
        })


# ============================================
# STEP 7: Save summary CSV
# ============================================

SUMMARY = RESULTS / "uint32_scaling_summary.csv"

with SUMMARY.open("w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(summary[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary)


# ============================================
# STEP 8: Print results
# ============================================

print("\n===== NATIVE 32-BIT MEDIAN SUMMARY =====")

for distribution in DISTRIBUTIONS:

    print(f"\nDistribution: {distribution}")

    print(
        f"{'P':>3} "
        f"{'MPI time':>12} "
        f"{'Speedup':>10} "
        f"{'Efficiency':>12} "
        f"{'Imbalance':>11}"
    )

    selected = [
        row for row in summary
        if row["distribution"] == distribution
    ]

    for row in selected:

        print(
            f"{row['processes']:>3} "
            f"{row['mpi_median_seconds']:>11.6f}s "
            f"{row['speedup']:>9.3f}x "
            f"{row['efficiency_percent']:>10.2f}% "
            f"{row['median_imbalance']:>10.3f}x"
        )


print("\nSaved:", RAW)
print("Saved:", SUMMARY)
print("Saved logs:", LOGS)

print("\n===== ALL 36 EXECUTIONS VERIFIED =====")
