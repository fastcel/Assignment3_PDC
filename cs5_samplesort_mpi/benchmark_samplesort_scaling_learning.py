
import csv
import math
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

PROCESSES = [1, 2, 4, 6, 8]
DISTRIBUTIONS = ["uniform", "duplicates"]

OVERSAMPLING = 4

SERIAL = Path("./sample_serial_baseline_learning")
PARALLEL = Path("./sample_sort_repro_learning")

RESULTS = Path("results_cs5")
LOGS = RESULTS / "scaling_logs"

RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

# Reuse previously saved, valid experiment logs.
# This helps resume after a Python-script failure.
#
# Set to False when deliberately collecting
# a completely fresh set of measurements.

REUSE_VALID_LOGS = True

for program in [SERIAL, PARALLEL]:
    if not program.is_file():
        raise SystemExit(
            f"Missing executable: {program}"
        )


# ============================================
# STEP 2: Extract a text field
# ============================================

def field(output, label):

    match = re.search(
        rf"^{re.escape(label)}:[ \t]*([^\r\n]+)",
        output,
        re.MULTILINE
    )

    if match is None:
        raise ValueError(
            f"Missing output field: {label}"
        )

    return match.group(1).strip()


# ============================================
# STEP 3: Correctly extract numeric values
# ============================================

# Examples supported:
#
# "17.395528"
# "17.395528 s"
# "1.230x"
# "100000000"

NUMERIC_PATTERN = re.compile(
    r"^([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?)"
    r"(?:\s*(?:s|x))?$"
)


def number(output, label):

    raw = field(output, label)

    match = NUMERIC_PATTERN.fullmatch(raw)

    if match is None:
        raise ValueError(
            f"Invalid numeric value for "
            f"{label}: {raw!r}"
        )

    value = float(match.group(1))

    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite numeric value: {label}"
        )

    return value


# ============================================
# STEP 4: Extract the timing section
# ============================================

def phase_section(output):

    heading = "===== PHASE TIMINGS ====="

    if heading not in output:
        raise ValueError(
            "Missing PHASE TIMINGS section"
        )

    return output.split(heading, 1)[1]


# ============================================
# STEP 5: Run or reuse one experiment
# ============================================

def execute(command, log_path):

    if REUSE_VALID_LOGS and log_path.is_file():

        saved = log_path.read_text()

        try:
            if field(saved, "Correctness") == "PASS":

                print(
                    f"  Reusing saved log: "
                    f"{log_path.name}",
                    flush=True
                )

                return saved

        except ValueError:
            pass

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    output = completed.stdout

    if completed.stderr:
        output += (
            "\n===== STDERR =====\n"
            + completed.stderr
        )

    # Save output before checking success so
    # failed runs can be investigated.

    log_path.write_text(output)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Program failed: {log_path}\n"
            f"Exit code: {completed.returncode}"
        )

    if field(output, "Correctness") != "PASS":
        raise ValueError(
            f"Correctness failed: {log_path}"
        )

    return output


# ============================================
# STEP 6: Build experiment configurations
# ============================================

configurations = []

for distribution in DISTRIBUTIONS:

    configurations.append(
        (distribution, "serial", 0)
    )

    for p in PROCESSES:

        configurations.append(
            (distribution, "mpi", p)
        )


rows = []
fingerprints = {}

print("===== SAMPLE SORT STRONG SCALING =====")

print("Total integers:", TOTAL)
print("Repetitions:", REPEATS)
print("MPI processes:", PROCESSES)

print(
    "Total configurations:",
    len(configurations)
)

print(
    "Total executions:",
    len(configurations) * REPEATS
)

print("Reuse saved logs:", REUSE_VALID_LOGS)


# ============================================
# STEP 7: Execute benchmark rounds
# ============================================

for run in range(1, REPEATS + 1):

    order = configurations.copy()

    random.Random(
        20261121 + run
    ).shuffle(order)

    print(
        f"\n===== ROUND {run} =====",
        flush=True
    )

    for distribution, mode, p in order:

        # ------------------------------------
        # A. Choose the executable
        # ------------------------------------

        if mode == "serial":

            command = [
                str(SERIAL.resolve()),
                str(TOTAL),
                distribution
            ]

            log = LOGS / (
                f"{distribution}_serial_r{run}.txt"
            )

        else:

            command = [
                "mpirun",
                "--use-hwthread-cpus",
                "-np",
                str(p),
                str(PARALLEL.resolve()),
                str(TOTAL),
                distribution,
                str(OVERSAMPLING)
            ]

            log = LOGS / (
                f"{distribution}_mpi_p{p}_r{run}.txt"
            )

        output = execute(command, log)

        # ------------------------------------
        # B. Verify common output
        # ------------------------------------

        if field(output, "Distribution") != distribution:
            raise ValueError(
                f"Wrong distribution: {log}"
            )

        if int(number(output, "Total integers")) != TOTAL:
            raise ValueError(
                f"Wrong element count: {log}"
            )

        if field(output, "Correctness") != "PASS":
            raise ValueError(
                f"Correctness failed: {log}"
            )

        if (
            field(
                output,
                "Fingerprint preservation"
            ) != "PASS"
        ):
            raise ValueError(
                f"Fingerprint failed: {log}"
            )

        # ------------------------------------
        # C. Verify identical global input
        # ------------------------------------

        fingerprint = (
            field(output, "Global fingerprint sum"),
            field(output, "Global fingerprint xor")
        )

        if distribution not in fingerprints:

            fingerprints[distribution] = fingerprint

        elif fingerprints[distribution] != fingerprint:

            raise ValueError(
                f"Fingerprint differs between "
                f"configurations: {log}"
            )

        # ------------------------------------
        # D. Prepare row for raw results
        # ------------------------------------

        row = {
            "distribution": distribution,
            "mode": mode,
            "processes": p,
            "run": run,
            "total_seconds": "",
            "imbalance": "",
            "local_sort_seconds": "",
            "sampling_seconds": "",
            "partition_seconds": "",
            "alltoallv_seconds": "",
            "final_sort_seconds": ""
        }

        # ------------------------------------
        # E. Parse serial timing
        # ------------------------------------

        if mode == "serial":

            if field(output, "Sorted ordering") != "PASS":
                raise ValueError(
                    f"Serial output not sorted: {log}"
                )

            runtime = number(
                output,
                "Serial sorting time"
            )

        # ------------------------------------
        # F. Parse parallel timings
        # ------------------------------------

        else:

            if int(number(output, "Processes")) != p:
                raise ValueError(
                    f"Wrong MPI process count: {log}"
                )

            if (
                int(number(output, "Oversampling"))
                != OVERSAMPLING
            ):
                raise ValueError(
                    f"Wrong oversampling: {log}"
                )

            for label in [
                "Local sorting",
                "Global ordering",
                "Elements preserved"
            ]:

                if field(output, label) != "PASS":
                    raise ValueError(
                        f"{label} failed: {log}"
                    )

            # Verify final partition counts.

            counts = [
                int(value)
                for value in field(
                    output,
                    "Final counts"
                ).split()
            ]

            if (
                len(counts) != p
                or any(x < 0 for x in counts)
                or sum(counts) != TOTAL
            ):
                raise ValueError(
                    f"Invalid final counts: {log}"
                )

            row["imbalance"] = (
                max(counts) / (TOTAL / p)
            )

            # IMPORTANT:
            # Parse numerical phase timings
            # from their dedicated section.
            #
            # The verification section also
            # contains "Local sorting: PASS".

            phases = phase_section(output)

            row["local_sort_seconds"] = number(
                phases,
                "Local sorting"
            )

            row["sampling_seconds"] = number(
                phases,
                "Sample selection"
            )

            row["partition_seconds"] = number(
                phases,
                "Partition and counts"
            )

            row["alltoallv_seconds"] = number(
                phases,
                "Alltoallv redistribution"
            )

            row["final_sort_seconds"] = number(
                phases,
                "Final sorting"
            )

            runtime = number(
                phases,
                "Total sorting time"
            )

        # ------------------------------------
        # G. Store completed measurement
        # ------------------------------------

        if runtime <= 0:
            raise ValueError(
                f"Non-positive runtime: {log}"
            )

        row["total_seconds"] = runtime

        rows.append(row)

        label = (
            "serial"
            if mode == "serial"
            else f"MPI p={p}"
        )

        print(
            f"{distribution:10s} "
            f"{label:10s} "
            f"run={run} "
            f"time={runtime:.6f}s "
            f"PASS",
            flush=True
        )


# ============================================
# STEP 8: Save all raw measurements
# ============================================

raw_path = RESULTS / "scaling_raw.csv"

with raw_path.open("w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================
# STEP 9: Calculate median performance
# ============================================

summary = []

for distribution in DISTRIBUTIONS:

    serial_times = [
        row["total_seconds"]
        for row in rows
        if row["distribution"] == distribution
        and row["mode"] == "serial"
    ]

    if len(serial_times) != REPEATS:
        raise ValueError(
            f"Missing serial measurements: "
            f"{distribution}"
        )

    baseline = median(serial_times)

    for p in PROCESSES:

        subset = [
            row for row in rows
            if row["distribution"] == distribution
            and row["mode"] == "mpi"
            and row["processes"] == p
        ]

        if len(subset) != REPEATS:
            raise ValueError(
                f"Incomplete results: "
                f"{distribution}, p={p}"
            )

        parallel_time = median(
            row["total_seconds"]
            for row in subset
        )

        speedup = baseline / parallel_time
        efficiency = 100.0 * speedup / p

        phase_columns = [
            "local_sort_seconds",
            "sampling_seconds",
            "partition_seconds",
            "alltoallv_seconds",
            "final_sort_seconds"
        ]

        phase_medians = {
            key: median(
                row[key] for row in subset
            )
            for key in phase_columns
        }

        summary.append({
            "distribution": distribution,
            "processes": p,
            "serial_baseline_seconds": baseline,
            "median_mpi_seconds": parallel_time,
            "speedup_vs_serial": speedup,
            "efficiency_percent": efficiency,
            "median_imbalance": median(
                row["imbalance"]
                for row in subset
            ),
            **phase_medians
        })


# ============================================
# STEP 10: Save the summary CSV
# ============================================

summary_path = RESULTS / "scaling_summary.csv"

with summary_path.open("w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(summary[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary)


# ============================================
# STEP 11: Print the performance tables
# ============================================

print("\n===== MEDIAN STRONG-SCALING RESULTS =====")

for distribution in DISTRIBUTIONS:

    subset = [
        row for row in summary
        if row["distribution"] == distribution
    ]

    baseline = subset[0]["serial_baseline_seconds"]

    print(f"\nDistribution: {distribution}")
    print(f"Serial baseline: {baseline:.6f}s")

    print(
        f"{'P':>3} "
        f"{'Median(s)':>11} "
        f"{'Speedup':>10} "
        f"{'Efficiency':>12} "
        f"{'Imbalance':>11}"
    )

    for row in subset:

        print(
            f"{row['processes']:>3} "
            f"{row['median_mpi_seconds']:>11.6f} "
            f"{row['speedup_vs_serial']:>10.3f} "
            f"{row['efficiency_percent']:>11.2f}% "
            f"{row['median_imbalance']:>10.3f}x"
        )


# ============================================
# STEP 12: Report verified fingerprints
# ============================================

print("\n===== VERIFIED INPUT FINGERPRINTS =====")

for distribution, (total_sum, total_xor) in (
    fingerprints.items()
):
    print(
        f"{distribution}: "
        f"sum={total_sum}, xor={total_xor}"
    )

print("\nRaw measurements:", raw_path)
print("Median summary:", summary_path)
print("Individual logs:", LOGS)

print("\n===== ALL 36 RUNS VERIFIED =====")
