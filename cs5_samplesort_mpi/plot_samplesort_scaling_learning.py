
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

# ============================================
# STEP 1: Input and output locations
# ============================================

RESULTS = Path("results_cs5")
FIGURES = Path("figures_cs5")

FIGURES.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = RESULTS / "scaling_summary.csv"
RAW_FILE = RESULTS / "scaling_raw.csv"

if not SUMMARY_FILE.is_file():
    raise SystemExit("Missing scaling_summary.csv")

if not RAW_FILE.is_file():
    raise SystemExit("Missing scaling_raw.csv")


# ============================================
# STEP 2: Load benchmark results
# ============================================

with SUMMARY_FILE.open(newline="") as file:
    summary = list(csv.DictReader(file))

with RAW_FILE.open(newline="") as file:
    raw = list(csv.DictReader(file))

distributions = ["uniform", "duplicates"]

results = {}

for distribution in distributions:

    subset = [
        row for row in summary
        if row["distribution"] == distribution
    ]

    subset.sort(
        key=lambda row: int(row["processes"])
    )

    results[distribution] = subset

    expected = [1, 2, 4, 6, 8]

    actual = [
        int(row["processes"])
        for row in subset
    ]

    if actual != expected:
        raise ValueError(
            f"Unexpected process counts for "
            f"{distribution}: {actual}"
        )


# ============================================
# STEP 3: Shared graph helpers
# ============================================

def save_figure(filename):

    path = FIGURES / filename

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


def values(distribution, column):

    return [
        float(row[column])
        for row in results[distribution]
    ]


processes = [1, 2, 4, 6, 8]


# ============================================
# FIGURE 1: Strong-scaling execution time
# ============================================

plt.figure(figsize=(9, 5.5))

for distribution in distributions:

    times = values(
        distribution,
        "median_mpi_seconds"
    )

    serial = values(
        distribution,
        "serial_baseline_seconds"
    )[0]

    line, = plt.plot(
        processes,
        times,
        marker="o",
        linewidth=2,
        label=f"{distribution.capitalize()} MPI"
    )

    plt.axhline(
        serial,
        linestyle="--",
        alpha=0.6,
        color=line.get_color(),
        label=f"{distribution.capitalize()} serial"
    )

plt.title("Sample Sort — Strong-Scaling Runtime")
plt.xlabel("MPI processes")
plt.ylabel("Median execution time (seconds)")
plt.xticks(processes)
plt.grid(alpha=0.25)
plt.legend()

save_figure("01_scaling_runtime.png")


# ============================================
# FIGURE 2: Speedup versus serial baseline
# ============================================

plt.figure(figsize=(9, 5.5))

plt.plot(
    processes,
    processes,
    linestyle="--",
    label="Ideal linear speedup"
)

for distribution in distributions:

    plt.plot(
        processes,
        values(
            distribution,
            "speedup_vs_serial"
        ),
        marker="o",
        linewidth=2,
        label=distribution.capitalize()
    )

plt.title("Sample Sort — Speedup")
plt.xlabel("MPI processes")
plt.ylabel("Speedup relative to serial qsort")
plt.xticks(processes)
plt.ylim(bottom=0)
plt.grid(alpha=0.25)
plt.legend()

save_figure("02_scaling_speedup.png")


# ============================================
# FIGURE 3: Parallel efficiency
# ============================================

plt.figure(figsize=(9, 5.5))

plt.axhline(
    100,
    linestyle="--",
    label="Ideal efficiency (100%)"
)

for distribution in distributions:

    plt.plot(
        processes,
        values(
            distribution,
            "efficiency_percent"
        ),
        marker="o",
        linewidth=2,
        label=distribution.capitalize()
    )

plt.title("Sample Sort — Parallel Efficiency")
plt.xlabel("MPI processes")
plt.ylabel("Parallel efficiency (%)")
plt.xticks(processes)
plt.ylim(bottom=0)
plt.grid(alpha=0.25)
plt.legend()

save_figure("03_scaling_efficiency.png")


# ============================================
# FIGURE 4: Load imbalance
# ============================================

plt.figure(figsize=(9, 5.5))

plt.axhline(
    1.0,
    linestyle="--",
    label="Perfect balance"
)

for distribution in distributions:

    plt.plot(
        processes,
        values(
            distribution,
            "median_imbalance"
        ),
        marker="o",
        linewidth=2,
        label=distribution.capitalize()
    )

plt.title("Sample Sort — Partition Load Imbalance")
plt.xlabel("MPI processes")
plt.ylabel("Maximum count / mean count")
plt.xticks(processes)
plt.grid(alpha=0.25)
plt.legend()

save_figure("04_load_imbalance.png")


# ============================================
# FIGURES 5 AND 6: Phase-timing comparisons
# ============================================

# These are MEDIANS of per-phase maximum times
# across MPI ranks.
#
# Different phase maxima may come from
# different ranks, so they are not additive.
#
# We use GROUPED bars, not stacked total-runtime
# bars, to avoid misrepresenting the measurement.

phases = [
    ("local_sort_seconds", "Local sort"),
    ("sampling_seconds", "Sampling"),
    ("partition_seconds", "Partition/counts"),
    ("alltoallv_seconds", "Alltoallv"),
    ("final_sort_seconds", "Final sort")
]

for distribution in distributions:

    plt.figure(figsize=(11, 6))

    selected_processes = [2, 4, 6, 8]

    width = 0.16

    selected_rows = {
        int(row["processes"]): row
        for row in results[distribution]
    }

    for phase_index, (column, label) in enumerate(phases):

        horizontal_positions = [
            group_index +
            (phase_index - 2) * width
            for group_index in range(
                len(selected_processes)
            )
        ]

        phase_times = [
            float(
                selected_rows[p][column]
            )
            for p in selected_processes
        ]

        plt.bar(
            horizontal_positions,
            phase_times,
            width=width,
            label=label
        )

    plt.xticks(
        range(len(selected_processes)),
        selected_processes
    )

    plt.title(
        f"Sample Sort — Phase Timings "
        f"({distribution.capitalize()})"
    )

    plt.xlabel("MPI processes")
    plt.ylabel("Median phase maximum (seconds)")
    plt.grid(axis="y", alpha=0.25)

    plt.legend(
        ncol=3,
        fontsize=9
    )

    filename = (
        "05_uniform_phase_timings.png"
        if distribution == "uniform"
        else "06_duplicates_phase_timings.png"
    )

    save_figure(filename)


# ============================================
# FIGURE 7: Run-to-run variability
# ============================================

plt.figure(figsize=(9, 5.5))

for distribution in distributions:

    medians = []
    lower_errors = []
    upper_errors = []

    for p in processes:

        trials = sorted(
            float(row["total_seconds"])
            for row in raw
            if row["distribution"] == distribution
            and row["mode"] == "mpi"
            and int(row["processes"]) == p
        )

        if len(trials) != 3:
            raise ValueError(
                f"Expected three repetitions: "
                f"{distribution}, p={p}"
            )

        mid = median(trials)

        medians.append(mid)
        lower_errors.append(mid - min(trials))
        upper_errors.append(max(trials) - mid)

    plt.errorbar(
        processes,
        medians,
        yerr=[lower_errors, upper_errors],
        marker="o",
        linewidth=2,
        capsize=4,
        label=distribution.capitalize()
    )

plt.title("Sample Sort — Run-to-Run Variability")
plt.xlabel("MPI processes")
plt.ylabel("Execution time (seconds)")
plt.xticks(processes)
plt.grid(alpha=0.25)
plt.legend()

save_figure("07_timing_variability.png")


# ============================================
# STEP 4: Print verification summary
# ============================================

print("\n===== CS5 FIGURES COMPLETE =====")

for distribution in distributions:

    final = results[distribution][-1]

    print(
        f"{distribution.capitalize()}: "
        f"p=8, "
        f"speedup="
        f"{float(final['speedup_vs_serial']):.3f}x, "
        f"efficiency="
        f"{float(final['efficiency_percent']):.2f}%"
    )

print("\nTotal figures generated: 7")
print("Output directory:", FIGURES)
