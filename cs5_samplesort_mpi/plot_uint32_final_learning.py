import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================
# STEP 1: Paths and configuration
# ============================================

RESULTS = Path("results_cs5")
FIGURES = Path("figures_cs5")
LOGS = RESULTS / "uint32_scaling_logs"

FIGURES.mkdir(exist_ok=True)

DISTRIBUTIONS = ["uniform", "duplicates"]
PROCESSES = [1, 2, 4, 6, 8]

PHASES = [
    ("local_sort_s", "Local sort"),
    ("sampling_s", "Sampling"),
    ("partition_s", "Partition/counts"),
    ("alltoallv_s", "Alltoallv"),
    ("final_sort_s", "Final sort"),
]


# ============================================
# STEP 2: Load the benchmark CSV files
# ============================================

def load_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


summary = load_csv(
    RESULTS / "uint32_scaling_summary.csv"
)

raw = load_csv(
    RESULTS / "uint32_scaling_raw.csv"
)

if len(summary) != 10:
    raise ValueError(
        "Expected 10 summary configurations"
    )

if len(raw) != 36:
    raise ValueError(
        "Expected 36 benchmark executions"
    )

print("===== FINAL 32-BIT FIGURE GENERATION =====")
print("Summary configurations:", len(summary))
print("Raw executions:", len(raw))


# ============================================
# STEP 3: Draw the four scaling graphs
# ============================================

METRICS = [
    (
        "mpi_median_seconds",
        "Median MPI runtime (seconds)",
        "13_uint32_runtime.png",
        "Native 32-bit Sample Sort — Runtime"
    ),
    (
        "speedup",
        "Speedup against serial baseline",
        "14_uint32_speedup.png",
        "Native 32-bit Sample Sort — Speedup"
    ),
    (
        "efficiency_percent",
        "Parallel efficiency (%)",
        "15_uint32_efficiency.png",
        "Native 32-bit Sample Sort — Efficiency"
    ),
    (
        "median_imbalance",
        "Max / mean partition imbalance",
        "16_uint32_imbalance.png",
        "Native 32-bit Sample Sort — Load Imbalance"
    ),
]

for column, ylabel, filename, title in METRICS:

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for distribution in DISTRIBUTIONS:

        selected = sorted(
            (
                row for row in summary
                if row["distribution"] == distribution
            ),
            key=lambda row: int(row["processes"])
        )

        x = [
            int(row["processes"])
            for row in selected
        ]

        y = [
            float(row[column])
            for row in selected
        ]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=distribution.capitalize()
        )

    if column == "speedup":
        ax.plot(
            PROCESSES,
            PROCESSES,
            linestyle="--",
            label="Ideal"
        )

    if column == "median_imbalance":
        ax.axhline(
            1.0,
            linestyle="--",
            label="Perfect balance"
        )

    ax.set_title(title)
    ax.set_xlabel("MPI processes")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PROCESSES)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()

    output = FIGURES / filename

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output)


# ============================================
# STEP 4: Read per-rank phase measurements
# ============================================

START = "===== PER-RANK PHASE TIMINGS ====="
END = "===== END PER-RANK PHASE TIMINGS ====="


def read_phases(distribution, p, run):

    path = LOGS / (
        f"{distribution}_p{p}_r{run}.txt"
    )

    text = path.read_text()

    if "Correctness: PASS" not in text:
        raise ValueError(
            f"Incorrect result: {path}"
        )

    if START not in text or END not in text:
        raise ValueError(
            f"Missing phase timings: {path}"
        )

    block = text.split(START, 1)[1]
    block = block.split(END, 1)[0].strip()

    rows = list(
        csv.DictReader(io.StringIO(block))
    )

    if len(rows) != p:
        raise ValueError(
            f"Expected {p} ranks in {path}"
        )

    parsed = []

    for row in rows:

        phase_times = {
            key: float(row[key])
            for key, _ in PHASES
        }

        total = float(
            row["phase_window_s"]
        )

        if abs(
            sum(phase_times.values()) - total
        ) > 0.00001:
            raise ValueError(
                f"Phase sum mismatch: {path}"
            )

        parsed.append({
            "rank": int(row["rank"]),
            "total": total,
            **phase_times,
        })

    if sorted(
        row["rank"] for row in parsed
    ) != list(range(p)):
        raise ValueError(
            f"Rank ID mismatch: {path}"
        )

    return max(
        parsed,
        key=lambda row: row["total"]
    )


# ============================================
# STEP 5: Select representative median runs
# ============================================

representative = []

for distribution in DISTRIBUTIONS:

    for p in PROCESSES:

        selected = [
            row for row in raw
            if row["distribution"] == distribution
            and row["mode"] == "mpi"
            and int(row["processes"]) == p
        ]

        if len(selected) != 3:
            raise ValueError(
                f"Expected 3 runs for {distribution}, p={p}"
            )

        # Middle runtime out of three actual runs.
        median_run = sorted(
            selected,
            key=lambda row: float(row["seconds"])
        )[1]

        run_id = int(median_run["run"])

        bottleneck = read_phases(
            distribution, p, run_id
        )

        if abs(
            bottleneck["total"]
            - float(median_run["seconds"])
        ) > 0.00001:
            raise ValueError(
                f"Runtime mismatch: {distribution}, p={p}"
            )

        representative.append({
            "distribution": distribution,
            "processes": p,
            "run": run_id,
            "bottleneck_rank": bottleneck["rank"],
            "total_seconds": bottleneck["total"],
            **{
                key: bottleneck[key]
                for key, _ in PHASES
            }
        })


# ============================================
# STEP 6: Save representative phase data
# ============================================

phase_csv = (
    RESULTS / "uint32_phase_representative.csv"
)

with phase_csv.open("w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(representative[0].keys())
    )

    writer.writeheader()
    writer.writerows(representative)

print("Saved:", phase_csv)


# ============================================
# STEP 7: Create the stacked phase charts
# ============================================

for distribution in DISTRIBUTIONS:

    selected = [
        row for row in representative
        if row["distribution"] == distribution
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = list(range(len(selected)))
    bottoms = [0.0] * len(selected)

    for key, label in PHASES:

        heights = [
            row[key] for row in selected
        ]

        ax.bar(
            positions,
            heights,
            bottom=bottoms,
            label=label,
            width=0.65
        )

        bottoms = [
            a + b
            for a, b in zip(bottoms, heights)
        ]

    for i, row in enumerate(selected):

        ax.text(
            i,
            row["total_seconds"] * 1.02,
            (
                f"Rank {row['bottleneck_rank']}\n"
                f"Run {row['run']}"
            ),
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax.set_xticks(positions)

    ax.set_xticklabels([
        f"{row['processes']} processes"
        for row in selected
    ])

    ax.set_title(
        "Native 32-bit Sample Sort — "
        "Stacked Phase Breakdown\n"
        f"100M {distribution} integers"
    )

    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Phase-window time (seconds)")

    ax.set_ylim(
        0,
        max(
            row["total_seconds"]
            for row in selected
        ) * 1.2
    )

    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)

    filename = (
        "17_uint32_uniform_stacked_phases.png"
        if distribution == "uniform"
        else "18_uint32_duplicates_stacked_phases.png"
    )

    output = FIGURES / filename

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output)


# ============================================
# STEP 8: Final verification
# ============================================

print("\n===== REPRESENTATIVE PHASE RUNS =====")

for row in representative:
    print(
        f"{row['distribution']:<11} "
        f"p={row['processes']} "
        f"run={row['run']} "
        f"rank={row['bottleneck_rank']} "
        f"time={row['total_seconds']:.6f}s "
        "PASS"
    )

print("\n===== FINAL FIGURE VALIDATION =====")
print("Scaling figures: 4")
print("Stacked phase figures: 2")
print("Verified configurations: 10")
print("Status: PASS")
