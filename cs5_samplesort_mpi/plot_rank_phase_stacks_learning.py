import csv
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================
# STEP 1: Configuration
# ============================================

LOGS = Path("results_cs5/rank_phase_logs")
RESULTS = Path("results_cs5")
FIGURES = Path("figures_cs5")

PROCESSES = [2, 4, 6, 8]
DISTRIBUTIONS = ["uniform", "duplicates"]

PHASES = [
    ("local_sort_s", "Local sort"),
    ("sampling_s", "Sampling"),
    ("partition_s", "Partition/counts"),
    ("alltoallv_s", "Alltoallv"),
    ("final_sort_s", "Final sort"),
]

START = "===== PER-RANK PHASE TIMINGS ====="
END = "===== END PER-RANK PHASE TIMINGS ====="

FIGURES.mkdir(exist_ok=True)


# ============================================
# STEP 2: Read and validate each log
# ============================================

def read_log(distribution, processes):

    path = LOGS / (
        f"{distribution}_100m_p{processes}.txt"
    )

    text = path.read_text()

    if "Correctness: PASS" not in text:
        raise ValueError(
            f"Correctness missing in {path}"
        )

    if START not in text or END not in text:
        raise ValueError(
            f"Phase data missing in {path}"
        )

    block = text.split(START, 1)[1]
    block = block.split(END, 1)[0].strip()

    rows = list(
        csv.DictReader(io.StringIO(block))
    )

    if len(rows) != processes:
        raise ValueError(
            f"Expected {processes} ranks, "
            f"found {len(rows)} in {path}"
        )

    parsed = []

    for row in rows:

        phases = {
            key: float(row[key])
            for key, _ in PHASES
        }

        total = float(row["phase_window_s"])

        phase_sum = sum(phases.values())

        if abs(phase_sum - total) > 1e-6:
            raise ValueError(
                f"Phase sum mismatch in {path}, "
                f"rank {row['rank']}"
            )

        parsed.append({
            "rank": int(row["rank"]),
            "total": total,
            **phases,
        })

    ranks = sorted(
        row["rank"] for row in parsed
    )

    if ranks != list(range(processes)):
        raise ValueError(
            f"Unexpected rank IDs in {path}"
        )

    bottleneck = max(
        parsed,
        key=lambda row: row["total"]
    )

    # Verify that the bottleneck-rank timing
    # agrees with the program's reported
    # maximum sorting window.

    match = re.search(
        r"^Total sorting time:\s*"
        r"([0-9.]+)\s*s",
        text,
        re.MULTILINE
    )

    if match is None:
        raise ValueError(
            f"Missing total sorting time in {path}"
        )

    reported = float(match.group(1))

    if abs(bottleneck["total"] - reported) > 1e-5:
        raise ValueError(
            f"Reported timing mismatch in {path}"
        )

    return bottleneck


# ============================================
# STEP 3: Select the bottleneck rank
# ============================================

summary = []

print("===== BOTTLENECK-RANK PHASE ANALYSIS =====")

for distribution in DISTRIBUTIONS:

    for p in PROCESSES:

        result = read_log(distribution, p)

        summary.append({
            "distribution": distribution,
            "processes": p,
            "bottleneck_rank": result["rank"],
            "total_seconds": result["total"],
            **{
                key: result[key]
                for key, _ in PHASES
            }
        })

        print(
            f"{distribution:<11} "
            f"p={p} "
            f"rank={result['rank']} "
            f"total={result['total']:.6f}s "
            "PASS"
        )


# ============================================
# STEP 4: Save validated phase data
# ============================================

csv_path = (
    RESULTS / "rank_phase_bottleneck_summary.csv"
)

with csv_path.open("w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(summary[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary)

print("\nSaved:", csv_path)


# ============================================
# STEP 5: Create stacked charts
# ============================================

for distribution in DISTRIBUTIONS:

    selected = [
        row for row in summary
        if row["distribution"] == distribution
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = list(range(len(selected)))
    bottoms = [0.0] * len(selected)

    for key, label in PHASES:

        heights = [
            row[key] for row in selected
        ]

        ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=label,
            width=0.65
        )

        bottoms = [
            current + height
            for current, height
            in zip(bottoms, heights)
        ]

    for index, row in enumerate(selected):

        ax.text(
            index,
            row["total_seconds"] + 0.08,
            f"Rank {row['bottleneck_rank']}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_xticks(x)

    ax.set_xticklabels([
        f"{row['processes']} processes"
        for row in selected
    ])

    ax.set_title(
        "MPI Sample Sort — Bottleneck-Rank "
        f"Phase Breakdown\n"
        f"100M elements | "
        f"{distribution.capitalize()} input"
    )

    ax.set_xlabel("MPI process count")
    ax.set_ylabel("Phase-window time (seconds)")

    ax.set_ylim(
        0,
        max(row["total_seconds"] for row in selected) * 1.18
    )

    ax.grid(axis="y", alpha=0.25)

    ax.legend(
        title="Phase",
        fontsize=9,
        ncol=3
    )

    filename = (
        "11_uniform_stacked_rank_phases.png"
        if distribution == "uniform"
        else "12_duplicates_stacked_rank_phases.png"
    )

    output = FIGURES / filename

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output)


print("\n===== VALIDATION COMPLETE =====")
print("All 8 logs parsed successfully.")
print("All stacked phase sums verified.")
