
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================
# STEP 1: Locate experiment files
# ============================================

RESULTS = Path("results_cs5")
FIGURES = Path("figures_cs5")

PINGPONG = RESULTS / "pingpong_p2.csv"
SUMMARY = RESULTS / "scaling_summary.csv"
TRAFFIC = RESULTS / "traffic_logs"

FIGURES.mkdir(exist_ok=True)

# ============================================
# STEP 2: Read ping-pong measurements
# ============================================

with PINGPONG.open(newline="") as file:
    pingpong = list(csv.DictReader(file))

alpha = next(
    float(row["one_way_seconds"])
    for row in pingpong
    if int(row["bytes"]) == 0
)

# Fit beta using messages >= 256 KiB.

large = [
    row for row in pingpong
    if int(row["bytes"]) >= 262144
]

x = [float(r["bytes"]) for r in large]
y = [float(r["one_way_seconds"]) for r in large]

mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

beta = sum(
    (a - mean_x) * (b - mean_y)
    for a, b in zip(x, y)
) / sum(
    (a - mean_x) ** 2 for a in x
)

# ============================================
# STEP 3: Read scaling measurements
# ============================================

with SUMMARY.open(newline="") as file:
    scaling = list(csv.DictReader(file))

def observed_time(distribution, p):
    matches = [
        row for row in scaling
        if row["distribution"] == distribution
        and int(row["processes"]) == p
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Missing scaling result: "
            f"{distribution}, p={p}"
        )

    return float(matches[0]["alltoallv_seconds"])

# ============================================
# STEP 4: Extract measured traffic counts
# ============================================

def extract_integer(text, label):

    match = re.search(
        rf"^{re.escape(label)}:\s*(\d+)\s*$",
        text,
        re.MULTILINE
    )

    if match is None:
        raise ValueError(
            f"Missing traffic field: {label}"
        )

    return int(match.group(1))


def read_traffic(distribution, p):

    path = TRAFFIC / f"{distribution}_p{p}.txt"

    content = path.read_text()

    if "Traffic count consistency: PASS" not in content:
        raise ValueError(
            f"Traffic verification failed: {path}"
        )

    return {
        "total_remote_bytes": extract_integer(
            content,
            "Total remote payload bytes"
        ),
        "max_send_bytes": extract_integer(
            content,
            "Max remote send bytes per rank"
        ),
        "max_receive_bytes": extract_integer(
            content,
            "Max remote receive bytes per rank"
        )
    }

# ============================================
# STEP 5: Compare predictions with observations
# ============================================

TOTAL = 100_000_000
KEY_BYTES = 8

rows = []

for distribution in ["uniform", "duplicates"]:
    for p in [4, 8]:

        traffic = read_traffic(distribution, p)

        average_remote_bytes = (
            traffic["total_remote_bytes"] / p
        )

        # Maximum observed outgoing or incoming
        # payload at a rank.

        critical_bytes = max(
            traffic["max_send_bytes"],
            traffic["max_receive_bytes"]
        )

        # This is a simplified estimate,
        # not a complete MPI collective model.

        predicted = (
            (p - 1) * alpha
            + beta * critical_bytes
        )

        observed = observed_time(distribution, p)

        rows.append({
            "distribution": distribution,
            "processes": p,
            "total_remote_MB":
                traffic["total_remote_bytes"] / 1e6,
            "mean_remote_MB_per_rank":
                average_remote_bytes / 1e6,
            "critical_payload_MB":
                critical_bytes / 1e6,
            "predicted_seconds": predicted,
            "observed_seconds": observed,
            "observed_over_predicted":
                observed / predicted
        })

# ============================================
# STEP 6: Display the comparison
# ============================================

print("===== TRAFFIC-ADJUSTED MODEL =====")

print(f"Alpha: {alpha * 1e6:.3f} us")
print(f"Beta: {beta * 1e9:.3f} ns/byte")

print(
    f"\n{'Distribution':<12} "
    f"{'P':>3} "
    f"{'Critical MB':>12} "
    f"{'Predicted':>11} "
    f"{'Observed':>11} "
    f"{'Obs/Pred':>10}"
)

for row in rows:
    print(
        f"{row['distribution']:<12} "
        f"{row['processes']:>3} "
        f"{row['critical_payload_MB']:>12.2f} "
        f"{row['predicted_seconds']:>11.5f} "
        f"{row['observed_seconds']:>11.5f} "
        f"{row['observed_over_predicted']:>9.2f}x"
    )

# ============================================
# STEP 7: Save the analysis CSV
# ============================================

csv_path = RESULTS / "traffic_adjusted_model.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", csv_path)

# ============================================
# STEP 8: Plot estimated and measured times
# ============================================

labels = [
    f"{row['distribution']}\np={row['processes']}"
    for row in rows
]

positions = list(range(len(rows)))
width = 0.35

plt.figure(figsize=(10, 5.5))

plt.bar(
    [i - width / 2 for i in positions],
    [r["predicted_seconds"] for r in rows],
    width,
    label="Traffic-adjusted estimate"
)

plt.bar(
    [i + width / 2 for i in positions],
    [r["observed_seconds"] for r in rows],
    width,
    label="Measured Alltoallv"
)

plt.xticks(positions, labels)

plt.ylabel("Communication time (seconds)")

plt.title(
    "MPI Alltoallv: Traffic-Adjusted "
    "Model vs Measurement"
)

plt.grid(axis="y", alpha=0.25)
plt.legend()

figure_path = (
    FIGURES / "10_traffic_adjusted_model.png"
)

plt.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:", figure_path)

print("\n===== ANALYSIS COMPLETE =====")
