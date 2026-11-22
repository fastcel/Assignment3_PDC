
import csv
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

# ============================================
# STEP 1: Locate existing experimental data
# ============================================

RESULTS = Path("results_cs5")
FIGURES = Path("figures_cs5")

FIGURES.mkdir(exist_ok=True)

PINGPONG = RESULTS / "pingpong_p2.csv"
SCALING = RESULTS / "scaling_raw.csv"

for path in [PINGPONG, SCALING]:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")

with PINGPONG.open(newline="") as file:
    pingpong = list(csv.DictReader(file))

with SCALING.open(newline="") as file:
    scaling = list(csv.DictReader(file))


# ============================================
# STEP 2: Extract latency estimate
# ============================================

zero_byte = [
    row for row in pingpong
    if int(row["bytes"]) == 0
]

if len(zero_byte) != 1:
    raise ValueError(
        "Expected one zero-byte measurement."
    )

alpha = float(
    zero_byte[0]["one_way_seconds"]
)


# ============================================
# STEP 3: Fit the large-message model
# ============================================

# Use messages >= 256 KiB.
# Ordinary least squares:
#
# time = intercept + beta * bytes

large = [
    row for row in pingpong
    if int(row["bytes"]) >= 262144
]

x = [
    float(row["bytes"])
    for row in large
]

y = [
    float(row["one_way_seconds"])
    for row in large
]

if len(x) < 2:
    raise ValueError(
        "Insufficient large-message measurements."
    )

mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

numerator = sum(
    (a - mean_x) * (b - mean_y)
    for a, b in zip(x, y)
)

denominator = sum(
    (a - mean_x) ** 2
    for a in x
)

beta = numerator / denominator
intercept = mean_y - beta * mean_x

if beta <= 0:
    raise ValueError(
        "Unexpected non-positive transfer slope."
    )

throughput_gbps = 1.0 / beta / 1e9


# ============================================
# STEP 4: Display fitted parameters
# ============================================

print("===== MPI COMMUNICATION MODEL =====")

print(
    f"Zero-byte latency alpha: "
    f"{alpha * 1e6:.3f} microseconds"
)

print(
    f"Large-message beta: "
    f"{beta * 1e9:.3f} ns/byte"
)

print(
    f"Fitted bandwidth: "
    f"{throughput_gbps:.3f} GB/s"
)

print(
    f"Large-message intercept: "
    f"{intercept * 1e6:.3f} microseconds"
)


# ============================================
# STEP 5: Compare with actual Alltoallv
# ============================================

TOTAL = 100_000_000
KEY_BYTES = 8
PROCESS_COUNTS = [2, 4, 6, 8]
DISTRIBUTIONS = ["uniform", "duplicates"]

comparisons = []

for p in PROCESS_COUNTS:

    # Approximate initially owned bytes
    # per MPI rank.

    bytes_per_rank = TOTAL * KEY_BYTES / p

    # Assume approximately 1/p of each
    # rank's data remains on that rank.
    #
    # Thus this estimates bytes sent
    # to OTHER ranks, not exact traffic.

    estimated_remote_bytes = (
        bytes_per_rank * (p - 1) / p
    )

    # Very simple communication estimate.
    # This deliberately ignores network
    # contention, collective overhead,
    # and uneven message destinations.

    predicted = (
        (p - 1) * alpha
        + beta * estimated_remote_bytes
    )

    for distribution in DISTRIBUTIONS:

        trials = [
            float(row["alltoallv_seconds"])
            for row in scaling
            if row["distribution"] == distribution
            and row["mode"] == "mpi"
            and int(row["processes"]) == p
        ]

        if len(trials) != 3:
            raise ValueError(
                f"Expected 3 Alltoallv timings: "
                f"{distribution}, p={p}"
            )

        observed = median(trials)

        comparisons.append({
            "distribution": distribution,
            "processes": p,
            "estimated_remote_bytes_per_rank":
                round(estimated_remote_bytes),
            "predicted_seconds": predicted,
            "observed_seconds": observed,
            "observed_over_predicted":
                observed / predicted
        })


# ============================================
# STEP 6: Print model comparison
# ============================================

print("\n===== ALLTOALLV MODEL COMPARISON =====")

print(
    f"{'Input':<12} "
    f"{'P':>3} "
    f"{'Predicted(s)':>14} "
    f"{'Observed(s)':>14} "
    f"{'Obs/Pred':>10}"
)

for row in comparisons:

    print(
        f"{row['distribution']:<12} "
        f"{row['processes']:>3} "
        f"{row['predicted_seconds']:>14.6f} "
        f"{row['observed_seconds']:>14.6f} "
        f"{row['observed_over_predicted']:>10.2f}x"
    )


# ============================================
# STEP 7: Save comparison as CSV
# ============================================

output_csv = RESULTS / "communication_model.csv"

with output_csv.open("w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(comparisons[0].keys())
    )

    writer.writeheader()
    writer.writerows(comparisons)

print("\nSaved:", output_csv)


# ============================================
# STEP 8: Graph the large-message fit
# ============================================

plt.figure(figsize=(9, 5.5))

message_mib = [
    value / (1024 * 1024)
    for value in x
]

observed_ms = [
    value * 1000
    for value in y
]

fitted_ms = [
    (intercept + beta * value) * 1000
    for value in x
]

plt.scatter(
    message_mib,
    observed_ms,
    label="Measured ping-pong",
    s=60
)

plt.plot(
    message_mib,
    fitted_ms,
    linestyle="--",
    label="Large-message linear fit"
)

plt.title("MPI Ping-Pong — Large Message Fit")
plt.xlabel("Message size (MiB)")
plt.ylabel("One-way time (ms)")
plt.grid(alpha=0.25)
plt.legend()

path = FIGURES / "08_pingpong_model.png"

plt.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)


# ============================================
# STEP 9: Compare predicted and observed
# ============================================

plt.figure(figsize=(9, 5.5))

predicted = [
    next(
        row["predicted_seconds"]
        for row in comparisons
        if row["processes"] == p
    )
    for p in PROCESS_COUNTS
]

plt.plot(
    PROCESS_COUNTS,
    predicted,
    linestyle="--",
    marker="o",
    label="Simple ping-pong-based prediction"
)

for distribution in DISTRIBUTIONS:

    observed = [
        next(
            row["observed_seconds"]
            for row in comparisons
            if row["distribution"] == distribution
            and row["processes"] == p
        )
        for p in PROCESS_COUNTS
    ]

    plt.plot(
        PROCESS_COUNTS,
        observed,
        marker="o",
        label=f"{distribution.capitalize()} observed"
    )

plt.title(
    "MPI Alltoallv — Predicted vs Observed"
)

plt.xlabel("MPI processes")
plt.ylabel("Redistribution time (seconds)")
plt.xticks(PROCESS_COUNTS)
plt.ylim(bottom=0)
plt.grid(alpha=0.25)
plt.legend()

path = FIGURES / "09_alltoallv_model_comparison.png"

plt.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)

print("\n===== COMMUNICATION ANALYSIS COMPLETE =====")
