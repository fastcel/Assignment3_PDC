
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================
# STEP 1: Configure paths
# ============================================

RESULTS = Path("results_cs4")
FIGURES = Path("figures_cs4")

FIGURES.mkdir(parents=True, exist_ok=True)

THREADS = [1, 2, 4, 6, 8]

SCHEDULES = ["static", "dynamic", "guided"]
ALGORITHMS = ["atomic_push", "dynamic_pull"]


# ============================================
# STEP 2: Load the saved benchmark CSVs
# ============================================

def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required benchmark file missing: {path}"
        )

    with path.open(newline="") as file:
        return list(csv.DictReader(file))


schedule_rows = load_csv(
    RESULTS / "scheduling_benchmark_summary.csv"
)

algorithm_rows = load_csv(
    RESULTS / "algorithm_comparison_summary.csv"
)

if len(schedule_rows) != 15:
    raise ValueError(
        "Expected 15 scheduling configurations"
    )

if len(algorithm_rows) != 10:
    raise ValueError(
        "Expected 10 algorithm configurations"
    )


# ============================================
# STEP 3: Organize measurements
# ============================================

scheduling = {}

for row in schedule_rows:
    key = (row["schedule"], int(row["threads"]))

    if key in scheduling:
        raise ValueError(
            f"Duplicate scheduling configuration: {key}"
        )

    scheduling[key] = {
        "time": float(row["median_time_seconds"]),
        "speedup": float(
            row["speedup_vs_serial_pull"]
        ),
        "efficiency": float(
            row["efficiency_percent"]
        )
    }


algorithms = {}

for row in algorithm_rows:
    key = (row["algorithm"], int(row["threads"]))

    if key in algorithms:
        raise ValueError(
            f"Duplicate algorithm configuration: {key}"
        )

    algorithms[key] = {
        "time": float(row["median_seconds"]),
        "speedup": float(
            row["speedup_vs_serial_push"]
        ),
        "efficiency": float(
            row["efficiency_percent"]
        )
    }


for name in SCHEDULES:
    for p in THREADS:
        if (name, p) not in scheduling:
            raise ValueError(
                f"Missing schedule: {name}, p={p}"
            )


for name in ALGORITHMS:
    for p in THREADS:
        if (name, p) not in algorithms:
            raise ValueError(
                f"Missing algorithm: {name}, p={p}"
            )


# ============================================
# STEP 4: Figure helper
# ============================================

def save_figure(filename):
    plt.tight_layout()
    output = FIGURES / filename

    plt.savefig(output, dpi=200)
    plt.close()

    print("Saved:", output)


# ============================================
# FIGURE 1: Scheduling execution time
# ============================================

plt.figure(figsize=(8, 5))

for name in SCHEDULES:

    values = [
        scheduling[(name, p)]["time"]
        for p in THREADS
    ]

    plt.plot(
        THREADS,
        values,
        marker="o",
        label=name.capitalize()
    )

plt.title("PageRank: OpenMP Scheduling Runtime")
plt.xlabel("OpenMP Threads")
plt.ylabel("Median Execution Time (seconds)")
plt.xticks(THREADS)
plt.grid(alpha=0.3)
plt.legend()

save_figure("01_scheduling_runtime.png")


# ============================================
# FIGURE 2: Scheduling speedup
# ============================================

plt.figure(figsize=(8, 5))

for name in SCHEDULES:

    values = [
        scheduling[(name, p)]["speedup"]
        for p in THREADS
    ]

    plt.plot(
        THREADS,
        values,
        marker="o",
        label=name.capitalize()
    )

# Ideal scaling line for reference.
plt.plot(
    THREADS,
    THREADS,
    linestyle="--",
    label="Ideal linear speedup"
)

plt.title("PageRank: Scheduling Speedup")
plt.xlabel("OpenMP Threads")
plt.ylabel("Speedup vs Serial Pull")
plt.xticks(THREADS)
plt.grid(alpha=0.3)
plt.legend()

save_figure("02_scheduling_speedup.png")


# ============================================
# FIGURE 3: Push vs pull execution time
# ============================================

plt.figure(figsize=(8, 5))

for name, label in [
    ("atomic_push", "Atomic Push"),
    ("dynamic_pull", "Dynamic Pull")
]:

    values = [
        algorithms[(name, p)]["time"]
        for p in THREADS
    ]

    plt.plot(
        THREADS,
        values,
        marker="o",
        label=label
    )

plt.title("PageRank: Atomic Push vs Dynamic Pull")
plt.xlabel("OpenMP Threads")
plt.ylabel("Median Execution Time (seconds)")
plt.xticks(THREADS)
plt.grid(alpha=0.3)
plt.legend()

save_figure("03_push_vs_pull_runtime.png")


# ============================================
# FIGURE 4: Speedup against serial push
# ============================================

plt.figure(figsize=(8, 5))

for name, label in [
    ("atomic_push", "Atomic Push"),
    ("dynamic_pull", "Dynamic Pull")
]:

    values = [
        algorithms[(name, p)]["speedup"]
        for p in THREADS
    ]

    plt.plot(
        THREADS,
        values,
        marker="o",
        label=label
    )

plt.axhline(
    1.0,
    linestyle="--",
    label="Serial Push Baseline"
)

plt.title("PageRank: Speedup vs Serial Push")
plt.xlabel("OpenMP Threads")
plt.ylabel("Speedup")
plt.xticks(THREADS)
plt.grid(alpha=0.3)
plt.legend()

save_figure("04_algorithm_speedup.png")


# ============================================
# FIGURE 5: Relative pull advantage
# ============================================

ratios = []

for p in THREADS:

    push_time = algorithms[
        ("atomic_push", p)
    ]["time"]

    pull_time = algorithms[
        ("dynamic_pull", p)
    ]["time"]

    if push_time <= 0 or pull_time <= 0:
        raise ValueError(
            "Non-positive execution time"
        )

    ratios.append(push_time / pull_time)


plt.figure(figsize=(8, 5))

plt.bar(
    [str(p) for p in THREADS],
    ratios
)

plt.axhline(
    1.0,
    linestyle="--",
    label="Equal execution time"
)

plt.title("Dynamic Pull Advantage Over Atomic Push")
plt.xlabel("OpenMP Threads")
plt.ylabel("Atomic Time / Pull Time")
plt.grid(axis="y", alpha=0.3)
plt.legend()

save_figure("05_pull_advantage.png")


# ============================================
# STEP 5: Print result summary
# ============================================

print("\n===== SCHEDULING COMPARISON =====")

for p in THREADS:

    print(f"\nThreads: {p}")

    for name in SCHEDULES:

        result = scheduling[(name, p)]

        print(
            f"  {name:8s}: "
            f"{result['time']:.6f}s, "
            f"speedup {result['speedup']:.3f}x"
        )


print("\n===== ALGORITHM COMPARISON =====")

for i, p in enumerate(THREADS):

    print(
        f"p={p}: "
        f"dynamic pull advantage "
        f"{ratios[i]:.3f}x"
    )


best_schedule = min(
    scheduling,
    key=lambda key: scheduling[key]["time"]
)

best_algorithm = min(
    algorithms,
    key=lambda key: algorithms[key]["time"]
)

print("\n===== FASTEST CONFIGURATIONS =====")

print(
    "Scheduling benchmark:",
    best_schedule,
    scheduling[best_schedule]["time"]
)

print(
    "Algorithm benchmark:",
    best_algorithm,
    algorithms[best_algorithm]["time"]
)

print("\n===== ALL FIVE CS4 FIGURES GENERATED =====")
