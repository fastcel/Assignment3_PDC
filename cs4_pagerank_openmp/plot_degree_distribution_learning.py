from collections import Counter
from pathlib import Path
import csv

import matplotlib.pyplot as plt

# ============================================
# STEP 1: Input/output files
# ============================================

GRAPH = Path(
    "graphs_cs4/preferential_attachment_edges.txt"
)

FIGURES = Path("figures_cs4")
RESULTS = Path("results_cs4")

FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ============================================
# STEP 2: Read graph and calculate degrees
# ============================================

with GRAPH.open() as file:

    n, expected_edges = map(
        int, file.readline().split()
    )

    indegree = [0] * n
    outdegree = [0] * n

    actual_edges = 0

    for line in file:
        source, target = map(int, line.split())

        if not (0 <= source < n):
            raise ValueError("Invalid source vertex")

        if not (0 <= target < n):
            raise ValueError("Invalid destination vertex")

        outdegree[source] += 1
        indegree[target] += 1

        actual_edges += 1

if actual_edges != expected_edges:
    raise ValueError("Graph edge count mismatch")

assert sum(indegree) == actual_edges
assert sum(outdegree) == actual_edges

print("===== DEGREE DISTRIBUTION ANALYSIS =====")
print("Vertices:", n)
print("Edges:", actual_edges)

print("Maximum indegree:", max(indegree))
print("Maximum outdegree:", max(outdegree))

print(
    "Zero-indegree vertices:",
    sum(degree == 0 for degree in indegree)
)

# ============================================
# STEP 3: Calculate hub concentration
# ============================================

top_count = max(1, n // 100)

top_degrees = sorted(
    indegree, reverse=True
)[:top_count]

top_edges = sum(top_degrees)

top_percentage = (
    100.0 * top_edges / actual_edges
)

print(
    f"Incoming edges targeting the top "
    f"1% of vertices: {top_percentage:.2f}%"
)

# ============================================
# STEP 4: Degree-frequency histogram
# ============================================

frequency = Counter(indegree)

degrees = sorted(
    degree
    for degree in frequency
    if degree > 0
)

probabilities = [
    frequency[degree] / n
    for degree in degrees
]

# ============================================
# STEP 5: Save numerical distribution
# ============================================

csv_path = (
    RESULTS / "indegree_distribution.csv"
)

with csv_path.open("w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "indegree",
        "vertex_count",
        "vertex_fraction"
    ])

    for degree in sorted(frequency):

        writer.writerow([
            degree,
            frequency[degree],
            frequency[degree] / n
        ])

print("Saved:", csv_path)

# ============================================
# STEP 6: Log-log degree-distribution plot
# ============================================

plt.figure(figsize=(9, 5.5))

plt.loglog(
    degrees,
    probabilities,
    "o",
    markersize=3,
    alpha=0.65
)

plt.title(
    "Preferential-Attachment Graph — "
    "In-Degree Distribution"
)

plt.xlabel("In-degree k (log scale)")

plt.ylabel(
    "Fraction of vertices with in-degree k "
    "(log scale)"
)

plt.grid(
    True,
    which="both",
    alpha=0.2
)

figure_path = (
    FIGURES /
    "06_indegree_distribution_loglog.png"
)

plt.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:", figure_path)

# ============================================
# STEP 7: Check static scheduling workload
# ============================================

threads = 4

chunk = (
    n + threads - 1
) // threads

work = [0] * threads

for vertex, degree in enumerate(indegree):

    tid = min(
        vertex // chunk,
        threads - 1
    )

    work[tid] += degree

mean_work = sum(work) / threads

edge_imbalance = (
    max(work) / mean_work
)

print("\n===== STATIC WORKLOAD CHECK =====")

for tid, edges in enumerate(work):

    print(
        f"Thread {tid}: "
        f"{edges:,} incoming edges"
    )

print(
    f"Max/mean edge imbalance: "
    f"{edge_imbalance:.3f}x"
)

print("\n===== ANALYSIS COMPLETE =====")
