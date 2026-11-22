
from pathlib import Path
import random

# ============================================
# STEP 1: Graph configuration
# ============================================

NODES = 100_000
SEED = 20261121

DANGLING_END = 1_000
HEAVY_END = 10_000

HEAVY_DEGREE = 80
REGULAR_DEGREE = 6

HOT_VERTICES = 200

OUTPUT = Path("graphs_cs4/synthetic_edges.txt")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# ============================================
# STEP 2: Define the outdegree distribution
# ============================================

def outdegree(vertex):

    if vertex < DANGLING_END:
        return 0

    if vertex < HEAVY_END:
        return HEAVY_DEGREE

    return REGULAR_DEGREE


EXPECTED_EDGES = sum(
    outdegree(u) for u in range(NODES)
)

print("===== GENERATING SYNTHETIC GRAPH =====")
print(f"Vertices: {NODES:,}")
print(f"Expected edges: {EXPECTED_EDGES:,}")
print(f"Random seed: {SEED}")


# ============================================
# STEP 3: Generate a directed edge list
# ============================================

rng = random.Random(SEED)

edge_count = 0
hot_targeted_edges = 0

with OUTPUT.open("w", buffering=1024 * 1024) as file:

    # File header: number of vertices and edges
    file.write(f"{NODES} {EXPECTED_EDGES}\n")

    for u in range(NODES):

        degree = outdegree(u)

        for _ in range(degree):

            # Every fifth generated edge is
            # deliberately directed to one
            # of 200 popular destination nodes.
            #
            # The remaining destinations are
            # selected across the full graph.

            if edge_count % 5 == 0:
                v = rng.randrange(HOT_VERTICES)
                hot_targeted_edges += 1
            else:
                v = rng.randrange(NODES)

            file.write(f"{u} {v}\n")

            edge_count += 1

assert edge_count == EXPECTED_EDGES


# ============================================
# STEP 4: Verify the saved graph
# ============================================

print("\n===== VERIFYING SAVED GRAPH =====")

actual_degrees = [0] * NODES
actual_edges = 0

with OUTPUT.open("r") as file:

    header = file.readline().split()

    assert len(header) == 2

    file_nodes, file_edges = map(int, header)

    assert file_nodes == NODES
    assert file_edges == EXPECTED_EDGES

    previous_source = -1

    for line_number, line in enumerate(file, start=2):

        parts = line.split()

        if len(parts) != 2:
            raise ValueError(
                f"Invalid edge at line {line_number}"
            )

        u, v = map(int, parts)

        if not (0 <= u < NODES):
            raise ValueError(
                f"Invalid source at line {line_number}"
            )

        if not (0 <= v < NODES):
            raise ValueError(
                f"Invalid destination at line {line_number}"
            )

        if u < previous_source:
            raise ValueError(
                "Edges are not sorted by source"
            )

        previous_source = u
        actual_degrees[u] += 1
        actual_edges += 1

assert actual_edges == EXPECTED_EDGES

for u in range(NODES):

    if actual_degrees[u] != outdegree(u):
        raise ValueError(
            f"Outdegree mismatch at vertex {u}"
        )


# ============================================
# STEP 5: Print summary statistics
# ============================================

dangling = sum(
    degree == 0 for degree in actual_degrees
)

heavy = sum(
    degree == HEAVY_DEGREE
    for degree in actual_degrees
)

regular = sum(
    degree == REGULAR_DEGREE
    for degree in actual_degrees
)

print(f"Actual vertices: {NODES:,}")
print(f"Actual edges: {actual_edges:,}")

print(f"Dangling vertices: {dangling:,}")
print(f"Heavy vertices: {heavy:,}")
print(f"Regular vertices: {regular:,}")

print(f"Minimum outdegree: {min(actual_degrees)}")
print(f"Maximum outdegree: {max(actual_degrees)}")

print(
    f"Intentionally hot-targeted edges: "
    f"{hot_targeted_edges:,}"
)

print(f"\nGraph saved: {OUTPUT}")

print("\nGraph verification: PASS")
