from array import array
from pathlib import Path
import random

# ============================================
# STEP 1: Graph configuration
# ============================================

NODES = 100_000
EDGES_PER_NEW_NODE = 13
SEED = 20261121

OUTPUT = Path(
    "graphs_cs4/preferential_attachment_edges.txt"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

rng = random.Random(SEED)

# Begin with a fully connected seed network.
seed_nodes = EDGES_PER_NEW_NODE + 1

initial_edges = (
    seed_nodes * (seed_nodes - 1) // 2
)

expected_edges = (
    initial_edges
    + (NODES - seed_nodes) * EDGES_PER_NEW_NODE
)

# ============================================
# STEP 2: Track vertex degrees
# ============================================

in_degree = [0] * NODES
out_degree = [0] * NODES

# Each undirected connection contributes
# both endpoints to this sampling pool.
#
# A node with higher degree appears more
# often, so it is more likely to be selected.
#
# array('I') avoids the memory overhead
# of a large Python integer list.

target_pool = array("I")

written_edges = 0

print("===== PREFERENTIAL ATTACHMENT GRAPH =====")
print("Vertices:", NODES)
print("Edges per new vertex:", EDGES_PER_NEW_NODE)
print("Expected edges:", expected_edges)
print("Seed:", SEED)

# ============================================
# STEP 3: Generate directed edge list
# ============================================

with OUTPUT.open(
    "w",
    buffering=1024 * 1024
) as file:

    file.write(
        f"{NODES} {expected_edges}\n"
    )

    # Construct the initial clique.
    #
    # For each undirected connection,
    # write one directed edge.
    #
    # Sources remain sorted.

    for u in range(seed_nodes):
        for v in range(u):

            file.write(f"{u} {v}\n")

            out_degree[u] += 1
            in_degree[v] += 1

            target_pool.extend((u, v))

            written_edges += 1

    # ========================================
    # STEP 4: Preferential attachment
    # ========================================

    for u in range(seed_nodes, NODES):

        targets = set()

        # Select distinct existing vertices
        # with probability proportional to
        # their current degree.
        #
        # Do not add the new vertex to the
        # sampling pool until its targets
        # have all been selected.

        while len(targets) < EDGES_PER_NEW_NODE:

            index = rng.randrange(
                len(target_pool)
            )

            v = target_pool[index]

            targets.add(v)

        # Write edges in source order,
        # then destination order.

        for v in sorted(targets):

            file.write(f"{u} {v}\n")

            out_degree[u] += 1
            in_degree[v] += 1

            written_edges += 1

        # Update the sampling weights after
        # selecting this new node's targets.

        for v in targets:
            target_pool.extend((u, v))

# ============================================
# STEP 5: Verify graph statistics
# ============================================

assert written_edges == expected_edges
assert sum(in_degree) == expected_edges
assert sum(out_degree) == expected_edges

print("\n===== VERIFICATION =====")
print("Actual edges:", written_edges)

print(
    "Maximum indegree:",
    max(in_degree)
)

print(
    "Maximum outdegree:",
    max(out_degree)
)

print(
    "Vertices with zero indegree:",
    sum(d == 0 for d in in_degree)
)

print(
    "Vertices with zero outdegree:",
    sum(d == 0 for d in out_degree)
)

print("Graph consistency: PASS")
print("Output:", OUTPUT)
