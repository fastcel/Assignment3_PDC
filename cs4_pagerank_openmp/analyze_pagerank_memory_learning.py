import csv
import re
from pathlib import Path

RESULTS = Path("results_cs4")

push_text = (
    RESULTS / "preferential_atomic_push_p4.txt"
).read_text()

pull_text = (
    RESULTS / "preferential_dynamic_pull_p4.txt"
).read_text()


def field(text, label):
    match = re.search(
        rf"^{re.escape(label)}:\s*"
        rf"([0-9]+(?:\.[0-9]+)?)",
        text,
        re.MULTILINE
    )

    if match is None:
        raise ValueError(f"Missing {label}")

    return float(match.group(1))


def section(text, heading):
    marker = f"===== {heading} ====="

    if marker not in text:
        raise ValueError(f"Missing {heading}")

    return text.split(marker, 1)[1].split(
        "=====", 1
    )[0]


if (
    "Correctness: PASS" not in push_text
    or "Correctness: PASS" not in pull_text
):
    raise ValueError("Correctness verification missing")


edges = int(field(push_text, "Edges"))
vertices = int(field(push_text, "Vertices"))

if int(field(pull_text, "Edges")) != edges:
    raise ValueError("Graph mismatch")

push_serial = section(push_text, "SERIAL PUSH")
push_parallel = section(
    push_text, "OPENMP ATOMIC PUSH"
)

iterations = int(
    field(push_parallel, "Iterations")
)

if (
    int(field(push_serial, "Iterations")) != iterations
    or int(field(pull_text, "OpenMP pull iterations"))
       != iterations
    or int(field(pull_text, "Serial pull iterations"))
       != iterations
):
    raise ValueError("Iteration counts differ")


times = {
    "Serial push": field(
        push_serial, "Execution time"
    ),
    "Atomic push": field(
        push_parallel, "Execution time"
    ),
    "Serial pull": field(
        pull_text, "Serial pull time"
    ),
    "Dynamic pull": field(
        pull_text, "OpenMP pull time"
    ),
}

edge_visits = edges * iterations

print("===== PAGERANK MEMORY ANALYSIS =====")
print(f"Vertices: {vertices:,}")
print(f"Edges: {edges:,}")
print(f"Iterations: {iterations}")
print(f"Total edge visits: {edge_visits:,}")

print("\n===== EDGE PROCESSING THROUGHPUT =====")

rows = []

for algorithm, seconds in times.items():
    throughput = edge_visits / seconds / 1e6

    rows.append({
        "algorithm": algorithm,
        "seconds": seconds,
        "million_edges_per_second": throughput
    })

    print(
        f"{algorithm:<16} "
        f"{seconds:.6f} s | "
        f"{throughput:.2f} million edges/s"
    )


print("\n===== NOMINAL PULL MEMORY ESTIMATE =====")

bytes_per_edge = 4 + 8 + 16

logical_bytes = edge_visits * bytes_per_edge

pull_seconds = times["Dynamic pull"]

nominal_bandwidth = (
    logical_bytes / pull_seconds / 1e9
)

print(f"Estimated bytes/edge: {bytes_per_edge}")
print(
    f"Total logical reads: "
    f"{logical_bytes / 1e9:.3f} GB"
)

print(
    f"Nominal access rate: "
    f"{nominal_bandwidth:.3f} GB/s"
)

print(
    "\nNOTE: This is NOT measured DRAM bandwidth."
)
print(
    "Caching, reused data and other memory "
    "operations are not modeled."
)


output = RESULTS / "pagerank_memory_analysis.csv"

with output.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", output)
print("===== ANALYSIS COMPLETE =====")
