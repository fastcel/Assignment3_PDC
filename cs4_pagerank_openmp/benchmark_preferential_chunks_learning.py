from pathlib import Path
import subprocess
import csv
import statistics
import re

GRAPH = "graphs_cs4/preferential_attachment_edges.txt"
THREADS = 4
REPEATS = 3

SCHEDULES = [
    "static",
    "dynamic",
    "guided",
]

CHUNKS = [1, 8, 32, 128, 512, 2048]

OUTDIR = Path("results_cs4/preferential_chunks")
LOGDIR = OUTDIR / "logs"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

def run_case(schedule, chunk, run_id):
    log_path = LOGDIR / f"{schedule}_chunk{chunk}_r{run_id}.txt"

    cmd = [
        "./pagerank_scheduling_learning",
        GRAPH,
        str(THREADS),
        schedule,
        str(chunk),
    ]

    env = {
        **dict(__import__("os").environ),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    output = completed.stdout + "\n" + completed.stderr
    log_path.write_text(output)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed for {schedule}, chunk={chunk}, run={run_id}"
        )

    if "Correctness: PASS" not in output:
        raise RuntimeError(
            f"Correctness failed for {schedule}, chunk={chunk}, run={run_id}"
        )

    m_time = re.search(
        r"OpenMP pull time:\s*([0-9.]+)\s*seconds",
        output
    )
    if not m_time:
        raise RuntimeError(
            f"Could not parse runtime for {schedule}, chunk={chunk}, run={run_id}"
        )

    m_speedup = re.search(
        r"Speedup vs serial pull:\s*([0-9.]+)x",
        output
    )
    if not m_speedup:
        raise RuntimeError(
            f"Could not parse speedup for {schedule}, chunk={chunk}, run={run_id}"
        )

    m_residual = re.search(
        r"Final residual:\s*([0-9.eE+-]+)",
        output
    )
    if not m_residual:
        raise RuntimeError(
            f"Could not parse residual for {schedule}, chunk={chunk}, run={run_id}"
        )

    return {
        "schedule": schedule,
        "chunk": chunk,
        "run": run_id,
        "time": float(m_time.group(1)),
        "speedup": float(m_speedup.group(1)),
        "residual": float(m_residual.group(1)),
    }

raw_rows = []

print("===== PREFERENTIAL-ATTACHMENT CHUNK BENCHMARK =====")
print(f"Graph: {GRAPH}")
print(f"Threads: {THREADS}")
print(f"Repeats: {REPEATS}")
print(f"Schedules: {SCHEDULES}")
print(f"Chunks: {CHUNKS}")

for schedule in SCHEDULES:
    for chunk in CHUNKS:
        print(f"\n===== {schedule} | chunk={chunk} =====")
        for run_id in range(1, REPEATS + 1):
            row = run_case(schedule, chunk, run_id)
            raw_rows.append(row)
            print(
                f"run={run_id} "
                f"time={row['time']:.6f}s "
                f"speedup={row['speedup']:.3f}x "
                f"PASS"
            )

raw_csv = OUTDIR / "chunk_raw.csv"
with raw_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["schedule", "chunk", "run", "time", "speedup", "residual"]
    )
    writer.writeheader()
    writer.writerows(raw_rows)

summary_rows = []

for schedule in SCHEDULES:
    for chunk in CHUNKS:
        subset = [
            r for r in raw_rows
            if r["schedule"] == schedule and r["chunk"] == chunk
        ]
        times = [r["time"] for r in subset]
        speedups = [r["speedup"] for r in subset]

        summary_rows.append({
            "schedule": schedule,
            "chunk": chunk,
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "median_speedup": statistics.median(speedups),
        })

summary_rows.sort(key=lambda r: (r["schedule"], r["chunk"]))

summary_csv = OUTDIR / "chunk_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "schedule",
            "chunk",
            "median_time",
            "min_time",
            "max_time",
            "median_speedup",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("\n===== MEDIAN SUMMARY =====")
for row in summary_rows:
    print(
        f"{row['schedule']:<8} "
        f"chunk={row['chunk']:<5} "
        f"time={row['median_time']:.6f}s "
        f"speedup={row['median_speedup']:.3f}x"
    )

best = min(summary_rows, key=lambda r: r["median_time"])

print("\n===== BEST CONFIGURATION =====")
print(f"Schedule: {best['schedule']}")
print(f"Chunk: {best['chunk']}")
print(f"Median time: {best['median_time']:.6f}s")
print(f"Median speedup: {best['median_speedup']:.3f}x")

print("\nSaved raw data:", raw_csv)
print("Saved summary:", summary_csv)
print("Saved logs:", LOGDIR)
print("\n===== BENCHMARK COMPLETE =====")
