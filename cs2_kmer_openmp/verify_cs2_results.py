
import csv
import re
from pathlib import Path
from collections import Counter

ROOT = Path(".")
RESULTS = ROOT / "results_cs2"
LOGS = RESULTS / "logs"
FIGURES = ROOT / "figures_cs2"

THREADS = (1, 2, 4, 6)
K_VALUES = (8, 12)
RUNS = (1, 2, 3)

EXPECTED = {
    8: {
        "windows": 99999993,
        "fingerprint": "605ab0a06ea0f33e"
    },
    12: {
        "windows": 99999989,
        "fingerprint": "fdafb38ee6cbd6e0"
    }
}

FIGURE_NAMES = [
    "01_runtime_k8.png",
    "02_speedup_k8.png",
    "03_runtime_k12.png",
    "04_speedup_k12.png",
    "05_false_sharing.png",
    "06_throughput.png"
]

errors = []

def check(condition, message):
    if not condition:
        errors.append(message)
        print("FAIL:", message)

def field(text, label):
    match = re.search(
        rf"^{re.escape(label)}:\s*(\S+)",
        text,
        re.MULTILINE
    )
    return match.group(1) if match else None

print("===== CS2 RESULTS AUDIT =====")

# ----------------------------------------
# 1. Check benchmark CSV
# ----------------------------------------

csv_path = RESULTS / "benchmark.csv"

if csv_path.is_file():
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    check(len(rows) == 24,
          "Benchmark CSV must have 24 rows")

    combinations = Counter(
        (int(row["k"]), int(row["threads"]))
        for row in rows
    )

    for k in K_VALUES:
        for p in THREADS:
            check(
                combinations[(k, p)] == 3,
                f"Incorrect runs for k={k}, p={p}"
            )

    for row in rows:
        for method in ("serial", "atomic", "private"):
            try:
                value = float(row[method])
                check(
                    value > 0,
                    f"Invalid {method} execution time"
                )
            except (ValueError, KeyError):
                check(False, "Malformed benchmark CSV")

    print("Benchmark CSV: checked")
else:
    check(False, "Missing benchmark.csv")

# ----------------------------------------
# 2. Check all benchmark logs
# ----------------------------------------

checked_logs = 0

for k in K_VALUES:
    for p in THREADS:
        for run in RUNS:

            path = LOGS / f"k{k}_p{p}_r{run}.txt"

            if not path.is_file():
                check(False, f"Missing {path}")
                continue

            content = path.read_text()
            checked_logs += 1

            check(
                field(content, "Expected windows")
                    == str(EXPECTED[k]["windows"]),
                f"Wrong window count: {path}"
            )

            check(
                field(content, "Serial fingerprint")
                    == EXPECTED[k]["fingerprint"],
                f"Wrong fingerprint: {path}"
            )

            check(
                field(content, "Atomic mismatched bins")
                    == "0",
                f"Atomic mismatch: {path}"
            )

            check(
                field(content, "Private mismatched bins")
                    == "0",
                f"Private mismatch: {path}"
            )

            check(
                "Exact bin-by-bin correctness: PASS"
                    in content,
                f"Correctness failure: {path}"
            )

print(f"Benchmark logs checked: {checked_logs}/24")

# ----------------------------------------
# 3. Check false-sharing results
# ----------------------------------------

for p in THREADS:

    path = RESULTS / f"false_sharing_p{p}.txt"

    if not path.is_file():
        check(False, f"Missing {path}")
        continue

    content = path.read_text()

    check(
        "Correctness: PASS" in content,
        f"False-sharing correctness failure: {path}"
    )

    check(
        field(content, "Adjacent counters")
            is not None,
        f"Missing adjacent median: {path}"
    )

    check(
        field(content, "Padded counters")
            is not None,
        f"Missing padded median: {path}"
    )

print("False-sharing logs: checked")

# ----------------------------------------
# 4. Check all figures
# ----------------------------------------

for name in FIGURE_NAMES:
    path = FIGURES / name

    check(
        path.is_file() and path.stat().st_size > 0,
        f"Missing or empty figure: {path}"
    )

print("Figures checked: 6/6")

# ----------------------------------------
# 5. Check performance summary
# ----------------------------------------

summary_path = RESULTS / "performance_summary.csv"

if summary_path.is_file():

    with summary_path.open(newline="") as f:
        summary = list(csv.DictReader(f))

    check(
        len(summary) == 16,
        "Performance summary must have 16 rows"
    )

    actual = {
        (
            int(row["k"]),
            int(row["threads"]),
            row["method"]
        )
        for row in summary
    }

    expected = {
        (k, p, method)
        for k in K_VALUES
        for p in THREADS
        for method in ("atomic", "private")
    }

    check(
        actual == expected,
        "Performance summary has missing configurations"
    )

    print("Performance summary: checked")

else:
    check(False, "Missing performance_summary.csv")

# ----------------------------------------
# 6. Final result
# ----------------------------------------

print("\n===== FINAL AUDIT =====")

if errors:
    print(f"FAIL: {len(errors)} problem(s) found")
    raise SystemExit(1)

print("Benchmark data: PASS")
print("Numerical verification: PASS")
print("False-sharing results: PASS")
print("Figures: PASS")
print("Performance summary: PASS")
print("\nCS2 EXPERIMENTAL RESULTS AUDIT: PASS")
