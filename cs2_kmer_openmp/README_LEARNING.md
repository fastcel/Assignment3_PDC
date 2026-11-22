# CS2 — OpenMP k-mer Counting: Learning Experiments

## Overview

This directory contains experiments exploring serial and
parallel k-mer counting for synthetic DNA sequences.

The experiments investigate:

- Rolling 2-bit k-mer encoding
- OpenMP data races
- Atomic histogram updates
- Per-thread private histograms
- Correct handling of thread chunk boundaries
- Strong scaling and throughput
- False sharing and cache-line padding

These are educational experiments, not the independently
authored final submission.

## Requirements

- GCC with OpenMP support
- GNU Make
- Python 3
- Matplotlib
- Linux / WSL environment

## Build

Compile all programs:

    make

Run quick correctness tests:

    make check

Audit saved benchmark results:

    make verify-results

Regenerate performance plots:

    make plots

Remove compiled programs:

    make clean

The clean target preserves source code, results, and figures.

## Programs

| Program                | Purpose                                          |
| ---------------------- | ------------------------------------------------ |
| kmer_learning_demo     | Small DNA rolling-encoding demonstration         |
| serial_kmer_practice   | Serial counting for configurable L and k         |
| omp_race_demo          | Simplified shared-counter race demonstration     |
| omp_chunk_demo         | Correct allocation of k-mer start positions      |
| omp_histogram_demo     | Atomic versus private histogram correctness      |
| omp_fullsize_learning  | Full-scale serial, atomic and private comparison |
| false_sharing_learning | Adjacent versus cache-line-separated counters    |
| omp_kmer_race_learning | Actual shared-histogram race demonstration       |

## Input Generation

The full-scale experiments generate 100,000,000 DNA bases.

DNA bases are represented as:

- A = 0
- C = 1
- G = 2
- T = 3

A fixed initial pseudorandom generator state is used for
repeatable sequences.

The supported full-scale k values are 8 and 12.

## k-mer Encoding

Each DNA base occupies two bits.

The rolling representation shifts the existing code by
two bits, appends the new base, and applies a mask to
retain the most recent k bases.

The expected number of k-mers is:

    L - k + 1

For L = 100,000,000:

- k = 8: 99,999,993 windows
- k = 12: 99,999,989 windows

## Correctness Verification

The atomic and private OpenMP histograms are compared
against the serial reference bin by bin.

The full-scale test checks:

- Total k-mer counts
- Exact histogram-bin equality
- Deterministic serial histogram fingerprint

Reference fingerprints:

| k   | Fingerprint      |
| --- | ---------------- |
| 8   | 605ab0a06ea0f33e |
| 12  | fdafb38ee6cbd6e0 |

## OpenMP Approaches

### Unsynchronized Shared Histogram

Multiple threads update the same histogram without
protection. This intentionally introduces a data race.

This method is incorrect and is used only to demonstrate
the problem.

### Atomic Shared Histogram

Threads share one histogram. Atomic updates prevent
concurrent increments from overwriting one another.

The method avoids replicated histograms but introduces
synchronization overhead.

### Private Histograms

Each thread updates its own histogram. The histograms are
merged after parallel counting completes.

The method avoids atomic updates during counting but
requires extra memory and merging work.

## Private Histogram Memory

Each histogram contains 4^k unsigned 64-bit counters.

| Threads | k=8     | k=12    |
| ------- | ------- | ------- |
| 1       | 512 KiB | 128 MiB |
| 2       | 1 MiB   | 256 MiB |
| 4       | 2 MiB   | 512 MiB |
| 6       | 3 MiB   | 768 MiB |

These values describe private histogram storage only,
not the total memory used by the program.

## Benchmark Procedure

The benchmark evaluates:

- k = 8 and 12
- Threads = 1, 2, 4 and 6
- Three runs per configuration
- Median execution times

The benchmark script is:

    benchmark_cs2.sh

Existing measurements are saved in:

    results_cs2/benchmark.csv

Individual logs are stored under:

    results_cs2/logs/

The reported speedups use the median serial baseline.

Counting execution times exclude DNA generation,
histogram allocation and correctness checking.

The private-histogram method includes its merge stage
in the measured time.

## Measured Serial Baselines

- k=8: 0.164805 seconds
- k=12: 1.363016 seconds

## Six-Thread Results

| k   | Best method | Time       | Speedup |
| --- | ----------- | ---------- | ------- |
| 8   | Private     | 0.051265 s | 3.215x  |
| 12  | Atomic      | 0.521374 s | 2.614x  |

For k=8, the small private histograms make avoiding
atomic synchronization beneficial.

For k=12, much larger private histograms increase the
memory footprint and merge workload.

These are measured associations, not definitive proof
of memory-bandwidth saturation.

## False Sharing

The false-sharing experiment compares:

- Adjacent per-thread 64-bit counters
- Per-thread counters spaced 64 bytes apart

Each thread performs 20 million increments.

Measured median adjacent/padded ratios:

| Threads | Ratio  |
| ------- | ------ |
| 1       | 0.920x |
| 2       | 4.713x |
| 4       | 9.393x |
| 6       | 9.907x |

All counter values were verified.

The adjacent layout suffers from cache-line contention
when multiple cores update independent counters in the
same line.

Padding reduces this effect in the controlled experiment.

## Performance Figures

The directory figures_cs2 contains:

1. 01_runtime_k8.png
2. 02_speedup_k8.png
3. 03_runtime_k12.png
4. 04_speedup_k12.png
5. 05_false_sharing.png
6. 06_throughput.png

The script plot_cs2.py regenerates these figures.

## Limitations

- The timing measurements can vary due to CPU scheduling,
  cache state and competing system activity.
- The private-histogram merge is sequential in the
  learning implementation.
- Atomic contention depends on the distribution of
  k-mer frequencies.
- The false-sharing test is a focused microbenchmark.
- Hardware performance counters were not collected,
  so memory-bound behavior is inferred rather than
  conclusively demonstrated.
- The intentionally unsafe programs must not be treated
  as correct implementations.

## Result Verification

Run:

    make check
    make verify-results

Both commands should finish successfully before using
the saved measurements for further analysis.
