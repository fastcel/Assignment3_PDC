# CS5 — Distributed Sample Sort with MPI
## Educational Experiments — Not for Assessed Submission

This folder contains AI-assisted learning experiments used to understand distributed sample sort, MPI communication, correctness verification, performance scaling, and load imbalance.

The original assignment prohibits submitting AI-generated or AI-adapted code and explanations. These examples are study materials, not independently authored submission work.

**Experiment distinction:** Sections 2–14 describe the original **64-bit packed-key** implementation and its measurements unless specifically labeled otherwise. Section 15 describes the later, separately benchmarked **native 32-bit integer** implementation. Measurements and communication models from one representation must not be attributed to the other.

## 1. Environment

- Operating system: Ubuntu 24.04 under WSL
- C compiler: GCC 13.3.0
- MPI implementation: Open MPI 4.1.6
- Reported logical processors: 12
- Physical cores: 6
- Compile flags: `-O2 -Wall -Wextra -std=c11`
- Full-scale dataset: 100,000,000 integers
- Input distributions: uniform and duplicate-heavy

For eight MPI ranks, the experiments use:

```bash
mpirun --use-hwthread-cpus -np 8
```

The environment is one shared-memory machine, not a multi-node cluster.

## 2. Algorithm Studied

Each MPI process:

1. Generates its own subset of the input.
2. Sorts its local data with `qsort`.
3. Selects regular samples.
4. Participates in `MPI_Gather` for sample collection.
5. Receives global splitters with `MPI_Bcast`.
6. Partitions sorted data into destination buckets.
7. Exchanges bucket sizes using `MPI_Alltoall`.
8. Exchanges values using `MPI_Alltoallv`.
9. Sorts the received partition.
10. Participates in distributed correctness checks.

No process gathers the complete dataset during the full-scale sorting experiments.

## 3. Duplicate-Key Handling — Original Packed-Key Experiment

The original scalable learning implementation uses **64-bit packed keys**:

- Upper 32 bits = original integer value.
- Lower 32 bits = unique global index.

This preserves numerical sorting order while giving equal-valued integers deterministic tie-breaking IDs.

The packed representation uses **eight bytes per key** rather than four bytes per original integer. This is an important distinction from a plain 32-bit integer sample-sort implementation. **Section 15 documents the separate native-32-bit experiment**, which does not transport these packed keys.

## 4. Reproducible Input

The shared generator in `sample_input_learning.h` derives each key from its absolute global index. Therefore, input values remain the same when the number of MPI processes changes.

The serial and MPI reference programs produced identical global fingerprints for one million integers with 1, 2 and 4 MPI processes. The full-scale scaling benchmark also verified identical fingerprints across all tested configurations.

The native-32-bit experiment independently checked fingerprints against its corresponding 32-bit serial reference; see Section 15.

## 5. Correctness Verification

Checks include:

- Sorted order within each final partition.
- Correct ordering between adjacent nonempty partitions.
- Preservation of the total element count.
- Preservation of aggregate fingerprint sum.
- Preservation of aggregate fingerprint XOR.
- Identical expected fingerprints across MPI process counts.

Aggregate fingerprints provide strong error detection but do not mathematically prove multiset equality. Tiny examples additionally compare against a fully sorted serial reference.

Run the original automated tests:

```bash
make all
make check
```

The original check passed for uniform and duplicate-heavy inputs at MPI process counts 1, 2, 4 and 8. Test logs are stored in `results_cs5/smoke_logs/`.

Run the additional native-32-bit checks with `make check-uint32` (Section 15).

## 6. Strong-Scaling Experiment — Original Packed-Key Representation

- Input size: 100,000,000 integers
- MPI process counts: 1, 2, 4, 6, 8
- Repetitions: 3 per configuration
- Statistic: median execution time
- Oversampling factor: 4

| Distribution | Serial baseline (s) | MPI p=8 median (s) | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| Uniform | 23.235360 | 4.842034 | 4.799× | 59.98% |
| Duplicates | 13.623720 | 3.785902 | 3.599× | 44.98% |

The serial baseline is a separate optimized C `qsort` program. The MPI p=1 run is **not** the serial baseline: it still performs MPI operations and a second local sorting phase.

The original saved benchmark contains some reused logs from before the `--use-hwthread-cpus` launcher change. A controlled comparison of that original representation would rerun every configuration consistently. This qualification concerns the **original packed-key** results and should not be silently applied to, or removed from, the separate native-32-bit results.

Benchmark files:

- `results_cs5/scaling_raw.csv`
- `results_cs5/scaling_summary.csv`
- `results_cs5/scaling_logs/`

## 7. Load Imbalance — Original Packed-Key Representation

Metric:

\[
R = \frac{\text{maximum final partition count}}{\text{mean final partition count}}
\]

At 100 million integers, the measured median imbalance was:

| Distribution | p=4 | p=8 |
|---|---:|---:|
| Uniform | 1.230× | 1.104× |
| Duplicates | 1.054× | 1.183× |

Balanced partition sizes are desirable because the slowest process can delay completion of the collective computation. Oversampling can improve splitter selection, but improvement is not guaranteed for every distribution.

**Do not reuse the original duplicate-heavy imbalance figures for the native-32-bit algorithm.** That implementation has different equal-key partitioning behavior.

## 8. MPI Ping-Pong Experiment

Two ranks exchanged messages of different sizes.

Measured estimates:

- Zero-byte one-way latency: 0.184 microseconds.
- Large-message beta: 0.232 nanoseconds per byte.
- Fitted throughput: 4.316 decimal GB/s.

The large-message regression has a different intercept from the zero-byte startup measurement. Consequently, one alpha-beta line does not perfectly describe the full range of message sizes.

Results: `results_cs5/pingpong_p2.csv`.

## 9. Actual Sample-Sort Traffic — Original Packed Keys

Application-level remote payload bytes:

| Distribution | MPI p=4 | MPI p=8 |
|---|---:|---:|
| Uniform | 600.08 MB | 700.03 MB |
| Duplicates | 198.47 MB | 287.26 MB |

These figures count **64-bit packed keys** sent between different ranks. They exclude rank-to-self transfers.

The traffic-inspection program calculates send counts without executing the full `MPI_Alltoallv` data exchange. Its byte counts are **not native-32-bit traffic measurements**.

Send-count matrices: `results_cs5/traffic_logs/`.

## 10. Communication Model — Original Packed-Key Experiment

The simplified model estimates:

\[
T = (p - 1)\alpha + \beta\,\text{critical\_remote\_bytes}
\]

Here `critical_remote_bytes` is the larger of the maximum per-rank remote send volume and maximum per-rank remote receive volume.

Comparison of modeled and observed collective-phase times:

| Distribution | MPI ranks | Predicted (s) | Observed (s) |
|---|---:|---:|---:|
| Uniform | 4 | 0.04277 | 0.34424 |
| Uniform | 8 | 0.02238 | 0.16107 |
| Duplicates | 4 | 0.01533 | 0.26851 |
| Duplicates | 8 | 0.01622 | 0.20117 |

The simple model substantially underpredicts the observed times because it omits contention, scheduling, synchronization, and MPI collective implementation details.

The measured `Alltoallv` phase may include time spent waiting for other ranks to enter the collective.

Communication results:

- `results_cs5/communication_model.csv`
- `results_cs5/traffic_adjusted_model.csv`

These models use the **original packed-key payload sizes**. To model native-32-bit traffic, the communication bytes and any representation-dependent measurements must be independently recomputed.

## 11. Timing Methodology

The core sorting interval excludes initial input generation. Elapsed times are measured with `MPI_Wtime`. The overall MPI interval is reported using the maximum across ranks.

In the original phase-summary method, **each phase is separately reported using its own maximum across ranks**. Those phase maxima may come from different ranks and should **not** be stacked as though they sum exactly to the total execution time.

Later representative-rank stacked plots use a different method. The original packed-key representative-rank plots (11–12) and native-32-bit representative-rank plots (17–18) each take phases from **one actual rank in an actual execution**, rather than combining unrelated phase maxima. These stacks describe that rank's work; they are **not** a globally additive decomposition of the distributed job's overall wall-clock time.

## 12. Generated Figures

`figures_cs5/` contains:

**Original packed-key scaling and communication figures:**

1. `01_scaling_runtime.png`
2. `02_scaling_speedup.png`
3. `03_scaling_efficiency.png`
4. `04_load_imbalance.png`
5. `05_uniform_phase_timings.png`
6. `06_duplicates_phase_timings.png`
7. `07_timing_variability.png`
8. `08_pingpong_model.png`
9. `09_alltoallv_model_comparison.png`
10. `10_traffic_adjusted_model.png`
11. `11_uniform_stacked_rank_phases.png`
12. `12_duplicates_stacked_rank_phases.png`

Figures 05–06 summarize separate phase maxima and must not be interpreted as additive stacked totals. Figures 11–12 use representative-rank phases for the original packed-key runs.

**Native-32-bit scaling and representative-rank phase figures:**

13. `13_uint32_runtime.png`
14. `14_uint32_speedup.png`
15. `15_uint32_efficiency.png`
16. `16_uint32_imbalance.png`
17. `17_uint32_uniform_stacked_phases.png`
18. `18_uint32_duplicates_stacked_phases.png`

Figures 13–18 relate to the native-32-bit experiment in Section 15, not the earlier packed-key benchmark.

## 13. Important Limitations

1. The **original packed-key** implementation transports 64-bit packed keys, not plain 32-bit integer payloads; the separate native-32-bit implementation addresses this representation difference.
2. The sample-sort experiments use `qsort` after redistribution instead of a specialized k-way merge.
3. The simple communication model is not a precise collective-performance predictor, and its original measurements concern the packed-key representation.
4. Some **original packed-key** scaling logs have mixed launcher histories and should not be presented as a single controlled placement campaign.
5. Independently maximized phase timings (original Figures 05–06) are not additive. Representative-rank stacks (Figures 11–12 and 17–18) fix that plotting issue for one rank, but do not prove a global additive phase breakdown.
6. These experiments run on one shared-memory machine, not a multi-node computer cluster. Performance therefore includes shared-resource and process-placement effects.
7. Numerical fingerprints and finite test suites provide useful verification but do not establish correctness for every possible input.
8. The material is AI-assisted educational reference work and must not be represented as independently authored coursework.

## 14. Useful Commands

Compile:

```bash
make all
```

Run original correctness tests:

```bash
make check
```

Run additional native-32-bit correctness tests:

```bash
make check-uint32
```

Regenerate original performance figures:

```bash
make plots
```

Regenerate communication figures:

```bash
make communication
```

Regenerate native-32-bit performance figures:

```bash
make plots-uint32
```

Remove executables while preserving results:

```bash
make clean
```

## 15. Additional Native 32-bit Sample-Sort Experiments

### 15.1 Why a second implementation was tested

The original packed-key sorter uses eight bytes per element to carry both a value and a global tie-breaking index. The later educational experiment tests sorting the **original 32-bit values directly**. This makes the element payload four bytes instead of eight bytes, avoiding the packed representation's twofold payload expansion.

The separate programs are:

- `sample_sort_uint32_learning.c`: MPI sample sort operating on `uint32_t` values.
- `sample_serial_uint32_learning.c`: corresponding serial reference.

MPI redistribution uses `MPI_UINT32_T`. The core full-scale input of **100,000,000 integers** corresponds to **400,000,000 bytes (400 decimal MB)** of original 32-bit values, before temporary buffers and auxiliary storage.

### 15.2 Duplicate-aware partitioning

Equal-valued integers are not artificially made unique with appended global IDs. Instead, the native-32-bit sorter has duplicate-aware boundary balancing: it can distribute equal keys across adjacent destination partitions while maintaining nondecreasing global output order.

The boundary correctness condition is nondecreasing order; equality across neighboring partitions is valid. Consequently, the inter-rank boundary check should reject a previous maximum **greater than** the next minimum, not an equal boundary value.

This treatment is important for the duplicate-heavy input, where repeatedly choosing the same splitter value can otherwise produce highly uneven partitions.

### 15.3 Correctness and experimental coverage

The native-32-bit correctness suite passed for **both uniform and duplicate-heavy inputs** at **1, 2, 4, 6, and 8 MPI processes** on the one-million-element test dataset: **10 tests passed**.

Additional full-scale checks used 100 million elements with four processes for both distributions; MPI and serial runs passed correctness tests and their value-based fingerprints matched.

The full 100-million-element scaling campaign included:

- Two input distributions: uniform and duplicate-heavy.
- Five MPI process counts: 1, 2, 4, 6, and 8.
- Three repetitions of each MPI configuration.
- Three serial reference executions per distribution.
- **36 total benchmark executions**, all passing their correctness and fingerprint checks.

The benchmark recordings and summaries are in:

- `results_cs5/uint32_scaling_raw.csv`
- `results_cs5/uint32_scaling_summary.csv`
- `results_cs5/uint32_scaling_logs/`

The dedicated small-input test output is produced by `make check-uint32`.

### 15.4 Native 32-bit strong-scaling results

The following values are **medians over three repetitions**, calculated separately for each input distribution and process count. Speedup uses the corresponding standalone **native-32-bit serial `qsort` baseline**, not MPI p=1 and not the original packed-key serial baseline.

**Uniform 100-million-element input** (native-32-bit serial median: **20.063809 s**):

| MPI processes | MPI median time (s) | Speedup vs serial | Efficiency | Median count imbalance |
|---:|---:|---:|---:|---:|
| 1 | 26.501683 | 0.757× | 75.71% | 1.000× |
| 2 | 14.305001 | 1.403× | 70.13% | 1.200× |
| 4 | 7.127198 | 2.815× | 70.38% | 1.230× |
| 6 | 4.784985 | 4.193× | 69.88% | 1.143× |
| 8 | 3.738318 | **5.367×** | **67.09%** | 1.104× |

**Duplicate-heavy 100-million-element input** (native-32-bit serial median: **11.293678 s**):

| MPI processes | MPI median time (s) | Speedup vs serial | Efficiency | Median count imbalance |
|---:|---:|---:|---:|---:|
| 1 | 17.329585 | 0.652× | 65.17% | 1.000× |
| 2 | 9.171255 | 1.231× | 61.57% | 1.000× |
| 4 | 4.693291 | 2.406× | 60.16% | 1.000× |
| 6 | 3.354395 | 3.367× | 56.11% | 1.000× |
| 8 | 2.485132 | **4.544×** | **56.81%** | 1.000× |

The native-32-bit implementation's equal-key balancing produced **1.000× final-partition count imbalance** for the duplicate-heavy distribution at all tested process counts. The uniform distribution exhibited some imbalance, including 1.230× at four processes and 1.104× at eight processes.

At MPI p=1, speedup relative to the standalone serial `qsort` remains below one because the MPI sample-sort implementation performs additional sorting and coordination steps. With more processes, its measured full-scale runtime decreases, but the efficiency stays below 100% because of communication, synchronization, algorithmic overheads, and shared hardware resources.

### 15.5 Additive representative-rank phase visualizations

The later native-32-bit plotting script, `plot_uint32_final_learning.py`, generates the six native-32-bit figures. The stacked phase figures are:

- `figures_cs5/17_uint32_uniform_stacked_phases.png`
- `figures_cs5/18_uint32_duplicates_stacked_phases.png`

For each process count and distribution, the plotting workflow selects an **actual representative execution with the median overall runtime** and obtains the phase timings from the execution's **bottleneck rank**. Because the components come from the same rank in that execution, they can be shown as a stack for that rank, unlike independently maximized phase times drawn from different ranks.

However, the sum of one rank's measured phases must **not** be labeled as an exact decomposition of the complete distributed execution's wall-clock time without further synchronization and accounting. The stacks describe the selected rank's recorded phases.

The representative phase data are saved in:

- `results_cs5/uint32_phase_representative.csv`

### 15.6 Reproduction and interpretation notes

From `cs5_samplesort_mpi/`:

```bash
make all
make check-uint32
make plots-uint32
```

The automated figure generator uses the already recorded native-32-bit benchmark results; rerunning plots does not itself rerun the complete 100-million-element scaling experiment.

For interpretation, keep these cases separate:

| Property | Original experiment | Later experiment |
|---|---|---|
| Element representation | Packed 64-bit value plus index | Native `uint32_t` value |
| Transport datatype | 64-bit integers | `MPI_UINT32_T` |
| Original-value storage for 100M elements | 800 decimal MB packed payload | 400 decimal MB native payload |
| Eight-process uniform speedup | 4.799× | 5.367× |
| Eight-process duplicate-heavy speedup | 3.599× | 4.544× |
| Traffic/model figures | Based on packed keys | Not independently remeasured in the original model |
| Scaling/imbalance figures | 01–04 | 13–16 |
| Representative-rank stacked figures | 11–12 | 17–18 |

These two campaigns use different representations and benchmark histories. Their reported timings should be labeled as **separate experiments** rather than presented as a controlled apples-to-apples payload-size experiment.

### 15.7 Limitations specific to the new data

- The native-32-bit results were obtained on a single WSL host using MPI processes on shared hardware; they are not multi-node network measurements.
- The strong-scaling results cover two synthetic input distributions and process counts up to eight. They do not establish performance for arbitrary key distributions or clusters.
- A final-partition count imbalance of 1.000× establishes equal *counts*, not equal CPU time or perfectly equal communication cost.
- Fingerprint agreement, sorting checks, and repeated runs support correctness, but do not constitute a formal proof.
- Original packed-key traffic measurements and the communication model cannot be directly assigned to the new native-32-bit transport without new representation-specific measurements.
- All programs, plots, and documentation here are **educational reference materials** subject to the assignment's independent-authorship requirements.
