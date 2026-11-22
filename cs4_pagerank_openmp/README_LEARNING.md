# CS4 — Parallel PageRank: Learning Experiments
## 1. Overview
This directory contains educational experiments for

serial and OpenMP PageRank on directed graphs.

The experiments investigate:

- Compressed Sparse Row (CSR) graph representation

- Serial PageRank and convergence

- Dangling-node handling

- Parallel push and pull PageRank

- Atomic updates and contention

- OpenMP static, dynamic and guided scheduling

- Load imbalance

- Strong scaling and parallel efficiency

- Numerical correctness and reproducibility

These programs are learning materials, not an

independently authored assignment submission.

## 2. Requirements
- Linux or WSL

- GCC with OpenMP support

- GNU Make

- Python 3

- Matplotlib

C programs are compiled using:

    -O2 -Wall -Wextra -std=c11

OpenMP programs additionally use:

    -fopenmp

## 3. Build and Run
Compile all learning programs:

    make

Run correctness smoke tests:

    make check

Verify saved experimental results:

    make verify-results

Regenerate performance figures:

    make plots

Remove compiled executables:

    make clean

Regenerate the synthetic graph if required:

    make graph

The clean command preserves source files,

graph data, benchmark logs and figures.

## 4. PageRank Algorithm
PageRank estimates the relative importance

of vertices in a directed graph.

A vertex receives importance from the

vertices that link to it.

For a destination vertex v:

    PR_new(v) =

        (1-d)/N

        + d * dangling_mass/N

        + d * sum(PR_old(u)/outdegree(u))

The summation covers incoming neighbours u.

Parameters used:

    Damping factor: 0.85

    Convergence tolerance: 1e-10

    Maximum iterations: 1000

All vertices start with equal rank:

    PR_initial(v) = 1/N

## 5. Convergence
The algorithm repeatedly calculates updated

rank values using the previous iteration's ranks.

After each iteration, the L1 difference is:

    L1 = sum(abs(PR_new(v) - PR_old(v)))

Convergence occurs when:

    L1 < 1e-10

Two rank arrays are maintained to avoid

mixing old and new values.

Their pointers are swapped after each

iteration to avoid unnecessary copying.

The final rank vector should also satisfy:

    sum(PR(v)) approximately equals 1

## 6. Dangling Nodes
A dangling vertex has no outgoing edges.

Such vertices still have PageRank and may

receive incoming edges.

Without special handling, their probability

mass would disappear during an iteration.

The implementation collects the rank of

all dangling vertices and redistributes

that mass uniformly across the graph.

## 7. CSR Graph Representation
Compressed Sparse Row stores graph

connectivity without a dense adjacency matrix.

Outgoing CSR uses:

    out_row[N+1]

    out_col[M]

Outgoing neighbours of source u occupy:

    out_col[out_row[u] : out_row[u+1]]

Incoming CSR uses:

    in_row[N+1]

    in_src[M]

Incoming neighbours of destination v occupy:

    in_src[in_row[v] : in_row[v+1]]

Outgoing CSR supports push-style traversal.

Incoming CSR supports pull-style traversal.

Both representations store O(N+M) elements.

## 8. Synthetic Graph
The graph is generated using:

    python3 generate_graph_learning.py

Output file:

    graphs_cs4/synthetic_edges.txt

Graph parameters:

    Vertices: 100000

    Directed edges: 1260000

    Random seed: 20261121

Degree distribution:

    Dangling vertices: 1000

    High-outdegree vertices: 9000

    Regular vertices: 90000

Outdegrees:

    Dangling: 0

    High-outdegree: 80

    Regular: 6

Measured graph statistics:

    Maximum outdegree: 80

    Maximum indegree: 1358

The generator deliberately targets 200

popular destinations with 252000 edges.

This creates uneven incoming-degree workloads.

The graph may contain repeated edges and

self-loops, so it is treated as a directed

multigraph.

## 9. Serial PageRank Reference
The large serial PageRank experiment used

the full graph containing 1.26 million edges.

Verified results:

    Iterations: 19

    Final L1 residual: 8.432802547252e-11

    Sum of ranks: 1.000000000000

    Correctness: PASS

Selected ranks:

    Node 0:     0.000726907359

    Node 1:     0.000728066569

    Node 1000:  0.000011138316

    Node 99999: 0.000006847203

Reference output:

    results_cs4/serial_reference.txt

Graph loading time is excluded from

the measured PageRank iteration time.

## 10. Parallel Push
Parallel push distributes the rank of

each source vertex to its destinations.

Multiple threads may update the same

destination simultaneously.

Therefore, the parallel push experiment

uses:

    #pragma omp atomic update

Atomic updates prevent lost contributions

caused by concurrent read-modify-write

operations.

However, they introduce synchronization

overhead and potential contention.

The synthetic graph intentionally contains

popular destination vertices that can

amplify atomic contention.

## 11. Parallel Pull
Parallel pull assigns destination vertices

to OpenMP workers.

Each worker gathers contributions from

incoming neighbours and calculates a

destination's complete new rank.

Every destination is written by only

one parallel loop iteration.

Therefore, the destination updates

do not require atomics.

The implementation still needs appropriate

synchronization for other shared operations,

including reductions and iteration boundaries.

## 12. Numerical Correctness
The parallel algorithms were compared

against serial PageRank references.

The full 100000-element rank vectors

were compared, not just selected nodes.

Verification checks include:

- Convergence within the specified tolerance

- Conservation of total PageRank mass

- Maximum absolute rank difference

- Full-vector L1 difference

- Successful program completion

The parallel implementations converged

in 19 iterations on the test graph.

Atomic-update ordering and floating-point

addition can introduce tiny differences.

Exact bit-for-bit equality is not required.

## 13. OpenMP Scheduling

The fixed-chunk settings described in this section apply to the **original synthetic-graph experiments**. The later preferential-attachment chunk-size study is described in Section 23.
Three scheduling policies were investigated.

STATIC:

Work is assigned in predetermined blocks.

Advantages:

- Low scheduling overhead

- Simple and predictable assignments

Disadvantage:

- Can perform poorly for uneven vertex degrees

DYNAMIC:

Workers request additional chunks as

they finish previous chunks.

Advantages:

- Better adaptation to uneven workloads

- Can reduce idle time

Disadvantage:

- Additional scheduling overhead

GUIDED:

Workers initially receive larger chunks,

which become smaller as the loop progresses.

Advantages:

- Potentially lower scheduling overhead

  than fine-grained dynamic scheduling

Disadvantage:

- Early large chunks may still be imbalanced

In these experiments:

    Dynamic minimum chunk size: 64

    Guided minimum chunk size: 64

Static scheduling uses approximately one

contiguous vertex block per worker.

## 14. Load Imbalance

The edge counts in this section come from the **original synthetic graph**. The later per-thread timing experiment on the preferential-attachment graph is documented in Section 24.
The synthetic graph has uneven indegrees.

For static scheduling with four threads,

the first iteration assigned approximately:

    Thread 0: 503828 incoming edges

    Thread 1: 252218 incoming edges

    Thread 2: 251529 incoming edges

    Thread 3: 252425 incoming edges

The maximum/minimum ratio was:

    2.003x

For dynamic scheduling with four threads,

the measured ratio was:

    1.342x

This indicates more balanced incoming-edge

assignments in that measured iteration.

The edge counts are a workload proxy,

not a direct measurement of thread time.

Some eight-thread experiments reported

a worker receiving no edges in the first

iteration. That does not establish that

the worker remained idle in later iterations.

## 15. Scheduling Benchmark
Configurations:

    Scheduling: static, dynamic, guided

    Threads: 1, 2, 4, 6, 8

    Repetitions: 3

    Total runs: 45

The benchmark uses median execution times.

Results were saved to:

    results_cs4/scheduling_benchmark_raw.csv

    results_cs4/scheduling_benchmark_summary.csv

Individual logs:

    results_cs4/scheduling_benchmark_logs/

Measured serial pull baseline:

    0.073743 seconds

Median parallel pull runtimes:

| Threads | Static (s) | Dynamic (s) | Guided (s) |

| ------- | ---------: | ----------: | ---------: |

| 1       |   0.077175 |    0.083927 |   0.077207 |

| 2       |   0.057063 |    0.051816 |   0.053341 |

| 4       |   0.034643 |    0.031420 |   0.045185 |

| 6       |   0.039982 |    0.029577 |   0.038616 |

| 8       |   0.039482 |    0.024872 |   0.036608 |

Fastest measured scheduling configuration:

    Dynamic, 8 threads

Median runtime:

    0.024872 seconds

Speedup versus serial pull:

    2.965x

Dynamic scheduling performed best at

every tested thread count greater than one

in this scheduling benchmark.

## 16. Atomic Push vs Dynamic Pull
The algorithm comparison benchmark used:

    Algorithms: atomic push, dynamic pull

    Threads: 1, 2, 4, 6, 8

    Repetitions: 3

    Total runs: 30

Raw measurements:

    results_cs4/algorithm_comparison_raw.csv

Summary:

    results_cs4/algorithm_comparison_summary.csv

Individual logs:

    results_cs4/algorithm_comparison_logs/

The common serial push baseline was:

    0.044405 seconds

Measured median runtimes:

| Threads | Atomic Push (s) | Dynamic Pull (s) |

| ------- | --------------: | ---------------: |

| 1       |        0.154235 |         0.075208 |

| 2       |        0.147112 |         0.055243 |

| 4       |        0.130148 |         0.035170 |

| 6       |        0.117556 |         0.029541 |

| 8       |        0.119521 |         0.030650 |

Dynamic pull was faster than atomic

push at every tested thread count.

Greatest measured relative advantage:

    3.979x at 6 threads

Fastest configuration in this

separate algorithm comparison:

    Dynamic pull, 6 threads

Median runtime:

    0.029541 seconds

Speedup versus serial push:

    1.503x

The two benchmark campaigns selected

different fastest thread counts because

their timing samples differed.

## 17. Understanding the Baselines
Two baselines appear in the experiments.

SERIAL PULL:

Used for the scheduling comparison.

SERIAL PUSH:

Used as the common baseline in the

atomic-push versus dynamic-pull comparison.

Speedup is calculated as:

    speedup = serial_time / parallel_time

Parallel efficiency is:

    efficiency = speedup / threads * 100

The baseline must always be clearly named.

A speedup measured against serial pull

must not be presented as a speedup

against serial push.

## 18. Performance Figures

Figures 01–05 come from the original synthetic-graph studies; Figures 06–07 are from the separate preferential-attachment experiments in Sections 22–24.
The figures_cs4 directory contains:

    01_scheduling_runtime.png

    02_scheduling_speedup.png

    03_push_vs_pull_runtime.png

    04_algorithm_speedup.png

    05_pull_advantage.png

    06_indegree_distribution_loglog.png

    07_thread_timing_imbalance.png

Regenerate figures using:

    make plots

The figures are based on the saved

benchmark summary CSV files.

## 19. Automated Verification
Run:

    make check

    make verify-results

The correctness tests cover:

- Small serial PageRank

- Iterative convergence

- Small parallel push and pull

- Atomic updates

- Large CSR loading

- Large serial PageRank

- Large parallel pull

- Large parallel atomic push

- Scheduling variants

The results audit verifies:

    45 scheduling benchmark logs

    30 algorithm comparison logs

    Benchmark medians and speedups

    Saved serial reference

    Five PNG performance figures

The saved results audit completed with:

    CS4 EXPERIMENTAL RESULTS AUDIT: PASS

## 20. Limitations
- The graph is synthetic and may not represent

  the structure of real web graphs.

- The graph generator permits duplicate edges

  and self-loops.

- Runtime depends on processor frequency,

  memory caching, scheduling and system load.

- The PageRank solver runs quickly, making

  short-duration measurements sensitive to noise.

- Scheduling and algorithm benchmarks were

  conducted in separate runs.

- Atomics can become expensive on highly

  connected destination vertices.

- Pull performance depends on indegree

  distribution and memory-access patterns.

- The scheduling experiment deliberately

  keeps some auxiliary operations serial.

- The scheduling worker counts describe

  incoming-edge assignments, not measured

  execution time per worker.

- Numerical verification was performed on

  this graph and selected test configurations.

- Saved-log auditing cannot independently

  prove correctness for arbitrary input graphs.

- The benchmark results do not prove that one

  scheduling policy is universally optimal.

## 21. Learning Workspace Notice
This directory contains educational reference

implementations and experimental results.

The assessed assignment code, report and

supporting explanations must be written

independently according to the course's

academic-integrity requirements.

---

## 22. Preferential-Attachment Graph (Additional Experiment)

This is a **separate graph and experiment** from the synthetic grouped-outdegree graph discussed in Sections 8–19. It was generated using `generate_preferential_graph_learning.py` and saved as `graphs_cs4/preferential_attachment_edges.txt`.

| Property | Verified value |
|---|---:|
| Vertices | 100,000 |
| Directed edges | 1,299,909 |
| Maximum indegree | 1,920 |
| Maximum outdegree | 13 |
| Vertices with zero indegree | 13,340 |
| Vertices with zero outdegree | 1 |
| Incoming edges assigned to top 1% of vertices | 19.62% |

The concentration of incoming edges produces an irregular **pull** workload. The degree-distribution visualization is `figures_cs4/06_indegree_distribution_loglog.png`, supported by `results_cs4/indegree_distribution.csv`. A log–log plot helps illustrate the long-tailed indegree distribution, but this finite synthetic graph alone is not proof of an exact mathematical power law.

For contiguous static assignment with four workers (25,000 vertices each), the first iteration assigned 974,875, 213,527, 88,109, and 23,398 incoming edges to threads 0–3, respectively. This is a **41.665× maximum/minimum edge-count ratio**. Edge counts are a workload proxy, not elapsed thread times.

## 23. Chunk-Size Scheduling Study on the Preferential Graph

The program `pagerank_scheduling_learning` now accepts **four user arguments**: `GRAPH THREADS SCHEDULE CHUNK`. For example:

```bash
./pagerank_scheduling_learning graphs_cs4/preferential_attachment_edges.txt 4 dynamic 2048
```

The experiment used four OpenMP threads, schedules `static`, `dynamic`, and `guided`, chunk sizes 1, 8, 32, 128, 512, and 2048, with three repetitions per configuration (**54 verified runs**). All runs passed the PageRank correctness test. The benchmark and results are:

- `benchmark_preferential_chunks_learning.py`
- `results_cs4/preferential_chunks/chunk_raw.csv`
- `results_cs4/preferential_chunks/chunk_summary.csv`
- `results_cs4/preferential_chunks/logs/`

| Schedule | Best tested chunk | Median OpenMP pull time |
|---|---:|---:|
| Static | 128 | 0.073364 s |
| Dynamic | 2048 | **0.058788 s** |
| Guided | 2048 | 0.145254 s |

The recorded median of the per-run speedup values for dynamic/2048 was **2.620×** relative to each run's serial-pull time. These speedups are **medians of individual ratios**, not necessarily the ratio of a fixed median serial baseline to the median parallel runtime. For a unified final scaling comparison, recompute speedup from consistently chosen serial and parallel baseline medians.

The smallest dynamic chunk (1) took 0.208338 s at the median, illustrating the cost of fine-grained work assignment. Guided scheduling was substantially slower on this graph ordering; the per-thread timing data below helps explain the imbalance. Results do not establish which schedule is universally best on other graphs or machines.

## 24. Actual Per-Thread Timing and Load Imbalance

A separate instrumented program, `pagerank_thread_timing_learning.c`, measures time around the destination-vertex pull loop. Its OpenMP work-sharing loop uses `schedule(runtime) nowait`, enabling each thread's time to be recorded before the loop-end barrier; synchronization still occurs when the enclosing parallel region ends. Times include execution and scheduling costs within the timed loop and are not direct CPU-cycle measurements.

The requested timing imbalance ratio is:

\[
R = \frac{\max_t T_t}{\frac{1}{P}\sum_{t=0}^{P-1} T_t}
\]

where `T_t` is either one iteration's measured loop time for thread `t` or its cumulative loop time across all iterations, and `P` is the number of threads. Four configurations were tested three times each (**12 verified executions**). These results are distinct from the 54-run chunk benchmark.

| Schedule | Chunk | Median first-iteration R | Median cumulative R | Median solver time |
|---|---:|---:|---:|---:|
| Static, contiguous | 25,000 | 2.361× | 2.617× | 0.158077 s |
| Static, cyclic blocks | 128 | 1.001× | 1.103× | 0.083355 s |
| Dynamic | 2048 | 1.004× | **1.008×** | **0.064999 s** |
| Guided | 2048 | 2.036× | 1.238× | 0.139354 s |

The dynamic/2048 configuration had the most balanced cumulative measured thread times; contiguous static had the greatest imbalance. The guided first-iteration imbalance is consistent with large early assignments capturing many high-indegree vertices. The experiment does not isolate all sources of runtime overhead.

Results and visualization:

- `benchmark_thread_imbalance_learning.py`
- `results_cs4/thread_imbalance/thread_imbalance_raw.csv`
- `results_cs4/thread_imbalance/thread_imbalance_summary.csv`
- `figures_cs4/07_thread_timing_imbalance.png`

## 25. Atomic Push vs. Dynamic Pull on the Preferential Graph

A separate, **single-run** four-thread comparison on the preferential-attachment graph completed 69 iterations for both algorithms and passed numerical verification:

| Implementation | Iteration time | Reference |
|---|---:|---|
| Serial push | 0.065200 s | Serial push code path |
| Four-thread atomic push | 0.199441 s | Uses `#pragma omp atomic update` |
| Serial pull | 0.158818 s | Serial pull code path |
| Four-thread dynamic pull, chunk 2048 | 0.072956 s | Destination-owned updates |

In this run, four-thread dynamic pull was approximately **2.73× faster** than four-thread atomic push. However, this is an **algorithm-level difference**, not a direct measurement of atomic-instruction overhead: implementations also differ in access patterns, data structures, and synchronization. In this particular run, serial push was faster than both four-thread versions.

These single-run results must not be combined as though they were medians from the earlier synthetic-graph comparison. Logs: `results_cs4/preferential_atomic_push_p4.txt` and `results_cs4/preferential_dynamic_pull_p4.txt`.

## 26. Edge Throughput and Memory-Access Estimates

The preferential graph has 1,299,909 edges and converged in 69 iterations, giving **89,693,721 logical edge visits**. Dividing by each measured solver time gives the following *single-run* edge-processing rates:

| Algorithm | Edge throughput |
|---|---:|
| Serial push | 1,375.67 million edges/s |
| Four-thread atomic push | 449.73 million edges/s |
| Serial pull | 564.76 million edges/s |
| Four-thread dynamic pull | 1,229.42 million edges/s |

A simplified pull read-traffic model counts an incoming source ID (4 B), a source PageRank value (8 B), and two outdegree offsets (16 B), or **28 nominal logical read bytes per edge**. At 89,693,721 visits this represents approximately **2.511 GB of logical reads**; for the measured 0.072956-second dynamic pull run, the implied **nominal logical access rate** is **34.424 GB/s**.

This rate is **not measured physical DRAM bandwidth**. CPU caches may serve repeated rank and row-offset accesses, while the simple model omits other traffic, cache lines, and vertex-level operations.

A separate STREAM-like experiment used three arrays with a combined 576 MB working set. Its estimated bandwidth was **30.886 GB/s with one thread** and **29.690 GB/s with four threads**, based on the benchmark's logical read/write accounting. Because the memory-access patterns and measurement methods differ, comparing these values does **not** prove that PageRank saturated memory bandwidth.

Relevant analysis: `analyze_pagerank_memory_learning.py`, `results_cs4/pagerank_memory_analysis.csv`, and `memory_bandwidth_learning.c`.

## 27. Reproducing the Additional Educational Experiments

These commands assume the CS4 working directory, compiled programs, the generated graph, and the necessary Python libraries. They are **additional educational experiments** rather than replacements for the original synthetic-graph results or the existing `make check` and `make verify-results` targets.

```bash
# Build the scheduling and timing programs if necessary.
gcc -O2 -Wall -Wextra -std=c11 -fopenmp \
    pagerank_scheduling_learning.c -lm -o pagerank_scheduling_learning

gcc -O2 -Wall -Wextra -std=c11 -fopenmp \
    pagerank_thread_timing_learning.c -lm -o pagerank_thread_timing_learning

# Recreate the preferential-attachment graph and its degree plot.
python3 generate_preferential_graph_learning.py
python3 plot_degree_distribution_learning.py

# Full chunk-size campaign (54 runs).
python3 benchmark_preferential_chunks_learning.py

# Per-thread timing campaign (12 runs).
python3 benchmark_thread_imbalance_learning.py

# Logical edge-throughput and byte-accounting analysis
# (uses existing verified push/pull logs).
python3 analyze_pagerank_memory_learning.py

# Existing quick correctness checks.
make check
```

The previous five performance figures concern the older synthetic-graph benchmark; the new degree-distribution and thread-time imbalance figures are numbered **06** and **07**. Graph ordering, hardware, scheduling, short runtimes, and operating-system noise can affect the measurements. The programs, notes, and added analysis remain **educational references only**; independently authored assessed work is required by the course instructions.

