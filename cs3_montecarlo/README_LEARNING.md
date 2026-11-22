# CS3 — Monte Carlo Option Pricing: Learning Experiments

## Overview

This directory contains educational experiments for pricing
a European call option using serial and OpenMP Monte Carlo
simulation.

The experiments investigate:

- Geometric Brownian motion
- Monte Carlo option pricing
- Black-Scholes analytical verification
- Standard error and confidence intervals
- OpenMP parallelization and reduction
- Thread-local random-number generation
- Reproducibility
- Strong scaling and parallel efficiency
- Fixed-time statistical efficiency
- Problems caused by identical RNG streams

These experiments are learning material, not the
independently authored final assignment submission.

## Requirements

- GCC with OpenMP support
- GNU Make
- Python 3
- Matplotlib
- Linux or WSL

## Build and Run

Compile all programs:

    make

Run quick correctness tests:

    make check

Verify saved experimental results:

    make verify-results

Regenerate performance figures:

    make plots

Remove executables:

    make clean

The clean command preserves source files,
benchmark results and figures.

## Monte Carlo Pricing Model

The terminal stock price follows geometric Brownian motion:

    ST = S0 * exp((r - sigma^2/2)*T
                  + sigma*sqrt(T)*Z)

where Z follows a standard normal distribution.

A European call option has payoff:

    payoff = max(ST - K, 0)

The Monte Carlo estimator averages discounted payoffs:

    price = exp(-r*T) * mean(payoff)

## Option Parameters

| Parameter | Meaning               | Value  |
| --------- | --------------------- | ------ |
| S0        | Initial stock price   | 100    |
| K         | Strike price          | 100    |
| r         | Risk-free annual rate | 0.05   |
| sigma     | Annual volatility     | 0.20   |
| T         | Expiration time       | 1 year |

Black-Scholes reference price:

    10.45058357

## Random-Number Generation

The experiments use a deterministic pseudorandom generator
and the Box-Muller transformation to produce approximately
standard-normal random samples.

The main seed is fixed for reproducibility.

Parallel workers maintain separate RNG states derived
from the base seed and worker ID using SplitMix64 mixing.

Distinct initial states prevent exact stream duplication,
but this initialization does not formally prove statistical
independence of the streams.

Using one shared rand() generator is avoided.

## Programs

| Program                     | Purpose                               |
| --------------------------- | ------------------------------------- |
| mc_serial_learning          | Serial Monte Carlo estimator          |
| mc_rng_streams_learning     | RNG stream demonstration              |
| mc_reduction_learning       | OpenMP reduction demonstration        |
| mc_openmp_learning          | Small parallel Monte Carlo estimator  |
| mc_fullsize_learning        | Full-scale serial/OpenMP comparison   |
| mc_rng_correlation_learning | Identical-seed correlation experiment |

## Serial and Parallel Implementation

The serial program simulates paths sequentially.

The OpenMP programs distribute paths across workers.

Each worker independently calculates discounted payoffs
and accumulates the sum and sum of squared payoffs.

The main full-scale learning implementation stores
thread-local partial sums and combines them in a fixed
worker order.

The reduction demonstration separately shows the
OpenMP reduction clause.

## Statistical Accuracy

The sample mean estimates the option price.

The standard error is calculated using the sample
variance of discounted payoffs divided by the
number of simulated paths.

An approximate 95% confidence interval is:

    estimate +/- 1.96 * standard_error

For independent paths, standard error decreases
approximately in proportion to:

    1 / sqrt(M)

Therefore, four times as many independent paths
should approximately halve the standard error.

Individual absolute errors can still fluctuate.

## Strong-Scaling Benchmark

The full-scale benchmark used:

- 100,000,000 simulated paths
- 1, 2, 4, 6 and 8 OpenMP threads
- Three runs per configuration
- Median execution times
- A serial baseline compiled with -O2

Raw results:

    results_cs3/strong_scaling.csv

Individual logs:

    results_cs3/logs/

Benchmark script:

    benchmark_mc_learning.py

## Measured Strong-Scaling Results

Serial baseline:

    5.477526 seconds

| Threads | Median time (s) | Speedup | Efficiency |
| ------- | --------------- | ------- | ---------- |
| 1       | 5.490271        | 0.998x  | 99.77%     |
| 2       | 2.930164        | 1.869x  | 93.47%     |
| 4       | 1.492793        | 3.669x  | 91.73%     |
| 6       | 1.054865        | 5.193x  | 86.54%     |
| 8       | 0.818334        | 6.694x  | 83.67%     |

Speedup is measured against the serial baseline,
not against the parallel program using one thread.

The simulation scales well because paths are
computationally independent and require relatively
little synchronization.

Eight threads exceed the six physical cores of the
test machine, so simultaneous multithreading may
influence performance.

## Amdahl's Law

The theoretical speedup model is:

    S(p) = 1 / (f + (1-f)/p)

where f represents the nonparallel fraction.

The effective serial fraction can be estimated from
observed speedup.

The estimate also reflects runtime overhead,
thread scheduling, reduction and processor behavior;
it is not a direct measurement of inherently serial code.

## Near-Equal-Time Statistical Efficiency

The comparison used:

| Measurement    | 1 thread   | 8 threads  |
| -------------- | ---------- | ---------- |
| Paths          | 18,000,000 | 97,000,000 |
| Median time    | 0.497278 s | 0.529521 s |
| Standard error | 0.00346921 | 0.00149468 |

Observed standard-error improvement:

    2.321x

Expected improvement from path counts:

    sqrt(97/18) = approximately 2.321x

The median times differ by approximately 6.5%,
so these results represent a near-equal-time
comparison, not an exact equal-time comparison.

The eight-thread version processes more independent
paths within a similar computation budget, reducing
the statistical uncertainty of the estimate.

## Identical-Seed RNG Experiment

The RNG experiment intentionally initializes
all workers with identical random states.

Each worker simulates 200,000 paths.

| Threads | Nominal paths | Distinct paths | SE underestimation |
| ------- | ------------- | -------------- | ------------------ |
| 4       | 800,000       | 200,000        | 2.000x             |
| 8       | 1,600,000     | 200,000        | 2.828x             |

When workers repeat identical samples, the naive
standard-error calculation falsely assumes that
all samples are independent.

For p identical streams, the naive standard error
is underestimated by approximately sqrt(p).

The separate-state experiment avoids exact
duplication but does not independently certify
the RNG's statistical quality.

## Reproducibility

Repeated runs with the same worker count and
seed produced matching option prices and standard
errors to the program's printed precision.

Changing the number of workers changes the
assignment of random-number streams to simulated
paths.

Floating-point addition is also non-associative,
so a change in reduction order can change the
result slightly.

Fixed-order merging improves repeatability for
the same worker configuration.

## Performance Figures

The figures_cs3 directory contains:

1. 01_runtime.png
2. 02_speedup.png
3. 03_efficiency.png
4. 04_fixed_time_precision.png
5. 05_rng_correlation.png

Generate them using:

    make plots

## Saved Experimental Data

The results_cs3 directory contains:

- Strong-scaling CSV
- Individual strong-scaling logs
- Repeated fixed-time experiment logs
- RNG correlation logs
- Performance summary CSV

Results can be audited using:

    make verify-results

## Limitations

- Timing results depend on CPU frequency, thermal
  conditions, process scheduling and system load.
- The Box-Muller transformation and exponential
  functions make this workload computation-intensive.
- The separately seeded PRNG streams are not formally
  proven statistically independent.
- The serial and parallel implementations use different
  random streams, so their option prices need not match
  exactly.
- Statistical validation against Black-Scholes does not
  guarantee correctness for every parameter choice.
- Fixed-time runs were approximately, not exactly,
  equal in duration.
- Reproducibility was checked at printed precision.
- Amdahl serial-fraction estimates rely on an idealized
  scaling model.
- These learning programs are not independently
  authored submission material.

## Validation

Run:

    make
    make check
    make verify-results

All checks should pass before using these results
for further analysis.
