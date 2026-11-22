# Case Study 1: Heat Diffusion with MPI

## Overview

This project investigates parallel heat diffusion on a
two-dimensional grid using MPI.

Five implementations are included:

- serial.c: Serial reference implementation
- mpi_1d_demo.c: 1D blocking MPI
- mpi_2d_demo.c: 2D blocking MPI
- mpi_1d_nonblocking_demo.c: 1D non-blocking MPI
- mpi_2d_nonblocking_demo.c: 2D non-blocking MPI

## Simulation Model

The simulation uses a five-point stencil:

new[i,j] = old[i,j] + alpha * (
    old[i-1,j] + old[i+1,j] +
    old[i,j-1] + old[i,j+1] -
    4 * old[i,j]
)

Parameters:
- alpha = 0.1
- Initial centre temperature = 100
- All other initial temperatures = 0
- Global boundary temperatures remain fixed at 0
- Double-buffered grids prevent in-place updates

## Requirements

- Linux or Ubuntu WSL
- GCC
- OpenMPI
- GNU Make
- Python 3
- Matplotlib for plotting

## Compilation

Build all five programs:

    make

Remove compiled executables:

    make clean

## Execution

Serial:

    ./serial 1024 500

1D blocking:

    mpirun -np 4 ./mpi_1d_demo 1024 500 50

2D blocking:

    mpirun -np 4 ./mpi_2d_demo 1024 500 50

1D non-blocking:

    mpirun -np 4 ./mpi_1d_nonblocking_demo 1024 500 50

2D non-blocking:

    mpirun -np 4 ./mpi_2d_nonblocking_demo 1024 500 50

MPI arguments:

    N T K

N: Grid dimension
T: Number of time steps
K: Global maximum-change checking interval

## Parallelization

1D decomposition:
- Divides the global grid into horizontal row blocks
- Exchanges top and bottom halo rows

2D decomposition:
- Uses a Cartesian process topology
- Divides the grid into rectangular blocks
- Exchanges halo rows and columns
- Uses MPI_Type_vector for column communication

Blocking versions use MPI_Sendrecv.

Non-blocking versions use MPI_Irecv and MPI_Isend,
compute independent interior cells, and then call
MPI_Waitall before updating halo-dependent cells.

MPI_Allreduce calculates the global maximum change.

## Numerical Correctness

Run:

    make check

The verification compares the complete temperature
grid against the serial reference.

Test configurations:
- N = 32 and 64
- T = 500
- K = 50
- MPI processes = 1, 2, 4, and 6
- Four MPI implementations

All 32 full-grid comparisons passed with zero
observed absolute error at printed precision.

Additional tests covered smaller grids.

For N = 4096, T = 500, all implementations matched
the serial checksum, sum of squares, and centre
temperature within the test tolerance. This does
not constitute full-grid verification at N = 4096.

## Performance Experiments

Experiments include:
- Strong scaling for N = 1024
- Weak scaling with approximately constant work per rank
- Blocking versus non-blocking communication
- 1D versus 2D decomposition
- K = 1 versus K = 50
- N = 4096 large-grid benchmark
- Serial row-padding experiment

Three execution-time measurements were collected
for each reported configuration, using their median.

Large-grid benchmark results:

    results_4096/timings.csv

Padding experiment results:

    results_padding/timings.csv

## Figures

The figures_cs1 directory contains:

01_strong_speedup.png
02_strong_efficiency.png
03_weak_runtime.png
04_weak_efficiency.png
05_blocking_vs_nonblocking.png
06_large_grid_comparison.png

Regenerate figures using:

    python3 plot_cs1.py

The N = 1024 figures use recorded benchmark medians.
The N = 4096 comparison reads the saved CSV file.

## Observations

At N = 1024, strong-scaling efficiency fell below
70 percent at four MPI processes in both blocking
decompositions.

Weak-scaling efficiency also decreased as the
process count increased.

The optimized 2D non-blocking version avoided
unnecessary full-block scans when updating
boundary cells. This substantially improved its
execution time.

At N = 4096, the original serial implementation
experienced an unusually large slowdown. Adding
one padding element per row improved performance,
consistent with a memory-layout effect.

Speedups relative to the serial solver include
implementation differences. One-process MPI
timings should also be considered when evaluating
parallel scaling within the same implementation.

## Test Environment

OS: Ubuntu 24.04 under WSL
Compiler: GCC 13.3.0
MPI: OpenMPI 4.1.6
Processor: AMD Ryzen 7 7445HS

## Additional 4096 Strong-Scaling Results

The large-grid strong-scaling experiment uses:
- N = 4096
- T = 500
- K = 50
- MPI process counts = 1, 2, 4, 6
- Three runs per configuration

The saved results are in:

    results_4096/strong_scaling.csv

Run the experiment using:

    ./benchmark_4096_scaling.sh

Regenerate the additional figures using:

    python3 plot_4096_scaling.py

Additional figures:

07_large_strong_runtime.png
08_large_strong_speedup.png
09_large_strong_efficiency.png

These figures use each MPI implementation's
one-process execution time as its speedup baseline.
