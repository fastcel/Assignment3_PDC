# PDC Assignment 3

**Course:** Parallel and Distributed Computing (CS3006)  
**Group:** 10

**Group Members:**
- Shehryar Hassan (23L-0603)
- Meerab Munir (23L-0971)

## Overview

In this assignment, we implemented five case studies using OpenMP and MPI to explore parallel computing. We compared serial and parallel implementations, verified correctness and measured performance through speedup, efficiency and scaling experiments.

## Case Studies

| Case Study | Problem | Model | Lead |
|---|---|---|---|
| CS1 | Heat Diffusion | MPI | Shehryar |
| CS2 | k-mer Counting | OpenMP | Meerab |
| CS3 | Monte Carlo Option Pricing | OpenMP | Meerab |
| CS4 | PageRank | OpenMP | Shehryar |
| CS5 | Distributed Sample Sort | MPI | Shehryar |

## Environment

We carried out our experiments on the following system:

- **Processor:** AMD Ryzen 7 7445HS (6 cores, 12 threads)
- **RAM:** 16 GB
- **Operating System:** Ubuntu 24.04 (WSL)
- **Languages:** C and Python
- **Tools:** GCC, OpenMP, Open MPI and Matplotlib

## Build and Run

Each case study has its own Makefile and README with the necessary instructions.

To compile and run the correctness tests for any case study:

```bash
cd cs1_heat_mpi
make all
make check
```

The same general procedure applies to the other case-study folders.

## Report and Presentation

Our complete report is available in the `report/` folder and our six-slide presentation is available in the `slides/` folder.

## Contributions

We worked together on the implementation, testing and analysis of all five case studies. Each member led their assigned case studies and reviewed the other's work.

Our detailed contribution breakdown is available in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
