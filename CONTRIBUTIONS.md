# CONTRIBUTIONS.md - Group 10

**Course:** CS3006 - Parallel and Distributed Computing  
**Repository:** https://github.com/fastcel/Assignment3_PDC  
**Group members:** Shehryar Hassan (23L-0603) and Meerab Munir (23L-0971)

## Working arrangement and estimated effort

Both of us worked together in person on a **shared workstation**, participating in coding, experiments, debugging, analysis, and discussion. The lead/reviewer designations below describe our actual **primary focus and review responsibilities**, not exclusive authorship of all code in that folder. Both of us estimate **16–25 hours** of participation; precise totals were not independently logged and are therefore not claimed.

| Member          | Roll number | Primary lead studies | Reviewed studies | Estimated total effort |
| --------------- | ----------- | -------------------- | ---------------- | ---------------------- |
| Shehryar Hassan | 23L-0603    | CS1, CS4, CS5        | CS2, CS3         | 16–25 hours            |
| Meerab Munir    | 23L-0971    | CS2, CS3             | CS1, CS4, CS5    | 16–25 hours            |

## Study ownership and peer review

| Case study                  | Lead     | Reviewer | Scope and file evidence                                                                                                                                                        |
| --------------------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CS1 - MPI heat diffusion    | Shehryar | Meerab   | 1D/2D decomposition, blocking/nonblocking halo exchange, strong/weak scaling; `cs1_heat_mpi/` code, README, 4096-grid logs and charts                                          |
| CS2 - OpenMP k-mer counting | Meerab   | Shehryar | Shared atomics and private histogram counting, k=8/12 scaling, false-sharing experiment; `cs2_kmer_openmp/` code, README, benchmark CSVs and charts                            |
| CS3 - OpenMP Monte Carlo    | Meerab   | Shehryar | Path parallelism, separate random streams, accuracy/standard error, strong scaling; `cs3_montecarlo/` code, README, saved CSVs and charts                                      |
| CS4 - OpenMP PageRank       | Shehryar | Meerab   | CSR push/pull, convergence, scheduling/chunk size, preferential graph, thread-time imbalance; `cs4_pagerank_openmp/` code, README and experimental results                     |
| CS5 - MPI sample sort       | Shehryar | Meerab   | Sampling/splitters, Alltoallv redistribution, duplicate-aware native-32-bit sorting, communication modeling and correctness; `cs5_samplesort_mpi/` code, README, CSVs and logs |

We have **reviewed each other's case studies**. The group-wide local verification campaign recorded nine passed checks: five study `make check` targets, native-32-bit CS5 checks, and saved-result verification for CS2–CS4. The tests were carried out on the shared development environment. Reviewer attribution reflects joint review; the command logs do not distinguish which member typed each command.

## Shared responsibilities

- **Design and development:** Pair work, joint troubleshooting and experiment interpretation on the same workstation.
- **Integration, testing and benchmarking:** Shared; both discussed the implementation, correctness and interpretation of the outputs.
- **Report and cross-case analysis:** Joint responsibility. The final report covers all five studies and the impact of computation, communication, scheduling and memory.
- **Slides and viva preparation:** Shared responsibility; the presentation allocates CS1/CS4/CS5 to Shehryar and CS2/CS3 to Meerab as primary presenters, while both can be questioned about any study.

## Git provenance - important clarification

Initial development and testing occurred **collaboratively on Meerab's machine**, so the commits created under its configured Git identity cannot be used to infer that only one member contributed. The repository was initialized after much of the implementation/benchmarking work was already complete. Shehryar subsequently cloned the same repository to his own laptop for ongoing genuine revisions.

We have documented our actual responsibilities through the lead and reviewer assignments above. We have not changed or fabricated Git history to represent work that was completed earlier.

**Repository:** https://github.com/fastcel/Assignment3_PDC

## Group affirmation

The responsibilities and the 16–25-hour ranges above are based on the two members' reported collaboration. Both members acknowledge that the early Git author field reflects a shared-machine workflow and is not a reliable standalone measure of effort. No individual authorship or historical commit has been invented.

**Shehryar Hassan (23L-0603)** - reviewed and confirmed: DONE  
**Meerab Munir (23L-0971)** - reviewed and confirmed: DONE
