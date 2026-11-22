
#!/usr/bin/env bash

set -euo pipefail

echo "==================================="
echo " MPI 2D COMMUNICATION BENCHMARK"
echo "==================================="

N=1024
T=500
K=50

for p in 1 2 4 6; do

    echo ""
    echo "==================================="
    echo "MPI Processes: $p"
    echo "Grid: ${N}x${N}"
    echo "Time Steps: $T"
    echo "Convergence K: $K"
    echo "==================================="

    for run in 1 2 3; do

        echo ""
        echo "---------- Run $run ----------"

        echo "2D BLOCKING:"

        mpirun -np "$p" \
            ./mpi_2d_demo "$N" "$T" "$K" \
            | grep -E "Execution Time|Checksum|Centre Temperature"

        echo ""

        echo "2D NON-BLOCKING:"

        mpirun -np "$p" \
            ./mpi_2d_nonblocking_demo "$N" "$T" "$K" \
            | grep -E "Execution Time|Max Post Time|Max Interior Compute Time|Max Waitall Time|Max Boundary Compute Time|Checksum|Centre Temperature"

    done
done

echo ""
echo "==================================="
echo " BENCHMARK COMPLETE"
echo "==================================="
