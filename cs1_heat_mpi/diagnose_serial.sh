
#!/usr/bin/env bash
set -euo pipefail

echo "===== SERIAL PERFORMANCE DIAGNOSTIC ====="

T=20

echo ""
echo "Testing nearby grid sizes with $T iterations"
echo ""

for N in 4095 4096 4097; do

    echo "================================"
    echo "SERIAL: N = $N"
    echo "================================"

    ./serial "$N" "$T" |
        grep -E "Grid Size|Execution Time|Checksum|Centre Temperature"

done

echo ""
echo "================================"
echo "MPI 1D: ONE PROCESS, N = 4096"
echo "================================"

mpirun -np 1 ./mpi_1d_demo 4096 "$T" 50 |
    grep -E "Execution Time|Checksum|Centre Temperature"

echo ""
echo "================================"
echo "MPI 2D: ONE PROCESS, N = 4096"
echo "================================"

mpirun -np 1 ./mpi_2d_demo 4096 "$T" 50 |
    grep -E "Execution Time|Checksum|Centre Temperature"

echo ""
echo "===== DIAGNOSTIC COMPLETE ====="
