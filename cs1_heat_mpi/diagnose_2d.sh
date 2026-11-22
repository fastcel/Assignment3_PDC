
#!/usr/bin/env bash
set -euo pipefail

N=1024
T=500
K=50

echo "===== MPI 2D OPTIMIZATION DIAGNOSTIC ====="

# Build both variants using identical compiler settings
for opt in O2 O3; do

    echo ""
    echo "Compiling with $opt"

    extra_flags=()

    if [[ "$opt" == "O3" ]]; then
        extra_flags=(-march=native)
    fi

    mpicc "-$opt" "${extra_flags[@]}" \
        -Wall -Wextra -std=c11 \
        mpi_2d_demo.c -lm \
        -o "mpi_2d_block_$opt"

    mpicc "-$opt" "${extra_flags[@]}" \
        -Wall -Wextra -std=c11 \
        mpi_2d_nonblocking_demo.c -lm \
        -o "mpi_2d_nonblock_$opt"

done

# Benchmark the new binaries
for opt in O2 O3; do

    echo ""
    echo "===== OPTIMIZATION: $opt ====="

    for p in 1 4; do

        echo ""
        echo "MPI processes: $p"

        for version in block nonblock; do

            program="./mpi_2d_${version}_$opt"
            times=$(mktemp)

            echo "$version:"

            for run in 1 2 3; do

                result=$(
                    mpirun -np "$p" "$program" "$N" "$T" "$K" |
                    awk '/^Execution Time:/ {print $3}'
                )

                echo "Run $run: $result seconds"
                echo "$result" >> "$times"

            done

            median=$(sort -n "$times" | sed -n '2p')

            echo "MEDIAN: $median seconds"
            rm -f "$times"

        done
    done
done

echo ""
echo "===== DIAGNOSTIC COMPLETE ====="
