
#!/usr/bin/env bash
set -euo pipefail

N=7
T=20
K=1

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

echo "===== FULL SMALL-GRID VERIFICATION ====="
echo "Grid: ${N}x${N}"
echo "Time steps: $T"
echo ""

# Run serial reference.
./serial "$N" "$T" > "$tmpdir/serial.out"

# Extract final grid from serial output.
awk -v n="$N" '
    /^Temperature After / {
        capture=1
        next
    }
    capture && count < n {
        print
        count++
        if (count == n) exit
    }
' "$tmpdir/serial.out" > "$tmpdir/reference.txt"

# Test all existing MPI implementations.
for program in \
    mpi_1d_demo \
    mpi_2d_demo \
    mpi_1d_nonblocking_demo
do
    echo "Testing $program"

    for p in 1 2 4 6; do

        output="$tmpdir/${program}_${p}.out"
        grid="$tmpdir/${program}_${p}.grid"

        mpirun -np "$p" "./$program" \
            "$N" "$T" "$K" > "$output"

        # Extract final MPI grid.
        awk -v n="$N" '
            /^Final Temperature Grid:/ {
                capture=1
                next
            }
            capture && count < n {
                print
                count++
                if (count == n) exit
            }
        ' "$output" > "$grid"

        if diff -q "$tmpdir/reference.txt" \
                   "$grid" > /dev/null; then
            echo "  $p processes: PASS"
        else
            echo "  $p processes: FAIL"
            diff -u "$tmpdir/reference.txt" "$grid" || true
            exit 1
        fi
    done

    echo ""
done

echo "===== ALL TESTS PASSED ====="
