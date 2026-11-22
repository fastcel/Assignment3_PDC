
#!/usr/bin/env bash
set -euo pipefail

echo "===== PREPARING CASE STUDY 1 ====="

# Create a fresh, separate review folder
DEST=$(mktemp -d ../cs1_review_XXXXXX)

mkdir -p "$DEST/diagnostics"

echo "Review folder: $DEST"

# Core implementations and build instructions
cp -p \
    Makefile \
    README.md \
    serial.c \
    mpi_1d_demo.c \
    mpi_2d_demo.c \
    mpi_1d_nonblocking_demo.c \
    mpi_2d_nonblocking_demo.c \
    "$DEST/"

# Reproducibility and testing scripts
cp -p \
    verify_full_grid.sh \
    verify_small.sh \
    verify_optimized.sh \
    benchmark_2d.sh \
    benchmark_4096.sh \
    benchmark_padded_serial.sh \
    stride_probe.c \
    plot_cs1.py \
    "$DEST/"

# Original experimental results and graphs
cp -a \
    results_4096 \
    results_padding \
    figures_cs1 \
    "$DEST/"

# Development experiments and old versions
cp -p \
    boundary_demo.c \
    diagnose_2d.sh \
    diagnose_serial.sh \
    mpi_2d_nonblocking_before_optimization.c \
    "$DEST/diagnostics/"

echo ""
echo "===== CHECKING REVIEW PACKAGE ====="

make -C "$DEST"

echo ""
echo "===== PACKAGE CONTENTS ====="
ls -lh "$DEST"

echo ""
echo "===== FINAL LOCATION ====="
realpath "$DEST"

echo ""
echo "Review package created successfully."
echo "Your original project folder is unchanged."
