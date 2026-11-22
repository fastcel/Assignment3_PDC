
# ==================================================
# CS5 — MPI SAMPLE SORT LEARNING MAKEFILE
# ==================================================

CC = gcc
MPICC = mpicc

CFLAGS = -O2 -Wall -Wextra -std=c11

PYTHON = python3

# --------------------------------------------------
# MPI learning programs
# --------------------------------------------------

MPI_PROGRAMS = \
	sample_selection_learning \
	sample_redistribution_learning \
	sample_balance_learning \
	sample_tiebreak_learning \
	sample_packed_learning \
	sample_sort_repro_learning \
	verify_input_partitions_learning \
	mpi_pingpong_learning \
	inspect_samplesort_traffic_learning \
	sample_sort_uint32_learning

# --------------------------------------------------
# Serial learning program
# --------------------------------------------------

SERIAL_PROGRAMS = \
	sample_serial_baseline_learning \
	sample_serial_uint32_learning

PROGRAMS = $(MPI_PROGRAMS) $(SERIAL_PROGRAMS)

.PHONY: all check plots communication clean

# ==================================================
# TARGET 1: Compile everything
# ==================================================

all: $(PROGRAMS)

$(MPI_PROGRAMS): %: %.c
	$(MPICC) $(CFLAGS) $< -o $@

$(SERIAL_PROGRAMS): %: %.c
	$(CC) $(CFLAGS) $< -o $@

# Recompile programs when the shared
# input-generation header changes.

sample_sort_repro_learning \
verify_input_partitions_learning \
inspect_samplesort_traffic_learning \
sample_serial_baseline_learning: sample_input_learning.h

# ==================================================
# TARGET 2: Automated correctness tests
# ==================================================

# Test 1 million integers under:
#
# - uniform and duplicate-heavy input
# - 1, 2, 4 and 8 MPI processes
# - oversampling factor 4
#
# Compare the fingerprints against the
# separately compiled serial reference.
#
# All results are written to smoke_logs.

check: all
	@mkdir -p results_cs5/smoke_logs
	@set -eu; \
	for distribution in uniform duplicates; do \
		echo ""; \
		echo "===== TESTING $$distribution ====="; \
		serial_log="results_cs5/smoke_logs/$${distribution}_serial.txt"; \
		./sample_serial_baseline_learning \
			1000000 "$$distribution" > "$$serial_log"; \
		grep -q '^Correctness: PASS$$' "$$serial_log"; \
		sum=$$(sed -n 's/^Global fingerprint sum: //p' "$$serial_log"); \
		xor=$$(sed -n 's/^Global fingerprint xor: //p' "$$serial_log"); \
		for p in 1 2 4 8; do \
			log="results_cs5/smoke_logs/$${distribution}_p$${p}.txt"; \
			mpirun --use-hwthread-cpus -np "$$p" \
				./sample_sort_repro_learning \
				1000000 "$$distribution" 4 > "$$log"; \
			grep -q '^Correctness: PASS$$' "$$log"; \
			grep -Fqx "Global fingerprint sum: $$sum" "$$log"; \
			grep -Fqx "Global fingerprint xor: $$xor" "$$log"; \
			echo "$$distribution | MPI p=$$p | PASS"; \
		done; \
	done; \
	echo ""; \
	echo "===== ALL SAMPLE SORT TESTS PASSED ====="

# ==================================================
# TARGET 3: Regenerate performance figures
# ==================================================

plots:
	$(PYTHON) plot_samplesort_scaling_learning.py

# ==================================================
# TARGET 4: Regenerate communication analysis
# ==================================================

communication:
	$(PYTHON) analyze_mpi_communication_learning.py
	$(PYTHON) compare_actual_traffic_learning.py

# ==================================================
# TARGET 5: Remove compiled executables only
# ==================================================

# This does NOT delete:
# - source code
# - CSV files
# - saved benchmark logs
# - figures

clean:
	rm -f $(PROGRAMS)


# ==================================================
# NATIVE 32-BIT EDUCATIONAL TEST TARGETS
# ==================================================

.PHONY: check-uint32 plots-uint32

sample_sort_uint32_learning sample_serial_uint32_learning: sample_input_learning.h

check-uint32: all
	@mkdir -p results_cs5/uint32_smoke_logs
	@set -eu; \
	for distribution in uniform duplicates; do \
		serial_log="results_cs5/uint32_smoke_logs/$${distribution}_serial.txt"; \
		./sample_serial_uint32_learning 1000000 "$$distribution" > "$$serial_log"; \
		grep -Fqx 'Correctness: PASS' "$$serial_log"; \
		sum=$$(sed -n 's/^Input fingerprint sum: //p' "$$serial_log"); \
		xor=$$(sed -n 's/^Input fingerprint xor: //p' "$$serial_log"); \
		for p in 1 2 4 6 8; do \
			log="results_cs5/uint32_smoke_logs/$${distribution}_p$${p}.txt"; \
			mpirun --use-hwthread-cpus -np "$$p" \
				./sample_sort_uint32_learning \
				1000000 "$$distribution" 4 > "$$log"; \
			grep -Fqx 'Correctness: PASS' "$$log"; \
			grep -Fqx 'Element bytes: 4' "$$log"; \
			grep -Fqx 'MPI payload datatype: MPI_UINT32_T' "$$log"; \
			grep -Fqx "Global fingerprint sum: $$sum" "$$log"; \
			grep -Fqx "Global fingerprint xor: $$xor" "$$log"; \
			echo "$$distribution | 32-bit MPI p=$$p | PASS"; \
		done; \
	done; \
	echo '===== ALL NATIVE 32-BIT TESTS PASSED ====='

plots-uint32:
	$(PYTHON) plot_uint32_final_learning.py
