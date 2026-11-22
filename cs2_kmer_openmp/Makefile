
# ==========================================
# CS2 - k-mer OpenMP Learning Experiments
# ==========================================

CC = gcc

CFLAGS = -O2 -Wall -Wextra -std=c11
OMPFLAGS = -fopenmp

SERIAL_PROGRAMS = \
	kmer_learning_demo \
	serial_kmer_practice

OPENMP_PROGRAMS = \
	omp_race_demo \
	omp_chunk_demo \
	omp_histogram_demo \
	omp_fullsize_learning \
	false_sharing_learning \
	omp_kmer_race_learning

PROGRAMS = $(SERIAL_PROGRAMS) $(OPENMP_PROGRAMS)

.PHONY: all check verify-results plots benchmark clean

# ==========================================
# Build all programs
# ==========================================

all: $(PROGRAMS)
	@echo "All CS2 learning programs compiled."

# ==========================================
# Serial programs
# ==========================================

$(SERIAL_PROGRAMS): %: %.c
	$(CC) $(CFLAGS) $< -o $@

# ==========================================
# OpenMP programs
# ==========================================

$(OPENMP_PROGRAMS): %: %.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@

# ==========================================
# Quick correctness tests
# ==========================================

check: all
	@echo "===== CS2 CORRECTNESS TESTS ====="

	@./kmer_learning_demo | \
		grep -q "PASS: Counts match!"
	@echo "Basic k-mer encoding: PASS"

	@./serial_kmer_practice 1000 8 | \
		grep -q "Correctness: PASS"

	@./serial_kmer_practice 1000 12 | \
		grep -q "Correctness: PASS"
	@echo "Serial counting: PASS"

	@for p in 1 2 4 6; do \
		OMP_NUM_THREADS=$$p ./omp_chunk_demo | \
			grep -q "Correctness: PASS" || exit 1; \
		OMP_NUM_THREADS=$$p ./omp_histogram_demo | \
			grep -q "Correctness: PASS" || exit 1; \
	done
	@echo "OpenMP window and histogram tests: PASS"

	@./omp_fullsize_learning 1000 8 4 | \
		grep -q "Exact bin-by-bin correctness: PASS"

	@./omp_fullsize_learning 1000 12 4 | \
		grep -q "Exact bin-by-bin correctness: PASS"
	@echo "Full-size algorithm, small-input tests: PASS"

	@./omp_kmer_race_learning 2000000 8 4 | \
		grep -q "Atomic correctness: PASS"
	@echo "Atomic race demonstration: PASS"

	@echo "===== ALL QUICK TESTS PASSED ====="

# ==========================================
# Audit existing benchmark data
# ==========================================

verify-results:
	python3 verify_cs2_results.py

# ==========================================
# Regenerate graphs from saved results
# ==========================================

plots:
	python3 plot_cs2.py

# ==========================================
# Optional: Rerun all benchmarks
# ==========================================

benchmark: omp_fullsize_learning
	bash benchmark_cs2.sh

# ==========================================
# Remove executables only
# ==========================================

clean:
	rm -f $(PROGRAMS)
	@echo "Executables removed."
	@echo "Source files, logs and figures preserved."
