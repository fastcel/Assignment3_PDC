
# ==========================================
# CS3 - Monte Carlo Learning Experiments
# ==========================================

CC = gcc

CFLAGS = -O2 -Wall -Wextra -std=c11
OMPFLAGS = -fopenmp
MATHFLAGS = -lm

SERIAL_PROGRAMS = \
	mc_serial_learning

OPENMP_PROGRAMS = \
	mc_rng_streams_learning \
	mc_reduction_learning \
	mc_openmp_learning \
	mc_fullsize_learning \
	mc_rng_correlation_learning

PROGRAMS = $(SERIAL_PROGRAMS) $(OPENMP_PROGRAMS)

.PHONY: all check verify-results plots benchmark clean

# ==========================================
# Build all learning programs
# ==========================================

all: $(PROGRAMS)
	@echo "All CS3 learning programs compiled."

# ==========================================
# Serial program
# ==========================================

$(SERIAL_PROGRAMS): %: %.c
	$(CC) $(CFLAGS) $< -o $@ $(MATHFLAGS)

# ==========================================
# OpenMP programs
# ==========================================

$(OPENMP_PROGRAMS): %: %.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(MATHFLAGS)

# ==========================================
# Quick correctness checks
# ==========================================

check: all
	@echo "===== CS3 CORRECTNESS TESTS ====="

	@./mc_serial_learning 1000 > /dev/null
	@./mc_serial_learning 10000 > /dev/null
	@echo "Serial Monte Carlo: PASS"

	@OMP_NUM_THREADS=4 \
		./mc_rng_streams_learning > /dev/null
	@echo "Thread-local RNG demonstration: PASS"

	@for p in 1 2 4 6; do \
		OMP_NUM_THREADS=$$p ./mc_reduction_learning | \
			grep -q "Correctness: PASS" || exit 1; \
	done
	@echo "OpenMP reduction: PASS"

	@./mc_openmp_learning 100000 4 > /dev/null
	@echo "Parallel Monte Carlo smoke test: PASS"

	@./mc_fullsize_learning 100000 4 > /dev/null
	@echo "Combined serial and parallel smoke test: PASS"

	@./mc_rng_correlation_learning 4 2000 | \
		grep -q "Identical worker results: YES"
	@echo "RNG duplication demonstration: PASS"

	@echo "===== ALL QUICK TESTS PASSED ====="

# ==========================================
# Verify existing experimental results
# ==========================================

verify-results:
	python3 verify_mc_results.py

# ==========================================
# Regenerate performance figures
# ==========================================

plots:
	python3 plot_mc_learning.py

# ==========================================
# Optional full benchmark
# ==========================================

benchmark: mc_fullsize_learning
	python3 benchmark_mc_learning.py

# ==========================================
# Remove compiled programs
# ==========================================

clean:
	rm -f $(PROGRAMS)
	@echo "Executables removed."
	@echo "Source files and results preserved."
