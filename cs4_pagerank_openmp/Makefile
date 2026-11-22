
# ==========================================
# CS4 - Parallel PageRank Learning Workspace
# ==========================================

CC = gcc

CFLAGS = -O2 -Wall -Wextra -std=c11
OMPFLAGS = -fopenmp
LDLIBS = -lm

GRAPH = graphs_cs4/synthetic_edges.txt

# ==========================================
# Serial learning programs
# ==========================================

SERIAL_PROGRAMS = \
	pagerank_learning_demo \
	pagerank_convergence_learning \
	csr_loader_learning \
	pagerank_serial_large_learning

# ==========================================
# OpenMP learning programs
# ==========================================

OPENMP_PROGRAMS = \
	pagerank_push_pull_learning \
	pagerank_atomic_learning \
	pagerank_parallel_pull_learning \
	pagerank_parallel_push_learning \
	pagerank_scheduling_learning

PROGRAMS = $(SERIAL_PROGRAMS) $(OPENMP_PROGRAMS)

.PHONY: all check verify-results plots graph \
        benchmark-scheduling benchmark-algorithms clean

# ==========================================
# Build all programs
# ==========================================

all: $(PROGRAMS)
	@echo "All CS4 learning programs compiled."

# ==========================================
# Compile serial programs
# ==========================================

$(SERIAL_PROGRAMS): %: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

# ==========================================
# Compile OpenMP programs
# ==========================================

$(OPENMP_PROGRAMS): %: %.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDLIBS)

# ==========================================
# Quick correctness checks
# ==========================================

check: all
	@echo "===== CS4 CORRECTNESS TESTS ====="

	@./pagerank_learning_demo > /dev/null
	@echo "Tiny CSR PageRank: PASS"

	@./pagerank_convergence_learning > /dev/null
	@echo "Serial convergence: PASS"

	@for p in 1 2 4; do \
		./pagerank_push_pull_learning $$p \
			> /dev/null || exit 1; \
	done
	@echo "Push vs pull small graph: PASS"

	@for p in 1 2 4 8; do \
		./pagerank_atomic_learning $$p \
			> /dev/null || exit 1; \
	done
	@echo "Atomic push small graph: PASS"

	@test -f $(GRAPH) || \
		(echo "ERROR: Graph file missing"; exit 1)

	@./csr_loader_learning $(GRAPH) > /dev/null
	@echo "Large CSR loader: PASS"

	@./pagerank_serial_large_learning \
		$(GRAPH) > /dev/null
	@echo "Large serial PageRank: PASS"

	@for p in 1 2 4 8; do \
		./pagerank_parallel_pull_learning \
			$(GRAPH) $$p > /dev/null || exit 1; \
	done
	@echo "Large parallel pull: PASS"

	@for p in 1 2 4 8; do \
		./pagerank_parallel_push_learning \
			$(GRAPH) $$p > /dev/null || exit 1; \
	done
	@echo "Large atomic push: PASS"

	@for schedule in static dynamic guided; do \
		./pagerank_scheduling_learning \
			$(GRAPH) 4 $$schedule 64 \
			> /dev/null || exit 1; \
	done
	@echo "OpenMP scheduling correctness: PASS"

	@echo "===== ALL CS4 QUICK TESTS PASSED ====="

# ==========================================
# Verify saved benchmark results
# ==========================================

verify-results:
	python3 verify_pagerank_results_learning.py

# ==========================================
# Regenerate performance graphs
# ==========================================

plots:
	python3 plot_pagerank_learning.py

# ==========================================
# Regenerate synthetic graph
# ==========================================

graph:
	python3 generate_graph_learning.py

# ==========================================
# Optional full benchmarks
# ==========================================

benchmark-scheduling: pagerank_scheduling_learning
	python3 benchmark_pagerank_scheduling_learning.py

benchmark-algorithms: pagerank_parallel_push_learning \
                      pagerank_scheduling_learning
	python3 benchmark_pagerank_algorithms_learning.py

# ==========================================
# Remove compiled executables
# ==========================================

clean:
	rm -f $(PROGRAMS)
	@echo "Compiled executables removed."
	@echo "Source files, graph, results and figures preserved."
