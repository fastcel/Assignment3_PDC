CC = gcc
MPICC = mpicc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDLIBS = -lm

MPI_PROGRAMS = mpi_1d_demo mpi_2d_demo mpi_1d_nonblocking_demo mpi_2d_nonblocking_demo

.PHONY: all clean check help

all: serial $(MPI_PROGRAMS)

serial: serial.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

mpi_1d_demo: mpi_1d_demo.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDLIBS)

mpi_2d_demo: mpi_2d_demo.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDLIBS)

mpi_1d_nonblocking_demo: mpi_1d_nonblocking_demo.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDLIBS)

mpi_2d_nonblocking_demo: mpi_2d_nonblocking_demo.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDLIBS)

check: all
	bash verify_full_grid.sh

clean:
	rm -f serial $(MPI_PROGRAMS)

help:
	@echo "Available commands:"
	@echo "  make       - Compile all implementations"
	@echo "  make check - Run correctness tests"
	@echo "  make clean - Remove executables"
