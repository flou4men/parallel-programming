# Сборка л/р №3 (MPI). На кластере: module load impi/4 && make
MPICC   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
TARGET  ?= matrix_mpi

all: $(TARGET)

$(TARGET): main_mpi.cpp
	$(MPICC) $(CXXFLAGS) main_mpi.cpp -o $(TARGET)

clean:
	rm -f $(TARGET) A.txt B.txt result.txt

.PHONY: all clean
