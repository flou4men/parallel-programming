# Сборка л/р №2 (OpenMP) компилятором g++/clang++
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -fopenmp -Wall -Wextra
TARGET   ?= matrix

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET) A.txt B.txt result.txt

.PHONY: all clean
