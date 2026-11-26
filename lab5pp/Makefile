# Сборка л/р №5 (Mandelbrot на std::thread)
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -pthread -Wall -Wextra
TARGET   ?= mandelbrot

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET) *.ppm *.bin counts_view*.bin

.PHONY: all clean
