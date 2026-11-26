# Сборка л/р №4 (CUDA) напрямую через nvcc
NVCC     ?= nvcc
NVFLAGS  ?= -O2 -arch=sm_89
TARGET   ?= matrix_cuda

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVFLAGS) main.cu -o $(TARGET)

clean:
	rm -f $(TARGET) A.txt B.txt result.txt

.PHONY: all clean
