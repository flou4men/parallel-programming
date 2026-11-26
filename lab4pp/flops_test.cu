// Контрольный замер пиковой производительности GPU в FP64 и FP32.
// Нужен, чтобы понять, упирается ли умножение матриц в арифметику: на потребительских
// картах блоки FP64 работают в 1/64 от FP32, поэтому double-версия медленнее на порядки.
#include <cuda_runtime.h>

#include <cstdio>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error_ = (call);                                           \
        if (error_ != cudaSuccess) {                                           \
            printf("CUDA error: %s\n", cudaGetErrorString(error_));            \
            return 1;                                                          \
        }                                                                      \
    } while (0)

template <typename T>
__global__ void fmaKernel(T* out, int iterations) {
    T a0 = static_cast<T>(threadIdx.x + 1), a1 = a0 + T(1), a2 = a0 + T(2), a3 = a0 + T(3);
    T a4 = a0 + T(4), a5 = a0 + T(5), a6 = a0 + T(6), a7 = a0 + T(7);
    const T m = static_cast<T>(1.0000001);
    const T c = static_cast<T>(1e-12);
    for (int i = 0; i < iterations; ++i) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
    }
    if (blockIdx.x == 123456u) out[0] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

template <typename T>
double measure(int blocks, int threads, int iterations, const char* name) {
    T* out = nullptr;
    cudaMalloc(&out, sizeof(T));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fmaKernel<T><<<blocks, threads>>>(out, iterations);  // прогрев
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    fmaKernel<T><<<blocks, threads>>>(out, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    const double operations = 2.0 * blocks * threads * iterations * 8.0;  // 8 FMA = 16 FLOP на итерацию
    const double gflops = operations / (ms / 1000.0) / 1e9;
    printf("%-6s blocks=%d threads=%d iterations=%d -> %.1f Гфлоп/с\n", name, blocks, threads,
           iterations, gflops);
    cudaFree(out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return gflops;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s, SM=%d, ширина шины памяти %d бит\n", prop.name,
           prop.multiProcessorCount, prop.memoryBusWidth);

    const int blocks = prop.multiProcessorCount * 4;
    const int threads = 256;
    const int iterations = 200000;
    const double fp64 = measure<double>(blocks, threads, iterations, "FP64");
    const double fp32 = measure<float>(blocks, threads, iterations, "FP32");
    printf("отношение FP32/FP64 = %.1f\n", fp32 / fp64);
    return 0;
}
