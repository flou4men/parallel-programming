// Лабораторная работа №4: умножение матриц на CUDA.
// Два ядра: "наивное" (один поток на элемент C) и плиточное с разделяемой памятью.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error_ = (call);                                           \
        if (error_ != cudaSuccess)                                             \
            throw runtime_error(string("CUDA error: ") + cudaGetErrorString(error_)); \
    } while (0)

// ------------------------------------------------------------------ ядра ---

// Один поток вычисляет один элемент C: скалярное произведение строки A на столбец B.
__global__ void multiplyNaive(const double* a, const double* b, double* c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;
    const double* aRow = a + static_cast<size_t>(row) * n;
    double sum = 0.0;
    for (int k = 0; k < n; ++k) sum += aRow[k] * b[static_cast<size_t>(k) * n + col];
    c[static_cast<size_t>(row) * n + col] = sum;
}

// Плиточное умножение: блок потока TILE x TILE загружает плитки A и B
// в разделяемую память и переиспользует их для всех элементов своего блока C.
template <int TILE>
__global__ void multiplyTiled(const double* a, const double* b, double* c, int n) {
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    const int tiles = (n + TILE - 1) / TILE;

    double sum = 0.0;
    for (int t = 0; t < tiles; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;
        tileA[ty][tx] = (row < n && aCol < n) ? a[static_cast<size_t>(row) * n + aCol] : 0.0;
        tileB[ty][tx] = (bRow < n && col < n) ? b[static_cast<size_t>(bRow) * n + col] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) sum += tileA[ty][k] * tileB[k][tx];
        __syncthreads();
    }
    if (row < n && col < n) c[static_cast<size_t>(row) * n + col] = sum;
}

// ------------------------------------------------------------- хост-код ---

class Matrix {
    int n;
    vector<double> values;

public:
    Matrix() : n(0) {}
    explicit Matrix(int size) : n(size), values(static_cast<size_t>(size) * size, 0.0) {}
    double& at(int row, int col) { return values[static_cast<size_t>(row) * n + col]; }
    double at(int row, int col) const { return values[static_cast<size_t>(row) * n + col]; }
    const double* data() const { return values.data(); }
    double* data() { return values.data(); }
    int size() const { return n; }

    static Matrix read(const string& filename) {
        ifstream file(filename);
        int n;
        if (!(file >> n) || n <= 0) throw runtime_error("Invalid input: " + filename);
        Matrix result(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (!(file >> result.at(i, j))) throw runtime_error("Missing elements: " + filename);
        return result;
    }

    void write(const string& filename) const {
        ofstream file(filename);
        if (!file) throw runtime_error("Cannot write result");
        file << n << '\n' << setprecision(17);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) file << at(i, j) << ' ';
            file << '\n';
        }
    }
};

static void usage() {
    cout << "Usage: matrix_cuda [--kernel naive|tiled] [--block B] [--repeats R] [--warmup W]\n"
            "                   [--a FILE] [--b FILE] [--out FILE]\n"
            "  --kernel      naive  — один поток на элемент C (блок BxB)\n"
            "                tiled  — плиточное ядро с разделяемой памятью (плитка BxB, B = 8/16/32)\n"
            "  --block B     размер стороны блока/плитки (по умолчанию 16)\n"
            "  --repeats R   число замеров ядра (по умолчанию 1)\n"
            "  --warmup W    число прогревных запусков ядра (по умолчанию 1)\n"
            "  --a, --b      файлы исходных матриц (по умолчанию A.txt, B.txt)\n"
            "  --out         файл результата (по умолчанию result.txt)\n";
}

int main(int argc, char** argv) {
    try {
        string kernel = "tiled";
        int block = 16;
        int repeats = 1;
        int warmup = 1;
        string aFile = "A.txt", bFile = "B.txt", outFile = "result.txt";

        for (int i = 1; i < argc; ++i) {
            const string arg = argv[i];
            auto value = [&]() -> string {
                if (i + 1 >= argc) throw runtime_error("Missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--kernel") kernel = value();
            else if (arg == "--block") block = stoi(value());
            else if (arg == "--repeats") repeats = stoi(value());
            else if (arg == "--warmup") warmup = stoi(value());
            else if (arg == "--a") aFile = value();
            else if (arg == "--b") bFile = value();
            else if (arg == "--out") outFile = value();
            else if (arg == "--help" || arg == "-h") { usage(); return 0; }
            else throw runtime_error("Unknown argument: " + arg);
        }
        if (repeats < 1) throw runtime_error("Repeats must be positive");
        if (warmup < 0) throw runtime_error("Warmup must be non-negative");
        if (kernel != "naive" && kernel != "tiled") throw runtime_error("Unknown kernel: " + kernel);
        if (kernel == "tiled" && block != 8 && block != 16 && block != 32)
            throw runtime_error("Tiled kernel supports block = 8, 16, 32");
        if (block < 1 || block > 32) throw runtime_error("Block size must be between 1 and 32");

        const Matrix a = Matrix::read(aFile);
        const Matrix b = Matrix::read(bFile);
        if (a.size() != b.size()) throw runtime_error("Different matrix sizes");
        const int n = a.size();
        const size_t bytes = static_cast<size_t>(n) * n * sizeof(double);

        double* deviceA = nullptr;
        double* deviceB = nullptr;
        double* deviceC = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceA, bytes));
        CUDA_CHECK(cudaMalloc(&deviceB, bytes));
        CUDA_CHECK(cudaMalloc(&deviceC, bytes));

        cudaEvent_t copyStart, copyEnd;
        CUDA_CHECK(cudaEventCreate(&copyStart));
        CUDA_CHECK(cudaEventCreate(&copyEnd));

        CUDA_CHECK(cudaEventRecord(copyStart));
        CUDA_CHECK(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(copyEnd));
        CUDA_CHECK(cudaEventSynchronize(copyEnd));
        float h2dMs = 0;
        CUDA_CHECK(cudaEventElapsedTime(&h2dMs, copyStart, copyEnd));

        const dim3 blockShape(block, block);
        const dim3 grid((n + block - 1) / block, (n + block - 1) / block);

        auto launch = [&]() {
            if (kernel == "naive") {
                multiplyNaive<<<grid, blockShape>>>(deviceA, deviceB, deviceC, n);
            } else if (block == 8) {
                multiplyTiled<8><<<grid, blockShape>>>(deviceA, deviceB, deviceC, n);
            } else if (block == 16) {
                multiplyTiled<16><<<grid, blockShape>>>(deviceA, deviceB, deviceC, n);
            } else {
                multiplyTiled<32><<<grid, blockShape>>>(deviceA, deviceB, deviceC, n);
            }
            CUDA_CHECK(cudaGetLastError());
        };

        for (int i = 0; i < warmup; ++i) launch();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        double total = 0.0, best = numeric_limits<double>::max();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            CUDA_CHECK(cudaEventRecord(start));
            launch();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            const double seconds = ms / 1000.0;
            total += seconds;
            best = min(best, seconds);
        }
        const double mean = total / repeats;

        Matrix c(n);
        CUDA_CHECK(cudaEventRecord(copyStart));
        CUDA_CHECK(cudaMemcpy(c.data(), deviceC, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(copyEnd));
        CUDA_CHECK(cudaEventSynchronize(copyEnd));
        float d2hMs = 0;
        CUDA_CHECK(cudaEventElapsedTime(&d2hMs, copyStart, copyEnd));

        c.write(outFile);

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        const double operations = 2.0 * n * n * n;
        cout << fixed << setprecision(6) << "N=" << n << " kernel=" << kernel
             << " block=" << block << "x" << block << " grid=" << grid.x << "x" << grid.y
             << " repeats=" << repeats << " warmup=" << warmup
             << " time_sec=" << mean << " time_min_sec=" << best
             << " h2d_sec=" << h2dMs / 1000.0 << " d2h_sec=" << d2hMs / 1000.0
             << " operations=" << static_cast<long long>(operations)
             << " memory_bytes=" << 3LL * n * n * static_cast<long long>(sizeof(double))
             << " gflops=" << setprecision(3) << operations / best / 1e9
             << " device=\"" << prop.name << "\" sm=" << prop.major << prop.minor << '\n';

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaEventDestroy(copyStart));
        CUDA_CHECK(cudaEventDestroy(copyEnd));
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
    } catch (const exception& error) {
        cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
