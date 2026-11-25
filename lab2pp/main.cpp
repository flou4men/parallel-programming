#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
// Заглушки, чтобы программа собиралась и без -fopenmp / /openmp
// (в этом случае она работает последовательно).
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_procs() { return 1; }
#endif

using namespace std;

class Matrix {
    int n;
    vector<double> values;
public:
    Matrix() : n(0) {}
    explicit Matrix(int size) : n(size), values(static_cast<size_t>(size) * size, 0) {}

    double& at(int row, int col) { return values[static_cast<size_t>(row) * n + col]; }
    double at(int row, int col) const { return values[static_cast<size_t>(row) * n + col]; }
    double* row(int index) { return values.data() + static_cast<size_t>(index) * n; }
    const double* row(int index) const { return values.data() + static_cast<size_t>(index) * n; }
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

// Умножение квадратных матриц, распараллеленное по строкам результата.
// Каждый поток вычисляет свою полосу строк C[i][*]; данные A и B только читаются,
// поэтому гонок нет, а синхронизация после цикла выполняется автоматически.
Matrix multiply(const Matrix& a, const Matrix& b, int threads) {
    if (a.size() != b.size()) throw runtime_error("Different matrix sizes");
    const int n = a.size();
    Matrix result(n);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < n; ++i) {
        double* c = result.row(i);
        for (int k = 0; k < n; ++k) {
            const double number = a.row(i)[k];
            const double* other = b.row(k);
            for (int j = 0; j < n; ++j) c[j] += number * other[j];
        }
    }
    return result;
}

static void usage() {
    cout << "Usage: matrix [--threads N] [--repeats R] [--warmup W] [--a FILE] [--b FILE] [--out FILE]\n"
            "  --threads N   number of OpenMP threads (default: max available)\n"
            "  --repeats R   how many times to multiply, mean/min time is reported (default: 1)\n"
            "  --warmup W    untimed multiplications before measuring (default: 1)\n"
            "  --a FILE      first matrix  (default: A.txt)\n"
            "  --b FILE      second matrix (default: B.txt)\n"
            "  --out FILE    result matrix (default: result.txt)\n";
}

int main(int argc, char** argv) {
    try {
        int threads = omp_get_max_threads();
        int repeats = 1;
        int warmup = 1;
        string aFile = "A.txt", bFile = "B.txt", outFile = "result.txt";

        for (int i = 1; i < argc; ++i) {
            const string arg = argv[i];
            auto value = [&]() -> string {
                if (i + 1 >= argc) throw runtime_error("Missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--threads") {
                threads = stoi(value());
            } else if (arg == "--repeats") {
                repeats = stoi(value());
            } else if (arg == "--warmup") {
                warmup = stoi(value());
            } else if (arg == "--a") {
                aFile = value();
            } else if (arg == "--b") {
                bFile = value();
            } else if (arg == "--out") {
                outFile = value();
            } else if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            } else {
                throw runtime_error("Unknown argument: " + arg);
            }
        }
        if (threads < 1) throw runtime_error("Threads must be positive");
        if (repeats < 1) throw runtime_error("Repeats must be positive");
        if (warmup < 0) throw runtime_error("Warmup must be non-negative");

        const Matrix a = Matrix::read(aFile);
        const Matrix b = Matrix::read(bFile);

        Matrix c;
        // Прогрев: первый запуск поднимает частоту процессора и заполняет кэш,
        // поэтому в измерение он не включается.
        for (int i = 0; i < warmup; ++i) c = multiply(a, b, threads);

        double total = 0, best = numeric_limits<double>::max();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            const auto start = chrono::steady_clock::now();
            c = multiply(a, b, threads);
            const double seconds =
                chrono::duration<double>(chrono::steady_clock::now() - start).count();
            total += seconds;
            best = min(best, seconds);
        }
        const double mean = total / repeats;

        c.write(outFile);

        const long long n = a.size();
        const double operations = 2.0 * n * n * n;
        cout << fixed << setprecision(6) << "N=" << n << " threads=" << threads
             << " repeats=" << repeats << " warmup=" << warmup
             << " time_sec=" << mean << " time_min_sec=" << best
             << " operations=" << static_cast<long long>(operations)
             << " memory_bytes=" << 3LL * n * n * static_cast<long long>(sizeof(double))
             << " gflops=" << setprecision(3) << operations / best / 1e9
             << " omp_procs=" << omp_get_num_procs() << '\n';
    } catch (const exception& error) {
        cerr << error.what() << '\n';
        return 1;
    }
}
