// Лабораторная работа №3: умножение матриц с использованием MPI.
// Матрица A делится по строкам между процессами, матрица B рассылается всем,
// каждый процесс считает свою полосу строк результата, строки собираются на rank 0.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

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
    cout << "Usage: matrix_mpi [--repeats R] [--warmup W] [--verify]\n"
            "                  [--generate N] [--seed S] [--a FILE] [--b FILE] [--out FILE]\n"
            "  --repeats R   число замеров (по умолчанию 1), выводятся среднее и минимум\n"
            "  --warmup W    число прогревных запусков (по умолчанию 1)\n"
            "  --verify      сверить результат с последовательным расчётом на rank 0\n"
            "  --generate N  создать A.txt и B.txt размера N (seed по умолчанию 42) и сразу считать\n"
            "  --a, --b      файлы исходных матриц (по умолчанию A.txt, B.txt)\n"
            "  --out         файл результата (по умолчанию result.txt)\n"
            "Запуск: mpiexec -n <число процессов> matrix_mpi\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    int exitCode = 0;
    try {
        int repeats = 1, warmup = 1;
        bool verify = false;
        int generatedSize = 0, seed = 42;
        string aFile = "A.txt", bFile = "B.txt", outFile = "result.txt";

        for (int i = 1; i < argc; ++i) {
            const string arg = argv[i];
            auto value = [&]() -> string {
                if (i + 1 >= argc) throw runtime_error("Missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--repeats") repeats = stoi(value());
            else if (arg == "--warmup") warmup = stoi(value());
            else if (arg == "--verify") verify = true;
            else if (arg == "--generate") generatedSize = stoi(value());
            else if (arg == "--seed") seed = stoi(value());
            else if (arg == "--a") aFile = value();
            else if (arg == "--b") bFile = value();
            else if (arg == "--out") outFile = value();
            else if (arg == "--help" || arg == "-h") { if (rank == 0) usage(); MPI_Finalize(); return 0; }
            else throw runtime_error("Unknown argument: " + arg);
        }
        if (repeats < 1) throw runtime_error("Repeats must be positive");
        if (warmup < 0) throw runtime_error("Warmup must be non-negative");

        // Режим --generate N: rank 0 создаёт A.txt и B.txt (чтобы на кластере не нужен был Python).
        if (generatedSize > 0) {
            if (rank == 0) {
                mt19937 rng(static_cast<unsigned>(seed));
                uniform_int_distribution<int> value(-5, 5);
                for (const string& name : {aFile, bFile}) {
                    ofstream file(name);
                    if (!file) throw runtime_error("Cannot write " + name);
                    file << generatedSize << '\n';
                    for (int i = 0; i < generatedSize; ++i) {
                        for (int j = 0; j < generatedSize; ++j) file << value(rng) << ' ';
                        file << '\n';
                    }
                }
                if (rank == 0 && generatedSize > 0) {
                    cerr << "generated " << generatedSize << "x" << generatedSize << " matrices ("
                         << aFile << ", " << bFile << ")" << '\n';
                }
            }
        }

        // Исходные матрицы читает только rank 0.
        int n = 0;
        Matrix a, b;
        if (rank == 0) {
            a = Matrix::read(aFile);
            b = Matrix::read(bFile);
            if (a.size() != b.size()) throw runtime_error("Different matrix sizes");
            n = a.size();
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n <= 0) throw runtime_error("Empty matrix");

        // Блочное распределение строк A (и строк результата C) между процессами.
        const int baseRows = n / procs, extra = n % procs;
        vector<int> counts(procs), displs(procs);
        for (int r = 0; r < procs; ++r) {
            const int rows = baseRows + (r < extra ? 1 : 0);
            counts[r] = rows * n;
            displs[r] = (baseRows * r + min(r, extra)) * n;
        }
        const int localElements = counts[rank];
        const int localRows = localElements / n;

        vector<double> localA(localElements), localC(localElements);
        vector<double> broadcastB(static_cast<size_t>(n) * n);
        vector<double> result(rank == 0 ? static_cast<size_t>(n) * n : 0);
        if (rank == 0) {
            copy(a.data(), a.data() + static_cast<size_t>(n) * n, broadcastB.begin());
        }

        // Один полный цикл: рассылка данных -> счёт -> сбор результата.
        auto runOnce = [&](double& total, double& compute, double& comm) {
            fill(localC.begin(), localC.end(), 0.0);
            const double t0 = MPI_Wtime();

            MPI_Scatterv(rank == 0 ? a.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                         localA.data(), localElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                copy(b.data(), b.data() + static_cast<size_t>(n) * n, broadcastB.begin());
            }
            MPI_Bcast(broadcastB.data(), static_cast<int>(static_cast<size_t>(n) * n), MPI_DOUBLE, 0,
                      MPI_COMM_WORLD);
            const double t1 = MPI_Wtime();

            for (int i = 0; i < localRows; ++i) {
                double* c = &localC[static_cast<size_t>(i) * n];
                const double* aRow = &localA[static_cast<size_t>(i) * n];
                for (int k = 0; k < n; ++k) {
                    const double number = aRow[k];
                    const double* bRow = &broadcastB[static_cast<size_t>(k) * n];
                    for (int j = 0; j < n; ++j) c[j] += number * bRow[j];
                }
            }
            const double t2 = MPI_Wtime();

            MPI_Gatherv(localC.data(), localElements, MPI_DOUBLE,
                        rank == 0 ? result.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            const double t3 = MPI_Wtime();

            total = t3 - t0;
            compute = t2 - t1;
            comm = (t1 - t0) + (t3 - t2);
        };

        for (int i = 0; i < warmup; ++i) {
            double total = 0, compute = 0, comm = 0;
            runOnce(total, compute, comm);
        }

        double bestTotal = numeric_limits<double>::max(), bestCompute = 0, bestComm = 0, sumTotal = 0;
        for (int r = 0; r < repeats; ++r) {
            double total = 0, compute = 0, comm = 0;
            runOnce(total, compute, comm);
            sumTotal += total;
            if (total < bestTotal) {
                bestTotal = total;
                bestCompute = compute;
                bestComm = comm;
            }
        }
        const double meanTotal = sumTotal / repeats;

        // Имена узлов: доказательство, что задача действительно шла на кластере.
        char hostName[MPI_MAX_PROCESSOR_NAME + 1];
        int nameLength = 0;
        MPI_Get_processor_name(hostName, &nameLength);
        if (nameLength < 0 || nameLength > MPI_MAX_PROCESSOR_NAME) nameLength = MPI_MAX_PROCESSOR_NAME;
        hostName[nameLength] = '\0';
        vector<char> allNames(static_cast<size_t>(procs) * MPI_MAX_PROCESSOR_NAME, 0);
        MPI_Allgather(hostName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allNames.data(),
                      MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

        if (rank == 0) {
            Matrix c(n);
            copy(result.begin(), result.end(), c.data());
            c.write(outFile);

            // Проверка: последовательный эталон на rank 0 против распределённого результата.
            string verifyText;
            if (verify) {
                Matrix reference(n);
                for (int i = 0; i < n; ++i) {
                    for (int k = 0; k < n; ++k) {
                        const double number = a.at(i, k);
                        for (int j = 0; j < n; ++j) reference.at(i, j) += number * b.at(k, j);
                    }
                }
                double maxDiff = 0.0, maxValue = 0.0;
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        maxDiff = max(maxDiff, fabs(c.at(i, j) - reference.at(i, j)));
                        maxValue = max(maxValue, fabs(reference.at(i, j)));
                    }
                const bool ok = maxDiff <= 1e-10 * max(1.0, maxValue);
                ostringstream text;
                text << " verify=" << (ok ? "PASS" : "FAIL") << " max_diff="
                     << scientific << setprecision(3) << maxDiff;
                verifyText = text.str();
            }

            set<string> uniqueHosts;
            for (int r = 0; r < procs; ++r) {
                uniqueHosts.insert(string(&allNames[static_cast<size_t>(r) * MPI_MAX_PROCESSOR_NAME]));
            }
            string hostList;
            for (const auto& item : uniqueHosts) {
                if (!hostList.empty()) hostList += ",";
                hostList += item;
            }

            const double operations = 2.0 * n * n * n;
            cout << fixed << setprecision(6) << "N=" << n << " procs=" << procs
                 << " nodes=" << uniqueHosts.size() << " repeats=" << repeats
                 << " warmup=" << warmup << " time_sec=" << meanTotal
                 << " time_min_sec=" << bestTotal << " compute_sec=" << bestCompute
                 << " comm_sec=" << bestComm << " operations=" << static_cast<long long>(operations)
                 << " memory_bytes=" << 3LL * n * n * static_cast<long long>(sizeof(double))
                 << " gflops=" << setprecision(3) << operations / bestTotal / 1e9 << " hosts=\""
                 << hostList << "\"" << verifyText << '\n';
        }
    } catch (const exception& error) {
        cerr << "rank " << rank << ": " << error.what() << '\n';
        exitCode = 1;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return exitCode;
}
