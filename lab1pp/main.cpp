#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace std;

class Matrix {
    int n;
    vector<double> values;
public:
    explicit Matrix(int size) : n(size), values(size * size, 0) {}
    double& at(int row, int col) { return values[row * n + col]; }
    double at(int row, int col) const { return values[row * n + col]; }
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

    Matrix operator*(const Matrix& other) const {
        if (n != other.n) throw runtime_error("Different matrix sizes");
        Matrix result(n);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k) {
                double number = at(i, k);
                for (int j = 0; j < n; ++j)
                    result.at(i, j) += number * other.at(k, j);
            }
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

int main() {
    try {
        Matrix a = Matrix::read("A.txt");
        Matrix b = Matrix::read("B.txt");
        auto start = chrono::steady_clock::now();
        Matrix c = a * b;
        double seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();
        c.write("result.txt");
        long long n = a.size();
        cout << fixed << setprecision(6) << "time_sec=" << seconds
             << " N=" << n << " operations=" << 2LL * n * n * n
             << " memory_bytes=" << 3LL * n * n * sizeof(double) << '\n';
    } catch (const exception& error) {
        cerr << error.what() << '\n';
        return 1;
    }
}
