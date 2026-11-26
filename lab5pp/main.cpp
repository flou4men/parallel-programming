// Лабораторная работа №5: параллельная генерация фрактала Мандельброта на std::thread.
// Основано на задании Program 1 из Stanford CS149 asst1 (собственная реализация).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std;

struct View {
    float x0, y0, x1, y1;
};

// Точка комплексной плоскости: возвращает число итераций до выхода за |z| > 2.
static inline int escapeIterations(float cRe, float cIm, int maxIterations) {
    float zRe = cRe;
    float zIm = cIm;
    int iteration = 0;
    for (; iteration < maxIterations; ++iteration) {
        if (zRe * zRe + zIm * zIm > 4.0f) break;
        const float newRe = zRe * zRe - zIm * zIm;
        const float newIm = 2.0f * zRe * zIm;
        zRe = cRe + newRe;
        zIm = cIm + newIm;
    }
    return iteration;
}

struct Renderer {
    View view;
    int width = 1600;
    int height = 1200;
    int maxIterations = 256;
    vector<uint16_t> counts;
    atomic<int> nextRow{0};

    Renderer() : counts(static_cast<size_t>(width) * height, 0) {}

    void resize() { counts.assign(static_cast<size_t>(width) * height, 0); }

    // Одна строка изображения: y = y0 + row * dy, x = x0 + i * dx.
    void renderRow(int row, uint64_t& iterationsDone) {
        const float dx = (view.x1 - view.x0) / static_cast<float>(width);
        const float dy = (view.y1 - view.y0) / static_cast<float>(height);
        const float y = view.y0 + static_cast<float>(row) * dy;
        uint16_t* line = &counts[static_cast<size_t>(row) * width];
        uint64_t local = 0;
        for (int i = 0; i < width; ++i) {
            const int value = escapeIterations(view.x0 + static_cast<float>(i) * dx, y, maxIterations);
            line[i] = static_cast<uint16_t>(value);
            local += static_cast<uint64_t>(value);
        }
        iterationsDone += local;
    }
};

static void writePpm(const Renderer& renderer, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot write " + filename);
    file << "P6\n" << renderer.width << ' ' << renderer.height << "\n255\n";
    for (const uint16_t value : renderer.counts) {
        const float t = renderer.maxIterations > 0
                            ? static_cast<float>(value) / static_cast<float>(renderer.maxIterations)
                            : 0.0f;
        const unsigned char r = static_cast<unsigned char>(255.0f * min(1.0f, 2.0f * t));
        const unsigned char g = static_cast<unsigned char>(255.0f * min(1.0f, 3.0f * t * t));
        const unsigned char b = static_cast<unsigned char>(255.0f * min(1.0f, 4.0f * sqrt(t)));
        const unsigned char pixel[3] = {value == 0 ? 0 : r, g, b};
        file.write(reinterpret_cast<const char*>(pixel), 3);
    }
}

static void writeCounts(const Renderer& renderer, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot write " + filename);
    file.write(reinterpret_cast<const char*>(renderer.counts.data()),
               static_cast<streamsize>(renderer.counts.size() * sizeof(uint16_t)));
}

static View makeView(int index) {
    View view{-2.0f, -1.0f, 1.0f, 1.0f};
    if (index == 2) {
        // Приближение к «долине» множества: как параметры view 2 в asst1.
        const float scale = 0.015f, shiftX = -0.986f, shiftY = 0.30f;
        view.x0 = view.x0 * scale + shiftX;
        view.x1 = view.x1 * scale + shiftX;
        view.y0 = view.y0 * scale + shiftY;
        view.y1 = view.y1 * scale + shiftY;
    } else if (index != 1) {
        throw runtime_error("Unknown view: " + to_string(index));
    }
    return view;
}

struct RunResult {
    double seconds = 0.0;
    uint64_t iterations = 0;
    vector<double> threadTimes;
    vector<uint64_t> threadIterations;
};

// mode: serial | block | interleaved | dynamic
static RunResult render(Renderer& renderer, const string& mode, int threads, int repeats) {
    RunResult best;
    best.seconds = 1e30;
    vector<vector<double>> times(threads);

    for (int repeat = 0; repeat < repeats; ++repeat) {
        fill(renderer.counts.begin(), renderer.counts.end(), static_cast<uint16_t>(0));
        renderer.nextRow.store(0, memory_order_relaxed);
        vector<uint64_t> iterations(threads, 0);
        vector<double> duration(threads, 0.0);

        const auto start = chrono::steady_clock::now();
        if (mode == "serial") {
            const auto threadStart = chrono::steady_clock::now();
            uint64_t done = 0;
            for (int row = 0; row < renderer.height; ++row) renderer.renderRow(row, done);
            duration[0] = chrono::duration<double>(chrono::steady_clock::now() - threadStart).count();
            iterations[0] = done;
        } else {
            vector<thread> pool;
            pool.reserve(threads);
            for (int t = 0; t < threads; ++t) {
                pool.emplace_back([&, t]() {
                    const auto threadStart = chrono::steady_clock::now();
                    uint64_t done = 0;
                    if (mode == "block") {
                        // Непрерывный блок строк на поток: вариант из пунктов 1-2 задания.
                        const int base = renderer.height / threads;
                        const int extra = renderer.height % threads;
                        const int first = t * base + min(t, extra);
                        const int last = first + base + (t < extra ? 1 : 0);
                        for (int row = first; row < last; ++row) renderer.renderRow(row, done);
                    } else if (mode == "interleaved") {
                        // Строки через одну: потоки получают примерно одинаковую работу без синхронизации.
                        for (int row = t; row < renderer.height; row += threads) renderer.renderRow(row, done);
                    } else if (mode == "dynamic") {
                        // Динамическая раздача строк через атомарный счётчик (для сравнения).
                        for (;;) {
                            const int row = renderer.nextRow.fetch_add(1, memory_order_relaxed);
                            if (row >= renderer.height) break;
                            renderer.renderRow(row, done);
                        }
                    } else {
                        throw runtime_error("Unknown mode: " + mode);
                    }
                    duration[t] = chrono::duration<double>(chrono::steady_clock::now() - threadStart).count();
                    iterations[t] = done;
                });
            }
            for (auto& item : pool) item.join();
        }
        const double seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();

        const uint64_t totalIterations =
            accumulate(iterations.begin(), iterations.end(), static_cast<uint64_t>(0));
        if (seconds < best.seconds) {
            best.seconds = seconds;
            best.iterations = totalIterations;
            best.threadTimes = mode == "serial" ? vector<double>{seconds} : duration;
            best.threadIterations = iterations;
        }
    }
    if (mode == "serial") best.threadTimes = vector<double>{best.seconds};
    return best;
}

static void usage() {
    cout << "Usage: mandelbrot [--view 1|2] [--width W] [--height H] [--max-iter M]\n"
            "                   [--mode serial|block|interleaved|dynamic] [--threads T]\n"
            "                   [--repeats R] [--verify] [--ppm FILE] [--counts FILE]\n"
            "  --view        вид фрактала: 1 — весь набор, 2 — приближение\n"
            "  --mode        serial — один поток; block — непрерывные блоки строк;\n"
            "                interleaved — строки через одну; dynamic — атомарная раздача строк\n"
            "  --threads T   число потоков (по умолчанию 1)\n"
            "  --repeats R   число замеров, выводится минимум (по умолчанию 1)\n"
            "  --verify      сверить результат с последовательным и напечатать verify=PASS/FAIL\n"
            "  --ppm FILE    записать картинку; --counts FILE — записать счётчики итераций (uint16)\n";
}

int main(int argc, char** argv) {
    try {
        int viewIndex = 1;
        int threads = 1;
        int repeats = 1;
        int maxIterations = 256;
        string mode = "serial";
        string ppmFile, countsFile;
        bool verify = false;
        Renderer renderer;

        for (int i = 1; i < argc; ++i) {
            const string arg = argv[i];
            auto value = [&]() -> string {
                if (i + 1 >= argc) throw runtime_error("Missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--view") viewIndex = stoi(value());
            else if (arg == "--width") renderer.width = stoi(value());
            else if (arg == "--height") renderer.height = stoi(value());
            else if (arg == "--max-iter") maxIterations = stoi(value());
            else if (arg == "--mode") mode = value();
            else if (arg == "--threads") threads = stoi(value());
            else if (arg == "--repeats") repeats = stoi(value());
            else if (arg == "--verify") verify = true;
            else if (arg == "--ppm") ppmFile = value();
            else if (arg == "--counts") countsFile = value();
            else if (arg == "--help" || arg == "-h") { usage(); return 0; }
            else throw runtime_error("Unknown argument: " + arg);
        }
        if (renderer.width < 1 || renderer.height < 1 || maxIterations < 1 || maxIterations > 65535)
            throw runtime_error("Invalid image or iteration parameters");
        if (threads < 1 || repeats < 1) throw runtime_error("Threads and repeats must be positive");
        if (mode == "serial") threads = 1;

        renderer.view = makeView(viewIndex);
        renderer.maxIterations = maxIterations;
        renderer.resize();

        const RunResult result = render(renderer, mode, threads, repeats);

        string verifyText;
        if (verify) {
            Renderer reference;
            reference.view = renderer.view;
            reference.width = renderer.width;
            reference.height = renderer.height;
            reference.maxIterations = renderer.maxIterations;
            reference.resize();
            render(reference, "serial", 1, 1);
            size_t mismatches = 0;
            int maxDiff = 0;
            for (size_t i = 0; i < renderer.counts.size(); ++i) {
                const int diff = abs(static_cast<int>(renderer.counts[i]) -
                                     static_cast<int>(reference.counts[i]));
                if (diff != 0) ++mismatches;
                maxDiff = max(maxDiff, diff);
            }
            verifyText = " verify=" + string(mismatches == 0 ? "PASS" : "FAIL") +
                         " mismatched_pixels=" + to_string(mismatches) +
                         " max_diff=" + to_string(maxDiff);
        }

        if (!ppmFile.empty()) writePpm(renderer, ppmFile);
        if (!countsFile.empty()) writeCounts(renderer, countsFile);

        const double megapixels = static_cast<double>(renderer.width) * renderer.height / 1e6;
        const double megapixelsPerSecond = megapixels / result.seconds;

        cout << fixed << setprecision(6) << "view=" << viewIndex << " width=" << renderer.width
             << " height=" << renderer.height << " max_iter=" << maxIterations << " mode=" << mode
             << " threads=" << threads << " repeats=" << repeats
             << " time_sec=" << result.seconds << " time_min_sec=" << result.seconds
             << " work_iters=" << result.iterations << " mpix_per_sec=" << setprecision(3)
             << megapixelsPerSecond << " time_per_pixel_ns=" << setprecision(3)
             << (megapixels > 0 ? result.seconds * 1e9 / (megapixels * 1e6) : 0.0)
             << " thread_times=\"";
        for (size_t i = 0; i < result.threadTimes.size(); ++i) {
            if (i) cout << ',';
            cout << setprecision(4) << result.threadTimes[i];
        }
        cout << "\" thread_iters=\"";
        for (size_t i = 0; i < result.threadIterations.size(); ++i) {
            if (i) cout << ',';
            cout << result.threadIterations[i];
        }
        cout << "\"" << verifyText << '\n';
    } catch (const exception& error) {
        cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
