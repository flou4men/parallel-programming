# Параллельное программирование

**Левин Григорий**, группа 6213-100503D

Лабораторные работы по курсу. Сквозная задача — умножение квадратных матриц `C = A × B`: один и тот
же алгоритм (три вложенных цикла в порядке i–k–j, без сторонних матричных библиотек) в каждой работе
реализуется заново и исследуется с новой технологией параллелизма.

## Лабораторные работы

| № | Работа | Технология | Отчёт |
|---:|---|---|---|
| 1 | Умножение квадратных матриц, последовательная версия | C++17 | [lab1pp/README.md](lab1pp/README.md) |
| 2 | То же умножение с распараллеливанием | OpenMP | [lab2pp/README.md](lab2pp/README.md) |
| 3 | То же умножение в MPI на суперкомпьютере | MPI, кластер «Сергей Королёв» | [lab3pp/README.md](lab3pp/README.md) |
| 4 | То же умножение на видеокарте | CUDA | [lab4pp/README.md](lab4pp/README.md) |
| 5 | Фрактал Мандельброта на потоках (программа 1 из Stanford CS149 asst1) | ``std::thread`` | [lab5pp/README.md](lab5pp/README.md) |

**Отчёт по каждой работе — это её `README.md`, скачивать ничего не нужно.** В отчёте: задание и цель,
описание реализации, методика экспериментов, таблицы с исходными данными, графики, проверка
результатов через NumPy и выводы. Рядом с отчётом в папке лежат исходный код, скрипты экспериментов,
`benchmark.csv` с результатами замеров и графики в PNG.

## Аппаратное и программное обеспечение

Все работы выполнены на одном ноутбуке; л/р №3 — на суперкомпьютере Самарского университета.

**Ноутбук (л/р №1, №2, №4, №5):**

| Компонент | Значение |
|---|---|
| Процессор | 13th Gen Intel(R) Core(TM) i5-13500H, 12 физических ядер (4 P-ядра + 8 E-ядер), 16 логических потоков, 3.19 ГГц |
| Оперативная память | 16 ГБ |
| Видеокарта | NVIDIA GeForce RTX 4050 Laptop GPU, 6 ГБ, compute capability 8.9 (Ada Lovelace), 20 SM, кэш L2 24 МБ |
| Операционная система | Windows 11, 26H2, сборка 26300, 64 бита |
| Компилятор C++ | MSVC 19.51.36260 (Visual Studio 18), флаги `/O2 /std:c++17` (для л/р №2 — плюс `/openmp`) |
| CUDA | Toolkit 13.5 (`nvcc` V13.5.37), runtime 13.4, драйвер 616.92, флаги `-O2 -arch=sm_89` |
| CMake / git | CMake 4.3.1, git 2.56 |
| Python | 3.12.14, NumPy 2.3.5 (проверка результатов), matplotlib 3.11.2 (графики) |

**Кластер «Сергей Королёв» (л/р №3):**

| Параметр | Значение |
|---|---|
| Конфигурация | IBM BladeCenter, 68 серверов, 1168 вычислительных ядер, пиковая производительность 40 Тфлоп/с |
| Процессоры узлов | 2× Intel Xeon E5-2665 (8 ядер, 2.40 ГГц), 2× Xeon E5-2680v2 (10 ядер), Xeon Gold 6230 / Silver 4316 (20 ядер) |
| Оперативная память | от 32 до 512 ГБ на узел |
| Межсоединение | InfiniBand QDR 40 Гбит/с |
| Операционная система | CentOS 7.9 |
| MPI / система заданий | Intel MPI 4 (`module load impi/4`), PBS/Torque + Moab (`qsub`, `qstat`) |
| Компилятор на кластере | `mpicxx` (Intel MPI), флаги `-O2 -std=c++17` |

## Требования

- Компилятор с поддержкой C++17: MSVC 2019+ или g++ 9+;
- OpenMP для л/р №2 (`/openmp` в MSVC, `-fopenmp` в g++), MPI для л/р №3 (на кластере — Intel MPI),
  CUDA Toolkit 12.4+ для л/р №4;
- CMake 3.20+ (для л/р №2 и №4; есть и сборка без CMake — `build.ps1` / `Makefile`);
- Python 3.9+ и пакеты из `requirements.txt` (NumPy — проверка, matplotlib — графики).

## Быстрый старт: л/р №1 (последовательная версия)

```powershell
cd lab1pp
cl /EHsc /O2 /std:c++17 main.cpp /Fe:matrix.exe
py -m pip install -r requirements.txt
py generate.py 800
.\matrix.exe
py check.py
py experiment.py --exe .\matrix.exe --repeats 5
```

## Быстрый старт: л/р №2 (OpenMP)

```powershell
cd lab2pp
cmake -S . -B build
cmake --build build --config Release
py -m pip install -r requirements.txt
py generate.py 800
.\build\Release\matrix.exe --threads 8 --repeats 5
py check.py
py experiment.py --exe .\build\Release\matrix.exe --repeats 5
```

## Быстрый старт: л/р №3 (MPI, суперкомпьютер «Сергей Королёв»)

```bash
# на кластере sk.ssau.ru (после входа по SSH/VPN университета)
cd lab3pp
qsub lab3.pbs        # одно задание на всю серию: сборка + все эксперименты
qstat -a             # Q → R → C, результат в lab3_matrix_mpi.o<номер>
```

```powershell
# на ноутбуке после запуска: перенести results.txt и выполнить
py experiment.py --from-log
```

## Быстрый старт: л/р №4 (CUDA)

```powershell
cd lab4pp
cmake -S . -B build
cmake --build build --config Release
py -m pip install -r requirements.txt
py generate.py 800
.\build\Release\matrix_cuda.exe --kernel tiled --block 16 --repeats 5
py check.py
py experiment.py --exe .\build\Release\matrix_cuda.exe --cpu-exe ..\lab1pp\matrix.exe --repeats 5
```

Архитектура GPU задаётся в `CMakeLists.txt` (`CMAKE_CUDA_ARCHITECTURES`, по умолчанию 89).

## Быстрый старт: л/р №5 (Mandelbrot на потоках)

```powershell
cd lab5pp
cl /EHsc /O2 /std:c++17 main.cpp /Fe:mandelbrot.exe      # или .\build.ps1 / make
py -m pip install -r requirements.txt
.\mandelbrot.exe --view 2 --mode interleaved --threads 8 --repeats 5 --verify
py experiment.py --repeats 5
```
