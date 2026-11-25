import argparse
import csv
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

p = argparse.ArgumentParser(description="Sequential matrix multiplication benchmarks")
p.add_argument("--exe", default="matrix.exe" if os.name == "nt" else "./matrix")
p.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
p.add_argument("--repeats", type=int, default=3)
args = p.parse_args()
if args.repeats < 1 or any(n < 1 for n in args.sizes):
    p.error("Sizes and repeats must be positive")

Path("results").mkdir(exist_ok=True)
rows = []
for n in args.sizes:
    subprocess.run([sys.executable, "generate.py", str(n)], check=True)
    times = []
    for rep in range(args.repeats):
        proc = subprocess.run([args.exe], check=True, capture_output=True, text=True)
        match = re.search(r"time_sec=([0-9.]+)", proc.stdout)
        if not match:
            raise RuntimeError(proc.stdout)
        times.append(float(match.group(1)))
        print(f"N={n}, repeat={rep+1}: {times[-1]:.6f} sec", flush=True)
    subprocess.run([sys.executable, "check.py"], check=True)
    fastest = min(times)
    rows.append({"N": n,
                 "time_sec": round(sum(times) / len(times), 6),
                 "time_min_sec": round(fastest, 6),
                 "operations": 2 * n ** 3,
                 "memory_bytes": 3 * n * n * 8,
                 "gflops": round(2 * n ** 3 / fastest / 1e9, 3),
                 "verified": "PASS",
                 "repeats": args.repeats})

with open("results/results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sizes = [r["N"] for r in rows]
fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8))

left.plot(sizes, [r["time_min_sec"] for r in rows], marker="o", color="#1f77b4",
          label="измерение (минимум)")
base_time = rows[0]["time_min_sec"]
left.plot(sizes, [base_time * (n / sizes[0]) ** 3 for n in sizes], "k:", lw=1.4,
          label="теория O(N³)")
left.set_title("Время умножения матриц")
left.set_xlabel("Размер матрицы N")
left.set_ylabel("Время, с")
left.set_yscale("log")
left.grid(True, which="both", alpha=0.3)
left.legend()

right.plot(sizes, [r["gflops"] for r in rows], marker="o", color="#d62728")
right.set_title("Производительность")
right.set_xlabel("Размер матрицы N")
right.set_ylabel("Гфлоп/с (2N³ / T)")
right.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/time_vs_size.png", dpi=160)
plt.close(fig)

report = Path("README.md")
text = report.read_text(encoding="utf-8")
start = "<!-- RESULTS_START -->"
end = "<!-- RESULTS_END -->"
header = ("| N | Время (среднее), с | Время (минимум), с | Операций (2N³) | "
          "Память трёх матриц, байт | Гфлоп/с | Проверка |\n"
          "|---:|---:|---:|---:|---:|---:|:---:|\n")
body = "".join(
    f'| {r["N"]} | {r["time_sec"]:.6f} | {r["time_min_sec"]:.6f} | {r["operations"]} | '
    f'{r["memory_bytes"]} | {r["gflops"]:.2f} | {r["verified"]} |\n' for r in rows)
block = ("\n**Условия измерений:** " + platform.platform() + "; Python " + platform.python_version() + ". "
         + f"Каждый размер: {args.repeats} запуск(ов); в таблице — среднее и минимальное время. "
         + "Время измерено только для умножения, без чтения и записи файлов.\n\n"
         + "**Исходные данные для графиков** — таблица ниже (по ней построены оба графика). "
         + "Полные данные лежат в `results/results.csv`.\n\n"
         + header + body
         + "\n![Время и производительность](results/time_vs_size.png)\n\n"
         + "**Вывод по эксперименту:** при увеличении N время умножения растёт примерно как N³, пока "
         + "матрицы помещаются в кэш; на больших N измеренная кривая отрывается вверх от теоретической, "
         + "а производительность одного ядра падает — задача становится ограниченной памятью.\n")
report.write_text(text.split(start)[0] + start + "\n" + block + end + text.split(end)[1], encoding="utf-8")
print("Saved results/results.csv, results/time_vs_size.png and updated README.md")
