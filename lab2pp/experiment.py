import argparse
import csv
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser(description="OpenMP matrix multiplication benchmarks")
p.add_argument("--exe", default="matrix.exe" if os.name == "nt" else "./matrix")
p.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
p.add_argument("--threads", type=int, nargs="+", default=None,
               help="thread counts, default: 1 2 4 8 and all available processors")
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--serial-exe", default=None,
               help="optional binary of lab1 (sequential version); reads A.txt/B.txt from cwd")
p.add_argument("--csv", default="benchmark.csv")
p.add_argument("--log", default="results.txt")
p.add_argument("--charts-only", action="store_true",
               help="do not measure anything, only redraw charts from the CSV")
args = p.parse_args()
if args.repeats < 1 or args.warmup < 0 or any(n < 1 for n in args.sizes):
    p.error("Sizes, repeats and warmup must be positive")

CSV_PATH = Path(args.csv)
LOG_PATH = Path(args.log)


def plot_charts(rows):
    """Графики: performance.png (время и ускорение) и efficiency.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: python -m pip install -r requirements.txt. "
              "Charts were not created.")
        return False

    sizes = sorted({r["N"] for r in rows})
    threads = sorted({r["threads"] for r in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for i, t in enumerate(threads):
        picked = [r for r in rows if r["threads"] == t]
        left.plot([r["N"] for r in picked], [r["time_min_sec"] for r in picked], marker="o",
                  color=colors[i % len(colors)], label=f"{t} поток(ов)")
    # теоретическая кривая O(N^3), привязанная к первой точке однопоточной серии
    base = min(sizes)
    t_base = min(r["time_min_sec"] for r in rows if r["N"] == base and r["threads"] == 1)
    left.plot(sizes, [t_base * (n / base) ** 3 for n in sizes], "k:", lw=1.4,
              label="теория O(N³)")
    left.set_title("Время умножения")
    left.set_xlabel("Размер матрицы N")
    left.set_ylabel("Время, с (минимум из замеров)")
    left.set_yscale("log")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=8)

    for i, n in enumerate(sizes):
        picked = [r for r in rows if r["N"] == n]
        right.plot([r["threads"] for r in picked], [r["speedup"] for r in picked],
                   marker="o", color=colors[i % len(colors)], label=f"N={n}")
    right.plot(threads, threads, "k--", alpha=0.6, label="линейное ускорение")
    right.set_title("Ускорение S(p) = T(1)/T(p)")
    right.set_xlabel("Число потоков p")
    right.set_ylabel("Ускорение")
    right.grid(True, alpha=0.3)
    right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("performance.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(8, 4.8))
    for i, n in enumerate(sizes):
        picked = [r for r in rows if r["N"] == n]
        plt.plot([r["threads"] for r in picked], [100 * r["efficiency"] for r in picked],
                 marker="o", color=colors[i % len(colors)], label=f"N={n}")
    plt.axhline(100, color="k", linestyle="--", alpha=0.6, label="100 %")
    plt.title("Эффективность распараллеливания E(p) = S(p)/p")
    plt.xlabel("Число потоков p")
    plt.ylabel("Эффективность, %")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("efficiency.png", dpi=160)
    plt.close()
    print("Saved performance.png and efficiency.png")
    return True


def read_csv(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "N": int(row["N"]),
                "threads": int(row["threads"]),
                "time_min_sec": float(row["time_min_sec"]),
                "time_mean_sec": float(row["time_mean_sec"]),
                "gflops": float(row["gflops"]),
                "speedup": float(row["speedup"]),
                "efficiency": float(row["efficiency"]),
                "verified": row["verified"],
            })
    return rows


if args.charts_only:
    rows = read_csv(CSV_PATH)
    print(f"Loaded {len(rows)} points from {CSV_PATH}")
    plot_charts(rows)
    raise SystemExit(0)

# Таблицы и графики строятся по минимальному времени из замеров:
# минимум меньше подвержен влиянию посторонней нагрузки, чем среднее.
available = os.cpu_count() or 1
threads = args.threads or sorted({1, 2, 4, 8, available})
threads = sorted({t for t in threads if 1 <= t <= available}) or [1]

rows = []
log = ["# Протокол экспериментов (л/р №2, OpenMP)",
       f"# Дата: {datetime.now():%Y-%m-%d %H:%M}",
       f"# Система: {platform.platform()}; логических процессоров: {available}",
       f"# Размеры: {args.sizes}",
       f"# Число потоков: {threads}",
       f"# Замеров на точку: {args.repeats}, прогревов: {args.warmup}",
       "#" + "-" * 70]

for n in args.sizes:
    out = subprocess.run([sys.executable, "generate.py", str(n)],
                         check=True, capture_output=True, text=True)
    log.append(f"$ python generate.py {n}\n{out.stdout.strip()}")

    # Контрольный замер программы л/р №1 на тех же данных (если задан --serial-exe).
    lab1_time = None
    if args.serial_exe:
        runs = []
        for _ in range(args.repeats):
            proc = subprocess.run([args.serial_exe], check=True, capture_output=True, text=True)
            runs.append(float(re.search(r"time_sec=([0-9.]+)", proc.stdout).group(1)))
        lab1_time = min(runs)
        log.append(f"$ {Path(args.serial_exe).name}  (л/р №1)\n"
                   + "\n".join(f"{v:.6f}" for v in runs))
        print(f"N={n}, lab1: min {lab1_time:.6f} s", flush=True)

    times = {}
    for t in threads:
        command = [args.exe, "--threads", str(t), "--repeats", str(args.repeats),
                   "--warmup", str(args.warmup)]
        proc = subprocess.run(command, check=True, capture_output=True, text=True)
        log.append(f"$ {' '.join(command)}\n{proc.stdout.strip()}")
        fastest = float(re.search(r"time_min_sec=([0-9.]+)", proc.stdout).group(1))
        mean = float(re.search(r"time_sec=([0-9.]+)", proc.stdout).group(1))
        times[t] = fastest
        print(f"N={n}, p={t}: min {fastest:.6f} s, mean {mean:.6f} s", flush=True)

        check = subprocess.run([sys.executable, "check.py"], check=True,
                               capture_output=True, text=True)
        log.append(f"$ python check.py\n{check.stdout.strip()}")
        verified = "PASS" if "PASS" in check.stdout else "FAIL"

        baseline = times[1]
        rows.append({
            "N": n,
            "threads": t,
            "time_min_sec": round(fastest, 6),
            "time_mean_sec": round(mean, 6),
            "operations": 2 * n ** 3,
            "memory_bytes": 3 * n * n * 8,
            "gflops": round(2 * n ** 3 / fastest / 1e9, 3),
            "speedup": round(baseline / fastest, 3),
            "efficiency": round(baseline / fastest / t, 3),
            "verified": verified,
            "repeats": args.repeats,
            "time_lab1_sec": round(lab1_time, 6) if lab1_time else "",
        })

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {CSV_PATH}")

LOG_PATH.write_text("\n".join(log) + "\n", encoding="utf-8")
print(f"Saved {LOG_PATH}")

charts = plot_charts(rows)

# --- обновление README.md ------------------------------------------------
def table(key, header, formatter):
    head = ("| N | " + " | ".join(f"p={t}" for t in threads) + " |\n|"
            + "---:|" + "---:|" * len(threads) + "\n")
    lines = []
    for n in args.sizes:
        picked = {r["threads"]: r for r in rows if r["N"] == n}
        lines.append(f"| {n} | " + " | ".join(formatter(picked[t][key]) for t in threads) + " |\n")
    return header + head + "".join(lines)


verified = "PASS" if all(r["verified"] == "PASS" for r in rows) else "FAIL"
images = ""
if charts:
    images = ("\n![Время и ускорение](performance.png)\n\n"
              "![Эффективность](efficiency.png)\n")

lab1_table = ""
if args.serial_exe:
    lab1_head = ("\n**Контрольный замер: программа л/р №1 на тех же матрицах** "
                 "(тот же компилятор и флаги, та же машина):\n\n"
                 "| N | л/р №1 (последовательно), с | л/р №2, p=1, с | Отношение |\n"
                 "|---:|---:|---:|---:|\n")
    lab1_rows = ""
    for n in args.sizes:
        picked = [r for r in rows if r["N"] == n]
        old = picked[0]["time_lab1_sec"]
        now = picked[0]["time_min_sec"]
        lab1_rows += f"| {n} | {old:.6f} | {now:.6f} | {old / now:.2f} |\n"
    lab1_table = lab1_head + lab1_rows

block = (
    f"\n**Условия измерений:** {platform.platform()}; логических процессоров: "
    f"{available}; Python {platform.python_version()}; число замеров на точку: "
    f"{args.repeats} (+{args.warmup} прогрев(ов)); в таблицах — минимальное время из замеров. "
    f"Время измеряется только для умножения, без чтения и записи файлов. "
    f"Верификация NumPy: **{verified}** для всех {len(rows)} точек.\n\n"
    + "**Исходные данные для графиков** — таблицы ниже: по времени построена левая панель "
    "`performance.png`, по ускорению — правая, по эффективности — `efficiency.png`. "
    "Полные данные с временем min/mean лежат в `benchmark.csv`, протокол прогонов — в `results.txt`.\n\n"
    + table("time_min_sec", "**Время умножения T(p), с:**\n\n", lambda v: f"{v:.6f}")
    + "\n" + table("speedup", "**Ускорение S(p) = T(1)/T(p):**\n\n", lambda v: f"{v:.2f}")
    + "\n" + table("efficiency", "**Эффективность E(p) = S(p)/p:**\n\n", lambda v: f"{100 * v:.1f} %")
    + "\n" + table("gflops", "**Производительность, Гфлоп/с (2N³ / T(p)):**\n\n", lambda v: f"{v:.2f}")
    + lab1_table
    + images
)

report = Path("README.md")
if report.exists():
    text = report.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    report.write_text(text.split(start)[0] + start + block + end + text.split(end)[1],
                      encoding="utf-8")
    print("Updated README.md")
