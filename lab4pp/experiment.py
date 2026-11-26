import argparse
import csv
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser(description="CUDA matrix multiplication benchmarks")
p.add_argument("--exe", default="matrix_cuda.exe" if os.name == "nt" else "./matrix_cuda")
p.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
p.add_argument("--configs", nargs="+", default=["tiled:8", "tiled:16", "tiled:32", "naive:16", "naive:32"],
               help="конфигурации вида kernel:block, например tiled:16")
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--cpu-exe", default=None,
               help="необязательный бинарник последовательной версии (л/р №1) для расчёта ускорения")
p.add_argument("--cpu-repeats", type=int, default=3)
p.add_argument("--csv", default="benchmark.csv")
p.add_argument("--log", default="results.txt")
p.add_argument("--charts-only", action="store_true",
               help="не измерять, только перерисовать графики по CSV")
args = p.parse_args()
if args.repeats < 1 or args.warmup < 0 or args.cpu_repeats < 1 or any(n < 1 for n in args.sizes):
    p.error("Sizes, repeats, warmup and cpu-repeats must be positive")

CONFIGS = []
for item in args.configs:
    kernel, _, block = item.partition(":")
    CONFIGS.append((kernel, int(block)))

CSV_PATH = Path(args.csv)
LOG_PATH = Path(args.log)


def label(kernel, block):
    return f"{kernel} {block}x{block}"


def plot_charts(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: python -m pip install -r requirements.txt. "
              "Charts were not created.")
        return False

    sizes = sorted({r["N"] for r in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for i, (kernel, block) in enumerate(CONFIGS):
        picked = [r for r in rows if r["kernel"] == kernel and r["block"] == block]
        if not picked:
            continue
        left.plot([r["N"] for r in picked], [r["time_min_sec"] for r in picked], marker="o",
                  color=colors[i % len(colors)], label=label(kernel, block))
        right.plot([r["N"] for r in picked], [r["gflops"] for r in picked], marker="o",
                   color=colors[i % len(colors)], label=label(kernel, block))
    left.set_title("Время ядра умножения")
    left.set_xlabel("Размер матрицы N")
    left.set_ylabel("Время, с (минимум из замеров)")
    left.set_yscale("log")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=8)
    right.set_title("Производительность ядра (double)")
    right.set_xlabel("Размер матрицы N")
    right.set_ylabel("Гфлоп/с (2N³ / T)")
    right.grid(True, alpha=0.3)
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("performance.png", dpi=160)
    plt.close(fig)

    if any(r.get("speedup_cpu") for r in rows):
        plt.figure(figsize=(8, 4.8))
        for i, (kernel, block) in enumerate(CONFIGS):
            picked = [r for r in rows if r["kernel"] == kernel and r["block"] == block]
            if not picked:
                continue
            plt.plot([r["N"] for r in picked], [r["speedup_cpu"] for r in picked], marker="o",
                     color=colors[i % len(colors)], label=label(kernel, block))
        plt.title("Ускорение относительно последовательной версии (л/р №1)")
        plt.xlabel("Размер матрицы N")
        plt.ylabel("S = T_cpu / T_gpu")
        plt.yscale("log")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("speedup.png", dpi=160)
        plt.close()
        print("Saved performance.png and speedup.png")
    else:
        print("Saved performance.png")
    return True


def read_csv(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "N": int(row["N"]),
                "kernel": row["kernel"],
                "block": int(row["block"]),
                "time_min_sec": float(row["time_min_sec"]),
                "time_mean_sec": float(row["time_mean_sec"]),
                "h2d_sec": float(row["h2d_sec"]),
                "d2h_sec": float(row["d2h_sec"]),
                "gflops": float(row["gflops"]),
                "speedup_cpu": float(row["speedup_cpu"]) if row.get("speedup_cpu") else None,
                "verified": row["verified"],
            })
    return rows


if args.charts_only:
    rows = read_csv(CSV_PATH)
    print(f"Loaded {len(rows)} points from {CSV_PATH}")
    plot_charts(rows)
    raise SystemExit(0)

rows = []
log = ["# Протокол экспериментов (л/р №4, CUDA)",
       f"# Дата: {datetime.now():%Y-%m-%d %H:%M}",
       f"# Система: {platform.platform()}",
       f"# Размеры: {args.sizes}",
       f"# Конфигурации (ядро, блок): {CONFIGS}",
       f"# Замеров на точку: {args.repeats}, прогревов: {args.warmup}",
       "#" + "-" * 70]

for n in args.sizes:
    out = subprocess.run([sys.executable, "generate.py", str(n)],
                         check=True, capture_output=True, text=True)
    log.append(f"$ python generate.py {n}\n{out.stdout.strip()}")

    cpu_time = None
    if args.cpu_exe:
        runs = []
        for _ in range(args.cpu_repeats):
            proc = subprocess.run([args.cpu_exe], check=True, capture_output=True, text=True)
            runs.append(float(re.search(r"time_sec=([0-9.]+)", proc.stdout).group(1)))
        cpu_time = min(runs)
        log.append(f"$ {Path(args.cpu_exe).name}  (л/р №1, последовательно)\n"
                   + "\n".join(f"{v:.6f}" for v in runs))
        print(f"N={n}, cpu: min {cpu_time:.6f} s", flush=True)

    for kernel, block in CONFIGS:
        command = [args.exe, "--kernel", kernel, "--block", str(block),
                   "--repeats", str(args.repeats), "--warmup", str(args.warmup)]
        proc = subprocess.run(command, check=True, capture_output=True, text=True)
        log.append(f"$ {' '.join(command)}\n{proc.stdout.strip()}")
        fastest = float(re.search(r"time_min_sec=([0-9.]+)", proc.stdout).group(1))
        mean = float(re.search(r"time_sec=([0-9.]+)", proc.stdout).group(1))
        output = dict(re.findall(r"(\w+)=(\"[^\"]*\"|\S+)", proc.stdout))
        h2d = float(output["h2d_sec"])
        d2h = float(output["d2h_sec"])
        print(f"N={n}, {kernel} {block}x{block}: min {fastest * 1e3:.3f} ms, "
              f"mean {mean * 1e3:.3f} ms", flush=True)

        check = subprocess.run([sys.executable, "check.py"], check=True,
                               capture_output=True, text=True)
        log.append(f"$ python check.py\n{check.stdout.strip()}")
        verified = "PASS" if "PASS" in check.stdout else "FAIL"

        rows.append({
            "N": n,
            "kernel": kernel,
            "block": block,
            "time_min_sec": round(fastest, 7),
            "time_mean_sec": round(mean, 7),
            "h2d_sec": round(h2d, 7),
            "d2h_sec": round(d2h, 7),
            "operations": 2 * n ** 3,
            "memory_bytes": 3 * n * n * 8,
            "gflops": round(2 * n ** 3 / fastest / 1e9, 3),
            "speedup_cpu": round(cpu_time / fastest, 2) if cpu_time else "",
            "verified": verified,
            "repeats": args.repeats,
        })

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {CSV_PATH}")

LOG_PATH.write_text("\n".join(log) + "\n", encoding="utf-8")
print(f"Saved {LOG_PATH}")

charts = plot_charts(rows)


# --- обновление README.md -------------------------------------------------
def table(key, header, formatter, scale=1.0):
    configs = [(k, b) for k, b in CONFIGS
               if any(r["kernel"] == k and r["block"] == b for r in rows)]
    head = ("| N | " + " | ".join(label(k, b) for k, b in configs) + " |\n|"
            + "---:|" + "---:|" * len(configs) + "\n")
    lines = []
    for n in args.sizes:
        picked = {(r["kernel"], r["block"]): r for r in rows if r["N"] == n}
        lines.append(f"| {n} | " + " | ".join(formatter(picked[(k, b)][key] * scale)
                                              for k, b in configs) + " |\n")
    return header + head + "".join(lines)


configs = [(k, b) for k, b in CONFIGS if any(r["kernel"] == k and r["block"] == b for r in rows)]
verified = "PASS" if all(r["verified"] == "PASS" for r in rows) else "FAIL"

transfer_head = ("| N | H2D, мс | D2H, мс |\n|---:|---:|---:|\n")
transfer_rows = "".join(
    f"| {n} | {max(r['h2d_sec'] for r in rows if r['N'] == n) * 1e3:.3f} | "
    f"{max(r['d2h_sec'] for r in rows if r['N'] == n) * 1e3:.3f} |\n" for n in args.sizes)

images = ""
if charts:
    images = "\n![Время и производительность](performance.png)\n"
    if any(r["speedup_cpu"] for r in rows):
        images += "\n![Ускорение относительно CPU](speedup.png)\n"

speedup_table = ""
if args.cpu_exe:
    speedup_table = "\n" + table("speedup_cpu", "**Ускорение относительно последовательной версии "
                                                "(л/р №1), S = T_cpu / T_gpu:**\n\n",
                                 lambda v: f"{v:.1f}×")

block = (
    f"\n**Условия измерений:** {platform.platform()}; Python {platform.python_version()}; "
    f"замеров на точку: {args.repeats} (+{args.warmup} прогрев(ов)), в таблицах — минимум; "
    f"время ядра измеряется событиями CUDA и не включает копирование данных в GPU и обратно. "
    f"Верификация NumPy: **{verified}** для всех {len(rows)} точек.\n\n"
    + "**Исходные данные для графиков** — таблицы ниже (по ним построены графики), "
      "полные данные с временем копирования лежат в `benchmark.csv`.\n\n"
    + table("time_min_sec", "**Время ядра, мс:**\n\n", lambda v: f"{v:.3f}", scale=1e3)
    + "\n" + table("gflops", "**Производительность, Гфлоп/с (2N³ / T):**\n\n", lambda v: f"{v:.1f}")
    + speedup_table
    + "\n**Время копирования данных (наибольшее по конфигурациям):**\n\n" + transfer_head + transfer_rows
    + images
)

report = Path("README.md")
if report.exists():
    text = report.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    report.write_text(text.split(start)[0] + start + block + end + text.split(end)[1],
                      encoding="utf-8")
    print("Updated README.md")
