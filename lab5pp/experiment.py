import argparse
import csv
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser(description="Threaded Mandelbrot benchmarks (CS149 asst1, program 1)")
p.add_argument("--exe", default="mandelbrot.exe" if os.name == "nt" else "./mandelbrot")
p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
p.add_argument("--modes", nargs="+", default=["block", "interleaved"])
p.add_argument("--dynamic-threads", type=int, nargs="+", default=[8, 16])
p.add_argument("--width", type=int, default=1600)
p.add_argument("--height", type=int, default=1200)
p.add_argument("--max-iter", type=int, default=256)
p.add_argument("--sizes", nargs="+", default=["400x300", "800x600", "1200x900", "1600x1200", "2400x1800"])
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--no-charts", action="store_true")
p.add_argument("--no-readme", action="store_true")
p.add_argument("--csv", default="benchmark.csv")
p.add_argument("--log", default="results.txt")
args = p.parse_args()

CSV_PATH = Path(args.csv)
LOG_PATH = Path(args.log)
log = []


def run(view, mode, threads, width, height, repeats, extra=()):
    command = [args.exe, "--view", str(view), "--mode", mode, "--threads", str(threads),
               "--width", str(width), "--height", str(height), "--max-iter", str(args.max_iter),
               "--repeats", str(repeats), *extra]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    log.append(f"$ {' '.join(command)}\n{proc.stdout.strip()}")
    fields = dict(re.findall(r"(\w+)=(\"[^\"]*\"|\S+)", proc.stdout))
    return {
        "view": view,
        "mode": mode,
        "threads": int(fields["threads"]),
        "width": width,
        "height": height,
        "time_min_sec": float(fields["time_min_sec"]),
        "work_iters": int(fields["work_iters"]),
        "mpix_per_sec": float(fields["mpix_per_sec"]),
        "thread_times": fields["thread_times"].strip('"'),
        "thread_iters": fields["thread_iters"].strip('"'),
        "verify": "PASS" if "verify=PASS" in proc.stdout else ("FAIL" if "verify=FAIL" in proc.stdout else "SKIP"),
    }


print("== серия 1: ускорение от числа потоков ==", flush=True)
rows = []
for view in (1, 2):
    baseline = run(view, "serial", 1, args.width, args.height, args.repeats)
    counts = f"counts_view{view}.bin"
    run(view, "serial", 1, args.width, args.height, 1, ("--counts", counts))
    check = subprocess.run([sys.executable, "check.py", "--counts", counts, "--width", str(args.width),
                            "--height", str(args.height), "--view", str(view),
                            "--max-iter", str(args.max_iter)], capture_output=True, text=True)
    log.append(f"$ python check.py (view {view})\n{check.stdout.strip()}")
    numpy_ok = "PASS" in check.stdout
    print(f"view={view}: serial {baseline['time_min_sec']:.4f} s, numpy {'PASS' if numpy_ok else 'FAIL'}",
          flush=True)

    for mode in args.modes:
        for threads in args.threads:
            extra = ("--verify",) if (mode == "interleaved" and threads == 8) else ()
            item = run(view, mode, threads, args.width, args.height, args.repeats, extra)
            item["speedup"] = baseline["time_min_sec"] / item["time_min_sec"]
            item["verified"] = "PASS" if (numpy_ok or item["verify"] == "PASS") else "FAIL"
            item["series"] = "threads"
            rows.append(item)
            print(f"  view={view} {mode:11s} p={threads:2d}: {item['time_min_sec']:.4f} s, "
                  f"speedup {item['speedup']:.2f}x", flush=True)

    for threads in args.dynamic_threads:
        item = run(view, "dynamic", threads, args.width, args.height, args.repeats)
        item["speedup"] = baseline["time_min_sec"] / item["time_min_sec"]
        item["verified"] = "PASS" if numpy_ok else "FAIL"
        item["series"] = "dynamic"
        rows.append(item)
        print(f"  view={view} dynamic     p={threads:2d}: {item['time_min_sec']:.4f} s, "
              f"speedup {item['speedup']:.2f}x", flush=True)

print("== пересчёт ускорения от стабильного однопоточного эталона ==", flush=True)
for view in (1, 2):
    single = [r["time_min_sec"] for r in rows if r["view"] == view and r["threads"] == 1]
    baseline = min(single)
    for r in rows:
        if r["view"] == view:
            r["speedup"] = baseline / r["time_min_sec"]
    print(f"  view {view}: однопоточный эталон {baseline:.4f} с", flush=True)

print("== серия 2: время от размера изображения ==", flush=True)
size_rows = []
for size in args.sizes:
    width, height = (int(v) for v in size.lower().split("x"))
    serial = run(1, "serial", 1, width, height, 2)
    threaded = run(1, "interleaved", 8, width, height, args.repeats)
    threaded["speedup"] = serial["time_min_sec"] / threaded["time_min_sec"]
    threaded["verified"] = "SKIP"
    threaded["series"] = "size"
    size_rows.append({"size": size, "width": width, "height": height,
                      "serial_sec": serial["time_min_sec"], "threaded_sec": threaded["time_min_sec"],
                      "speedup": threaded["speedup"], "mpix": width * height / 1e6})
    rows.append(threaded)
    print(f"  {size}: serial {serial['time_min_sec']:.3f} s, 8 потоков "
          f"{threaded['time_min_sec']:.3f} s, {threaded['speedup']:.2f}x", flush=True)

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["series", "view", "mode", "threads", "width", "height",
                                           "time_min_sec", "work_iters", "mpix_per_sec",
                                           "speedup", "thread_times", "thread_iters",
                                           "verified", "verify"])
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
print(f"Saved {CSV_PATH}")


# ---------------------------------------------------------------- графики ---
def charts():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed — графики не построены")
        return False

    colors = {"block": "#d62728", "interleaved": "#1f77b4", "dynamic": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, view in zip(axes, (1, 2)):
        for mode in args.modes:
            picked = [r for r in rows if r["series"] == "threads" and r["view"] == view and r["mode"] == mode]
            ax.plot([r["threads"] for r in picked], [r["speedup"] for r in picked], marker="o",
                    color=colors.get(mode), label=mode)
        picked = [r for r in rows if r["series"] == "dynamic" and r["view"] == view]
        if picked:
            ax.plot([r["threads"] for r in picked], [r["speedup"] for r in picked], "s",
                    color=colors["dynamic"], label="dynamic")
        ax.plot(args.threads, args.threads, "k--", alpha=0.6, label="линейное ускорение")
        ax.set_title(f"Ускорение, вид {view} ({args.width}×{args.height})")
        ax.set_xlabel("Число потоков")
        ax.set_ylabel("S = T(1) / T(p)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("speedup_vs_threads.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(8, 4.8))
    mpix = [r["mpix"] for r in size_rows]
    plt.plot(mpix, [r["serial_sec"] for r in size_rows], marker="o", label="1 поток")
    plt.plot(mpix, [r["threaded_sec"] for r in size_rows], marker="s", label="8 потоков (interleaved)")
    plt.title("Время генерации от размера изображения (вид 1)")
    plt.xlabel("Размер изображения, Мпикс")
    plt.ylabel("Время, с")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("time_vs_size.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, view in zip(axes, (1, 2)):
        for mode in args.modes:
            picked = [r for r in rows if r["series"] == "threads" and r["view"] == view
                      and r["mode"] == mode and r["threads"] == 8]
            if not picked:
                continue
            times = [float(v) for v in picked[0]["thread_times"].split(",")]
            ax.bar([i + (0.4 if mode == "interleaved" else 0) for i in range(len(times))], times,
                   width=0.4, color=colors.get(mode), label=mode)
        ax.set_title(f"Время потоков, вид {view}, 8 потоков")
        ax.set_xlabel("Поток")
        ax.set_ylabel("Время, с")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("thread_balance.png", dpi=160)
    plt.close(fig)

    # картинка фрактала для отчёта
    subprocess.run([args.exe, "--view", "2", "--mode", "serial", "--width", "800", "--height", "600",
                    "--ppm", "mandelbrot.ppm"], check=True, capture_output=True, text=True)
    import matplotlib.image as mpimg
    image = mpimg.imread("mandelbrot.ppm")
    plt.imsave("mandelbrot.png", image)
    print("Saved speedup_vs_threads.png, time_vs_size.png, thread_balance.png, mandelbrot.png")
    return True


charts_ok = False if args.no_charts else charts()


# ------------------------------------------------------------------ README ---
def table_speedup(view):
    head = ("| Режим |" + " | ".join(f"p={t}" for t in args.threads) + " |\n|"
            + "---:|" + "---:|" * len(args.threads) + "\n")
    lines = []
    for mode in args.modes:
        picked = {r["threads"]: r for r in rows
                  if r["series"] == "threads" and r["view"] == view and r["mode"] == mode}
        lines.append(f"| {mode} | " + " | ".join(f"{picked[t]['speedup']:.2f}"
                                                for t in args.threads) + " |\n")
    dynamic = {r["threads"]: r for r in rows if r["series"] == "dynamic" and r["view"] == view}
    if dynamic:
        cells = [f"{dynamic[t]['speedup']:.2f}" if t in dynamic else "—" for t in args.threads]
        lines.append("| dynamic | " + " | ".join(cells) + " |\n")
    return f"**Ускорение относительно 1 потока, вид {view}:**\n\n" + head + "".join(lines)


def write_readme(charts_ok):
    verified = "PASS" if all(r["verified"] == "PASS" for r in rows if r["series"] == "threads") else "FAIL"
    size_head = "| Изображение | Мпикс | 1 поток, с | 8 потоков, с | Ускорение |\n|---:|---:|---:|---:|---:|\n"
    size_body = "".join(f"| {r['size']} | {r['mpix']:.2f} | {r['serial_sec']:.4f} | "
                        f"{r['threaded_sec']:.4f} | {r['speedup']:.2f} |\n" for r in size_rows)
    images = ("\n![Ускорение от числа потоков](speedup_vs_threads.png)\n\n"
              "![Время от размера](time_vs_size.png)\n\n"
              "![Загрузка потоков](thread_balance.png)\n\n"
              "![Фрактал](mandelbrot.png)\n") if charts_ok else ""
    block = (
        f"\n**Условия измерений:** {platform.platform()}; Python {platform.python_version()}; "
        f"max_iter={args.max_iter}; замеров на точку: {args.repeats} (в таблицах — минимум). "
        f"Верификация NumPy и встроенная сверка с последовательной версией: **{verified}**.\n\n"
        + "**Исходные данные для графиков** — таблицы ниже, полные данные в `benchmark.csv`, "
          "протокол запусков в `results.txt`.\n\n"
        + "**Ускорение по сравнению с 1 потоком (вид 1 и вид 2):**\n\n"
        + table_speedup(1)
        + "\n" + table_speedup(2)
        + "\n**Время от размера изображения (вид 1):**\n\n" + size_head + size_body
        + images
    )
    report = Path("README.md")
    if not report.exists():
        return
    text = report.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    report.write_text(text.split(start)[0] + start + block + end + text.split(end)[1],
                      encoding="utf-8")
    print("Updated README.md")


LOG_PATH.write_text("\n".join(log) + "\n", encoding="utf-8")
print(f"Saved {LOG_PATH}")
if not args.no_readme:
    write_readme(charts_ok)
