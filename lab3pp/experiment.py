import argparse
import csv
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser(description="MPI matrix multiplication benchmarks")
p.add_argument("--exe", default="./matrix_mpi")
p.add_argument("--mpirun", default="mpirun")
p.add_argument("--mpirun-args", default="", help="дополнительные аргументы mpirun")
p.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
p.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--verify-at", type=int, default=4,
               help="для какого числа процессов добавить --verify (0 — не проверять встроенной сверкой)")
p.add_argument("--skip-verify", action="store_true", help="не запускать check.py (например, если нет NumPy)")
p.add_argument("--no-charts", action="store_true", help="не строить графики (на кластере может не быть matplotlib)")
p.add_argument("--no-readme", action="store_true", help="не обновлять README.md")
p.add_argument("--from-csv", action="store_true",
               help="не измерять заново: взять benchmark.csv, построить графики и обновить README")
p.add_argument("--from-log", action="store_true",
               help="не измерять заново: разобрать results.txt (протокол с кластера) и собрать всё по нему")
p.add_argument("--csv", default="benchmark.csv")
p.add_argument("--log", default="results.txt")
args = p.parse_args()
if args.repeats < 1 or args.warmup < 0 or any(n < 1 for n in args.sizes):
    p.error("Sizes, repeats and warmup must be positive")

CSV_PATH = Path(args.csv)
LOG_PATH = Path(args.log)


def mpirun_command(procs):
    command = args.mpirun.split()
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and nodefile not in args.mpirun_args:
        # Внутри задания PBS для Intel MPI их документация рекомендует -r ssh -machinefile
        command += ["-r", "ssh", "-machinefile", nodefile]
    if args.mpirun_args:
        command += args.mpirun_args.split()
    command += ["-np", str(procs), args.exe, "--repeats", str(args.repeats),
                "--warmup", str(args.warmup)]
    if args.verify_at and procs == args.verify_at:
        command.append("--verify")
    return command


def plot_charts(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: python3 -m pip install --user -r requirements.txt. "
              "Charts were not created.")
        return False

    sizes = sorted({r["N"] for r in rows})
    procs_list = sorted({r["procs"] for r in rows})
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for i, procs in enumerate(procs_list):
        picked = [r for r in rows if r["procs"] == procs]
        left.plot([r["N"] for r in picked], [r["time_min_sec"] for r in picked], marker="o",
                  color=colors[i % len(colors)], label=f"{procs} проц.")
        right.plot([r["N"] for r in picked], [r["gflops"] for r in picked], marker="o",
                   color=colors[i % len(colors)], label=f"{procs} проц.")
    left.set_title("Время умножения (MPI)")
    left.set_xlabel("Размер матрицы N")
    left.set_ylabel("Время, с (минимум из замеров)")
    left.set_yscale("log")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=8)
    right.set_title("Производительность")
    right.set_xlabel("Размер матрицы N")
    right.set_ylabel("Гфлоп/с (2N³ / T)")
    right.grid(True, alpha=0.3)
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("performance.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(8, 4.8))
    for i, n in enumerate(sizes):
        picked = [r for r in rows if r["N"] == n]
        plt.plot([r["procs"] for r in picked], [r["speedup"] for r in picked], marker="o",
                 color=colors[i % len(colors)], label=f"N={n}")
    plt.plot(procs_list, procs_list, "k--", alpha=0.6, label="линейное ускорение")
    plt.title("Ускорение S(p) = T(1)/T(p)")
    plt.xlabel("Число процессов p")
    plt.ylabel("Ускорение")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("speedup.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    for i, n in enumerate(sizes):
        picked = [r for r in rows if r["N"] == n]
        plt.plot([r["procs"] for r in picked], [100 * r["efficiency"] for r in picked], marker="o",
                 color=colors[i % len(colors)], label=f"N={n}")
    plt.axhline(100, color="k", linestyle="--", alpha=0.6, label="100 %")
    plt.title("Эффективность распараллеливания E(p) = S(p)/p")
    plt.xlabel("Число процессов p")
    plt.ylabel("Эффективность, %")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("efficiency.png", dpi=160)
    plt.close()
    print("Saved performance.png, speedup.png and efficiency.png")
    return True


def update_readme(rows, charts, source="local"):
    sizes = sorted({r["N"] for r in rows})
    procs_list = sorted({r["procs"] for r in rows})

    def table(key, header, formatter):
        head = ("| N | " + " | ".join(f"p={p}" for p in procs_list) + " |\n|"
                + "---:|" + "---:|" * len(procs_list) + "\n")
        lines = []
        for n in sizes:
            picked = {r["procs"]: r for r in rows if r["N"] == n}
            lines.append(f"| {n} | " + " | ".join(formatter(picked[p][key]) for p in procs_list) + " |\n")
        return header + head + "".join(lines)

    nodes = max(r["nodes"] for r in rows)
    verified = "PASS" if all(r["verified"] == "PASS" for r in rows) else "FAIL"
    images = ""
    if charts:
        images = ("\n![Время и производительность](performance.png)\n\n"
                  "![Ускорение](speedup.png)\n\n"
                  "![Эффективность](efficiency.png)\n")
    where = ("суперкомпьютер «Сергей Королёв» (данные из `results.txt`)" if source == "cluster"
             else f"{platform.platform()}, локальный прогон")
    block = (
        f"\n**Условия измерений:** {where}; Python {platform.python_version()}; "
        f"замеров на точку: {max(r['repeats'] for r in rows)} (+1 прогрев), в таблицах — минимум; "
        f"время — полный цикл «рассылка данных → счёт → сбор результата»; "
        f"наибольшее число узлов в замеренной серии: {nodes}. "
        f"Верификация: **{verified}** для всех {len(rows)} точек.\n\n"
        + "**Исходные данные для графиков** — таблицы ниже (по ним построены графики), "
          "полные данные лежат в `benchmark.csv`, протокол запусков — в `results.txt`.\n\n"
        + table("time_min_sec", "**Время T(p), с:**\n\n", lambda v: f"{v:.6f}")
        + "\n" + table("speedup", "**Ускорение S(p) = T(1)/T(p):**\n\n", lambda v: f"{v:.2f}")
        + "\n" + table("efficiency", "**Эффективность E(p) = S(p)/p:**\n\n", lambda v: f"{100 * v:.1f} %")
        + "\n" + table("comm_share", "**Доля обмена данными в общем времени, %:**\n\n", lambda v: f"{v:.1f}")
        + "\n" + table("gflops", "**Производительность, Гфлоп/с (2N³ / T(p)):**\n\n", lambda v: f"{v:.1f}")
        + "\n" + "**Число узлов по конфигурациям:** " + ", ".join(
            f"p={p} → {max(r['nodes'] for r in rows if r['procs'] == p)} узл." for p in procs_list) + "\n"
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


def read_csv(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = {k: row[k] for k in row}
            for key in ("N", "procs", "nodes", "repeats"):
                item[key] = int(row[key])
            for key in ("time_min_sec", "time_mean_sec", "compute_sec", "comm_sec",
                        "gflops", "speedup", "efficiency", "comm_share"):
                item[key] = float(row[key])
            rows.append(item)
    return rows


if args.from_csv:
    rows = read_csv(CSV_PATH)
    print(f"Loaded {len(rows)} points from {CSV_PATH}")
    charts = False if args.no_charts else plot_charts(rows)
    if not args.no_readme:
        update_readme(rows, charts)
    raise SystemExit(0)

def finish(rows, source):
    """Сохранить CSV, построить графики и обновить README по готовым строкам."""
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {CSV_PATH}")
    charts = False if args.no_charts else plot_charts(rows)
    if not args.no_readme:
        update_readme(rows, charts, source)


def rows_from_log(path):
    """Разобрать протокол кластера (results.txt) в строки результатов."""
    text = Path(path).read_text(encoding="utf-8")
    numPy_ok = "PASS: NumPy verification succeeded" in text
    numPy_fail = "FAIL:" in text
    rows = []
    baselines = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("N="):
            continue
        fields = dict(re.findall(r"(\w+)=(\"[^\"]*\"|\S+)", line))
        n = int(fields["N"])
        procs = int(fields["procs"])
        fastest = float(fields["time_min_sec"])
        mean = float(fields["time_sec"])
        compute = float(fields["compute_sec"])
        comm = float(fields["comm_sec"])
        baselines.setdefault(n, fastest)  # первая строка размера — это p = 1
        baseline = baselines[n]
        if "verify=FAIL" in line or numPy_fail:
            verified = "FAIL"
        elif "verify=PASS" in line or numPy_ok:
            verified = "PASS"
        else:
            verified = "SKIP"
        rows.append({
            "N": n,
            "procs": procs,
            "nodes": int(fields.get("nodes", 1)),
            "time_min_sec": round(fastest, 6),
            "time_mean_sec": round(mean, 6),
            "compute_sec": round(compute, 6),
            "comm_sec": round(comm, 6),
            "operations": 2 * n ** 3,
            "memory_bytes": 3 * n * n * 8,
            "gflops": round(2 * n ** 3 / fastest / 1e9, 3),
            "speedup": round(baseline / fastest, 3),
            "efficiency": round(baseline / fastest / procs, 3),
            "comm_share": round(100.0 * comm / fastest, 2) if fastest > 0 else 0.0,
            "verified": verified,
            "repeats": int(fields.get("repeats", 0)),
        })
    return rows


if args.from_log:
    rows = rows_from_log(LOG_PATH)
    if not rows:
        raise SystemExit(f"в {LOG_PATH} не найдено строк с результатами")
    print(f"Loaded {len(rows)} points from {LOG_PATH}")
    finish(rows, "cluster")
    raise SystemExit(0)
rows = []
log = ["# Протокол экспериментов (л/р №3, MPI)",
       f"# Дата: {datetime.now():%Y-%m-%d %H:%M}",
       f"# Система: {platform.platform()}",
       f"# Размеры: {args.sizes}",
       f"# Процессы: {args.procs}",
       f"# Замеров на точку: {args.repeats}, прогревов: {args.warmup}",
       f"# PBS_NODEFILE: {os.environ.get('PBS_NODEFILE', 'нет')}",
       "#" + "-" * 70]

for n in args.sizes:
    out = subprocess.run([sys.executable, "generate.py", str(n)],
                         check=True, capture_output=True, text=True)
    log.append(f"$ python3 generate.py {n}\n{out.stdout.strip()}")

    times = {}
    for procs in args.procs:
        command = mpirun_command(procs)
        log.append(f"$ {' '.join(command)}")
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            log.append(proc.stdout.strip())
            log.append(proc.stderr.strip())
            raise SystemExit(f"mpirun failed for N={n}, procs={procs}:\n{proc.stderr}")
        log.append(proc.stdout.strip())
        output = dict(re.findall(r"(\w+)=(\"[^\"]*\"|\S+)", proc.stdout))
        fastest = float(output["time_min_sec"])
        mean = float(output["time_sec"])
        compute = float(output["compute_sec"])
        comm = float(output["comm_sec"])
        nodes = int(output["nodes"])
        times[procs] = fastest
        print(f"N={n}, procs={procs}, nodes={nodes}: min {fastest:.6f} s "
              f"(compute {compute:.6f}, comm {comm:.6f})", flush=True)

        verified = "SKIP"
        if not args.skip_verify:
            check = subprocess.run([sys.executable, "check.py"], capture_output=True, text=True)
            log.append(f"$ python3 check.py\n{check.stdout.strip()}")
            verified = "PASS" if "PASS" in check.stdout else "FAIL"
        if "verify=PASS" in proc.stdout:
            verified = "PASS"

        baseline = times[1] if 1 in times else fastest
        rows.append({
            "N": n,
            "procs": procs,
            "nodes": nodes,
            "time_min_sec": round(fastest, 6),
            "time_mean_sec": round(mean, 6),
            "compute_sec": round(compute, 6),
            "comm_sec": round(comm, 6),
            "operations": 2 * n ** 3,
            "memory_bytes": 3 * n * n * 8,
            "gflops": round(2 * n ** 3 / fastest / 1e9, 3),
            "speedup": round(baseline / fastest, 3),
            "efficiency": round(baseline / fastest / procs, 3),
            "comm_share": round(100.0 * comm / fastest, 2) if fastest > 0 else 0.0,
            "verified": verified,
            "repeats": args.repeats,
        })



LOG_PATH.write_text("\n".join(log) + "\n", encoding="utf-8")
print(f"Saved {LOG_PATH}")
finish(rows, "local")
