#!/bin/bash
# Серия экспериментов на кластере «Сергей Королёв».
# Запускается из задания PBS (lab3.pbs). Python на узлах не требуется:
# матрицы генерирует сама MPI-программа (--generate), результат пишется в results.txt.
set -e

SIZES="200 400 800 1200 1600 2000"
PROCS="1 2 4 8 16 32"
REPEATS=3
LOG=results.txt

SLOTS=$(wc -l < "$PBS_NODEFILE" 2>/dev/null || echo 1)
NODES=$(sort -u "$PBS_NODEFILE" 2>/dev/null | wc -l)

{
    echo "# Протокол экспериментов (л/р №3, MPI)"
    echo "# Узлы задания: $(sort -u "$PBS_NODEFILE" 2>/dev/null | tr '\n' ' ')"
    echo "# Выделено процессов: $SLOTS (узлов: $NODES)"
    echo "# Размеры: $SIZES"
    echo "# Число процессов: $PROCS"
    echo "# Замеров на точку: $REPEATS, прогревов: 1"
} > "$LOG"

MPIRUN="mpirun -r ssh -machinefile $PBS_NODEFILE"

for N in $SIZES; do
    echo "== N=$N: генерирую матрицы =="
    $MPIRUN -np 1 ./matrix_mpi --generate "$N" --repeats 1 --warmup 0 > /dev/null
    for P in $PROCS; do
        if [ "$P" -gt "$SLOTS" ] && [ "$P" -gt 1 ]; then
            echo "-- p=$P пропущено: выделено только $SLOTS процессов"
            continue
        fi
        EXTRA=""
        if [ "$P" -eq 4 ]; then EXTRA="--verify"; fi
        echo "\$ mpirun -np $P ./matrix_mpi --repeats $REPEATS --warmup 1 $EXTRA" >> "$LOG"
        $MPIRUN -np "$P" ./matrix_mpi --repeats "$REPEATS" --warmup 1 $EXTRA | tee -a "$LOG"
        if python3 -c "import numpy" 2>/dev/null; then
            python3 check.py | tee -a "$LOG"
        fi
    done
done

echo "== готово: результаты в $LOG =="
