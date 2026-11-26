import argparse
import sys

import numpy as np


def make_view(index):
    x0, y0, x1, y1 = np.float32(-2.0), np.float32(-1.0), np.float32(1.0), np.float32(1.0)
    if index == 2:
        scale = np.float32(0.015)
        shift_x = np.float32(-0.986)
        shift_y = np.float32(0.30)
        x0 = x0 * scale + shift_x
        x1 = x1 * scale + shift_x
        y0 = y0 * scale + shift_y
        y1 = y1 * scale + shift_y
    elif index != 1:
        raise SystemExit(f"unknown view: {index}")
    return x0, y0, x1, y1


def mandelbrot(width, height, view, max_iterations):
    """Эталон на NumPy: те же формулы и порядок операций, что в C++.float."""
    x0, y0, x1, y1 = view
    dx = (x1 - x0) / np.float32(width)
    dy = (y1 - y0) / np.float32(height)

    xs = x0 + np.arange(width, dtype=np.float32) * dx
    ys = y0 + np.arange(height, dtype=np.float32) * dy
    c_re = np.broadcast_to(xs[None, :], (height, width)).astype(np.float32)
    c_im = np.broadcast_to(ys[:, None], (height, width)).astype(np.float32)

    z_re = c_re.copy()
    z_im = c_im.copy()
    result = np.full((height, width), max_iterations, dtype=np.int32)
    active = np.ones((height, width), dtype=bool)
    two = np.float32(2.0)
    four = np.float32(4.0)

    for iteration in range(max_iterations):
        magnitude = z_re * z_re + z_im * z_im
        escaped = active & (magnitude > four)
        result[escaped] = iteration
        active &= ~escaped
        if not active.any():
            break
        new_re = z_re * z_re - z_im * z_im
        new_im = two * z_re * z_im
        z_re = np.where(active, c_re + new_re, z_re)
        z_im = np.where(active, c_im + new_im, z_im)
    return result


p = argparse.ArgumentParser(description="Проверка счётчиков итераций Мандельброта через NumPy")
p.add_argument("--counts", default="counts.bin")
p.add_argument("--width", type=int, required=True)
p.add_argument("--height", type=int, required=True)
p.add_argument("--view", type=int, default=1)
p.add_argument("--max-iter", type=int, default=256)
p.add_argument("--allow-mismatch-percent", type=float, default=0.0,
               help="допустимая доля расходящихся пикселей, %%")
args = p.parse_args()

got = np.fromfile(args.counts, dtype=np.uint16)
expected_size = args.width * args.height
if got.size != expected_size:
    print(f"FAIL: в {args.counts} {got.size} значений, ожидалось {expected_size}")
    raise SystemExit(1)
got = got.reshape(args.height, args.width).astype(np.int32)

reference = mandelbrot(args.width, args.height, make_view(args.view), args.max_iter)
difference = np.abs(reference - got)
mismatches = int(np.count_nonzero(difference))
percent = 100.0 * mismatches / expected_size
max_diff = int(difference.max()) if mismatches else 0

print(f"pixels={expected_size} mismatched={mismatches} ({percent:.4f} %) max_diff={max_diff}")
if percent <= args.allow_mismatch_percent:
    print("PASS: NumPy verification succeeded")
else:
    print("FAIL: результаты расходятся с эталоном NumPy")
    raise SystemExit(1)
