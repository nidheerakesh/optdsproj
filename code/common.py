import os
import random
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ROOT, "results", ts)
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed, log_path):
    random.seed(seed)
    with open(log_path, "a") as fh:
        fh.write(f"seed={seed}\n")


def log(log_path, msg):
    with open(log_path, "a") as fh:
        fh.write(msg + "\n")
    print(msg)


def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def scale(c, a):
    return [c * a[i] for i in range(len(a))]


def norm(a):
    return dot(a, a) ** 0.5


def mat_vec(M, v):
    return [dot(row, v) for row in M]


def transpose(M):
    return [list(col) for col in zip(*M)]


def solve(A, b):
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [M[r][j] - f * M[c][j] for j in range(n + 1)]
    return [M[i][n] / M[i][i] for i in range(n)]


def save_plot(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def write_table(path, header, rows):
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")
