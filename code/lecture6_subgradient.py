import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

from common import dot, add, sub, scale, mat_vec, transpose, log, set_seed, save_plot, run_dir
import config as cfg


def sgn(x):
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def sub_abs(x):
    return [1.0] if x > 0 else ([-1.0] if x < 0 else [-1.0, 1.0])


def subgradient_method(f, subg, x0, step, max_it):
    x = list(x0)
    fbest = f(x)
    xbest = list(x)
    hist = [fbest]
    for k in range(max_it):
        g = subg(x)
        x = sub(x, scale(step(k), g))
        fn = f(x)
        if fn < fbest:
            fbest = fn
            xbest = list(x)
        hist.append(fbest)
    return xbest, fbest, hist


def hinge_check(lg):
    lg("Hinge f(x)=max(0,1-x): subdifferential at x=1 should be [-1,0]")
    for g in [-1.0, -0.5, 0.0, 0.5]:
        ok = all(max(0.0, 1 - z) >= max(0.0, 1 - 1) + g * (z - 1) - 1e-12 for z in [i / 100 for i in range(-300, 301)])
        lg(f"  g={g}: subgradient inequality holds on grid = {ok}")


def abs_check(lg):
    lg("f(x)=|x| at x=0: subgradient 0.5 supports the graph (f(z) >= 0.5 z for all z)")
    ok = all(abs(z) >= 0.5 * z - 1e-12 for z in [i / 100 for i in range(-300, 301)])
    lg(f"  inequality holds on grid = {ok}")


def relu(x):
    return max(0.0, x)


def relu_check(lg):
    lg("ReLU f(x)=max(0,x): subdifferential is {0} for x<0, {1} for x>0, [0,1] at x=0")
    for x0, gs in [(0.0, [-0.1, 0.0, 0.5, 1.0, 1.5]), (-1.0, [0.0, 0.1]), (2.0, [1.0, 0.5])]:
        for g in gs:
            ok = all(relu(z) >= relu(x0) + g * (z - x0) - 1e-12 for z in [i / 100 for i in range(-300, 301)])
            lg(f"  x={x0} g={g}: subgradient inequality holds on grid = {ok}")


def f_abs_sum(x):
    return abs(x[0]) + 2 * abs(x[1])


def subg_abs_sum(x):
    return [sgn(x[0]), 2.0 * sgn(x[1])]


def run_examples(out, lg):
    abs_check(lg)
    hinge_check(lg)
    relu_check(lg)
    x0 = [3.0, -2.0]
    xb, fb, hist = subgradient_method(f_abs_sum, subg_abs_sum, x0, lambda k: 0.5 / (k + 1) ** 0.5, 400)
    lg(f"f=|x1|+2|x2| subgradient method from {x0}: best_x={xb} best_f={fb:.6f}")
    lg("Note: the slide writes x1+2|x2| but its subgradient cases use |x1|, so the code minimizes |x1|+2|x2| (bounded below, minimum at 0)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(hist)), [max(h - min(hist) + 1e-12, 1e-12) for h in hist], label="best f - min f")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best f so far (shifted)")
    ax.set_title("Subgradient method on |x1| + 2|x2|")
    ax.legend()
    save_plot(fig, out, "l6_subgradient_abs_sum.png")


def lasso_subgrad(X, Y, lam, max_it):
    N = len(Y)
    d = len(X[0])
    XT = transpose(X)

    def loss(w):
        E = [y - p for y, p in zip(Y, mat_vec(X, w))]
        return sum(e * e for e in E) / (2 * N) + lam * sum(abs(v) for v in w)

    def subg(w):
        E = [y - p for y, p in zip(Y, mat_vec(X, w))]
        gd = scale(-1.0 / N, mat_vec(XT, E))
        return [gd[j] + lam * sgn(w[j]) for j in range(d)]

    w0 = [0.0] * d
    return subgradient_method(loss, subg, w0, lambda k: 0.1 / (k + 1) ** 0.5, max_it)


def hinge_svm_subgrad(X, Y, lam, max_it):
    N = len(Y)
    d = len(X[0])

    def loss(p):
        w, b = p[:d], p[d]
        s = 0.0
        for x, y in zip(X, Y):
            s += max(0.0, 1 - y * (dot(w, x) + b))
        return s / N + 0.5 * lam * dot(w, w)

    def subg(p):
        w, b = p[:d], p[d]
        gw = [lam * w[j] for j in range(d)]
        gb = 0.0
        for x, y in zip(X, Y):
            if 1 - y * (dot(w, x) + b) > 0:
                for j in range(d):
                    gw[j] -= y * x[j] / N
                gb -= y / N
        return gw + [gb]

    p0 = [0.0] * (d + 1)
    return subgradient_method(loss, subg, p0, lambda k: 0.5 / (k + 1) ** 0.5, max_it)


def run_lasso(out, lg):
    random.seed(cfg.SEED)
    N, d = 150, 8
    true_w = [3.0, -2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0]
    X = [[random.gauss(0, 1) for _ in range(d)] for _ in range(N)]
    Y = [dot(x, true_w) + random.gauss(0, 0.1) for x in X]
    lg(f"Lasso synthetic N={N} d={d} true_w={true_w} seed={cfg.SEED}")
    rows = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for lam in [0.01, 0.1, 0.5]:
        wb, fb, hist = lasso_subgrad(X, Y, lam, 3000)
        small = [round(v, 3) for v in wb]
        lg(f"lambda={lam}: best loss={fb:.6f} w={small} |w|<1e-2 count={sum(1 for v in wb if abs(v) < 1e-2)}")
        rows.append([lam, fb] + small)
        ax.semilogy(range(len(hist)), [max(h - min(hist) + 1e-12, 1e-12) for h in hist], label=f"lambda={lam}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best loss (shifted)")
    ax.set_title("Lasso subgradient method")
    ax.legend()
    save_plot(fig, out, "l6_lasso_convergence.png")


def run_svm(out, lg):
    random.seed(cfg.SEED + 1)
    N = 120
    X, Y = [], []
    for _ in range(N):
        x = [random.gauss(0, 1), random.gauss(0, 1)]
        y = 1.0 if x[0] + x[1] > 0 else -1.0
        X.append(x)
        Y.append(y)
    wb, fb, hist = hinge_svm_subgrad(X, Y, 0.01, 2000)
    d = 2
    lg(f"Hinge SVM subgradient: best objective={fb:.6f} w={[round(v, 4) for v in wb[:d]]} b={wb[d]:.4f}")
    acc = sum(1 for x, y in zip(X, Y) if y * (dot(wb[:d], x) + wb[d]) > 0) / N
    lg(f"Hinge SVM training accuracy={acc:.3f}")
    fig, ax = plt.subplots(figsize=(6, 4))
    for x, y in zip(X, Y):
        ax.scatter(x[0], x[1], c="tab:blue" if y > 0 else "tab:orange", s=12)
    xs = [-3, 3]
    if abs(wb[1]) > 1e-9:
        ax.plot(xs, [-(wb[0] * t + wb[d]) / wb[1] for t in xs], "k-")
    ax.set_title(f"Hinge-loss linear SVM (acc={acc:.2f})")
    save_plot(fig, out, "l6_svm_hinge_boundary.png")


def main(out):
    log_path = os.path.join(out, "log.txt")
    set_seed(cfg.SEED, log_path)

    def lg(msg):
        log(log_path, msg)

    run_examples(out, lg)
    run_lasso(out, lg)
    run_svm(out, lg)


if __name__ == "__main__":
    main(run_dir())
