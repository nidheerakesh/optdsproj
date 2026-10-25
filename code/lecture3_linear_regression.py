import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

from common import dot, sub, scale, mat_vec, transpose, log, set_seed, save_plot, write_table, run_dir, solve
import config as cfg


def augment(X):
    return [[1.0] + list(row) for row in X]


def predict(Xa, w):
    return mat_vec(Xa, w)


def loss(Xa, w, Y):
    E = sub(predict(Xa, w), Y)
    return dot(E, E) / (2 * len(Y))


def grad(Xa, w, Y):
    E = sub(predict(Xa, w), Y)
    N = len(Y)
    XT = transpose(Xa)
    return scale(1.0 / N, mat_vec(XT, E))


def gd(Xa, Y, w0, alpha, max_it):
    w = list(w0)
    hist = [loss(Xa, w, Y)]
    for _ in range(max_it):
        w = sub(w, scale(alpha, grad(Xa, w, Y)))
        hist.append(loss(Xa, w, Y))
    return w, hist


def sgd_step(xi, yi, w, alpha):
    E = dot(xi, w) - yi
    return sub(w, scale(alpha, scale(E, xi))), 0.5 * E * E


def sgd_fixed(Xa, Y, w0, alpha, picks):
    w = list(w0)
    rows = []
    for i in picks:
        w_new, li = sgd_step(Xa[i], Y[i], w, alpha)
        rows.append((i, dot(Xa[i], w), dot(Xa[i], w) - Y[i], li, w_new))
        w = w_new
    return w, rows


def sgd_random(Xa, Y, w0, alpha, max_it):
    w = list(w0)
    N = len(Y)
    hist = [loss(Xa, w, Y)]
    for _ in range(max_it):
        i = random.randrange(N)
        w, _ = sgd_step(Xa[i], Y[i], w, alpha)
        hist.append(loss(Xa, w, Y))
    return w, hist


def minibatch(Xa, Y, w0, alpha, max_it, bs):
    w = list(w0)
    N = len(Y)
    hist = [loss(Xa, w, Y)]
    for _ in range(max_it):
        idx = [random.randrange(N) for _ in range(bs)]
        Xb = [Xa[i] for i in idx]
        Yb = [Y[i] for i in idx]
        w = sub(w, scale(alpha, grad(Xb, w, Yb)))
        hist.append(loss(Xa, w, Y))
    return w, hist


def normal_equation(Xa, Y):
    XT = transpose(Xa)
    A = [[dot(XT[i], XT[j]) for j in range(len(XT))] for i in range(len(XT))]
    b = mat_vec(XT, Y)
    return solve(A, b)


def synthetic(N, d, seed):
    random.seed(seed)
    true_w = [random.uniform(-2, 2) for _ in range(d)]
    X = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(N)]
    Y = [dot(x, true_w) + 0.5 + random.gauss(0, 0.1) for x in X]
    return X, Y, true_w


def worked_example(out, lg):
    c = cfg.L3
    Xa = augment(c["X"])
    w0 = [c["b"]] + list(c["W"])
    lg(f"Worked example augmented X={Xa} w0={w0} Y={c['Y']} alpha={c['alpha']}")
    lg(f"Initial loss={loss(Xa, w0, c['Y']):.6f}")
    w = list(w0)
    for k in range(1, c["gd_iters"] + 1):
        E = sub(predict(Xa, w), c["Y"])
        g = grad(Xa, w, c["Y"])
        lg(f"GD iter {k}: pred={predict(Xa, w)} E={E} loss={loss(Xa, w, c['Y']):.6f} grad={g}")
        w = sub(w, scale(c["alpha"], g))
        lg(f"GD iter {k}: w_new={w}")
    w_s, rows = sgd_fixed(Xa, c["Y"], w0, c["alpha"], c["sgd_pick"])
    for i, pred, E, li, w_new in rows:
        lg(f"SGD pick row {i}: pred={pred} E={E} grad={scale(E, Xa[i])} loss={li:.6f} w_new={w_new}")
    return w, w_s


def run_synthetic(out, lg):
    N, d = 200, 3
    X, Y, true_w = synthetic(N, d, cfg.SEED)
    Xa = augment(X)
    lg(f"Synthetic N={N} d={d} true_w (weights, excl. bias 0.5)={true_w}")
    lg(f"Seed {cfg.SEED}: data generation; re-seeded before SGD; mini-batch continues that stream")
    w0 = [0.0] * (d + 1)
    alpha = 0.1
    w_gd, h_gd = gd(Xa, Y, w0, alpha, 300)
    random.seed(cfg.SEED)
    w_sgd, h_sgd = sgd_random(Xa, Y, w0, 0.01, 300 * N)
    random.seed(cfg.SEED)
    w_mb, h_mb = minibatch(Xa, Y, w0, 0.05, 2000, 16)
    lg(f"GD final loss={h_gd[-1]:.6f} w={w_gd}")
    lg(f"SGD final loss={h_sgd[-1]:.6f} w={w_sgd}")
    lg(f"MiniBatch(16) final loss={h_mb[-1]:.6f} w={w_mb}")
    w_ne = normal_equation(Xa, Y)
    lg(f"Normal equation (bias first) w={w_ne} loss={loss(Xa, w_ne, Y):.6f} true_w (excl. bias)={true_w}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(h_gd)), h_gd, label="GD")
    ax.semilogy([k / N for k in range(0, len(h_sgd), N)], h_sgd[::N], label="SGD (x = epochs)")
    ax.semilogy(range(len(h_mb)), h_mb, label="Mini-batch 16")
    ax.set_xlabel("iteration")
    ax.set_ylabel("training loss")
    ax.legend()
    save_plot(fig, out, "l3_linreg_loss_comparison.png")
    write_table(os.path.join(out, "l3_gd_loss.csv"), ["iter", "loss"], [[k, v] for k, v in enumerate(h_gd)])


def main(out):
    log_path = os.path.join(out, "log.txt")
    set_seed(cfg.SEED, log_path)

    def lg(msg):
        log(log_path, msg)

    worked_example(out, lg)
    run_synthetic(out, lg)


if __name__ == "__main__":
    main(run_dir())
