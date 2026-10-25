import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

from common import dot, add, sub, scale, mat_vec, transpose, log, set_seed, save_plot, run_dir
import config as cfg


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict(X, w, b):
    return [sigmoid(dot(x, w) + b) for x in X]


def cross_entropy(X, w, b, Y):
    P = predict(X, w, b)
    N = len(Y)
    s = 0.0
    for p, y in zip(P, Y):
        s += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -s / N


def grad(X, w, b, Y):
    E = [p - y for p, y in zip(predict(X, w, b), Y)]
    N = len(Y)
    gw = scale(1.0 / N, mat_vec(transpose(X), E))
    gb = sum(E) / N
    return gw, gb


def gd(X, Y, w0, b0, alpha, max_it):
    w, b = list(w0), b0
    hist = [cross_entropy(X, w, b, Y)]
    for _ in range(max_it):
        gw, gb = grad(X, w, b, Y)
        w = sub(w, scale(alpha, gw))
        b = b - alpha * gb
        hist.append(cross_entropy(X, w, b, Y))
    return w, b, hist


def sgd_fixed(X, Y, w0, b0, alpha, picks):
    w, b = list(w0), b0
    rows = []
    for i in picks:
        p = sigmoid(dot(X[i], w) + b)
        e = p - Y[i]
        w = sub(w, scale(alpha, scale(e, X[i])))
        b = b - alpha * e
        rows.append((i, p, e, w, b))
    return w, b, rows


def sgd_random(X, Y, w0, b0, alpha, max_it):
    w, b = list(w0), b0
    N = len(Y)
    hist = [cross_entropy(X, w, b, Y)]
    for _ in range(max_it):
        i = random.randrange(N)
        p = sigmoid(dot(X[i], w) + b)
        e = p - Y[i]
        w = sub(w, scale(alpha, scale(e, X[i])))
        b = b - alpha * e
        hist.append(cross_entropy(X, w, b, Y))
    return w, b, hist


def adagrad(X, Y, w0, b0, alpha, eps, max_it, trace=None):
    w, b = list(w0), b0
    G = [0.0] * len(w)
    Gb = 0.0
    hist = [cross_entropy(X, w, b, Y)]
    for k in range(1, max_it + 1):
        gw, gb = grad(X, w, b, Y)
        G = [G[j] + gw[j] * gw[j] for j in range(len(w))]
        Gb += gb * gb
        w = [w[j] - alpha / math.sqrt(G[j] + eps) * gw[j] for j in range(len(w))]
        b = b - alpha / math.sqrt(Gb + eps) * gb
        hist.append(cross_entropy(X, w, b, Y))
        if trace is not None:
            trace(k, gw, gb, G, Gb, w, b)
    return w, b, hist


def rmsprop(X, Y, w0, b0, alpha, rho, eps, max_it):
    w, b = list(w0), b0
    G = [0.0] * len(w)
    Gb = 0.0
    hist = [cross_entropy(X, w, b, Y)]
    for _ in range(max_it):
        gw, gb = grad(X, w, b, Y)
        G = [rho * G[j] + (1 - rho) * gw[j] * gw[j] for j in range(len(w))]
        Gb = rho * Gb + (1 - rho) * gb * gb
        w = [w[j] - alpha / math.sqrt(G[j] + eps) * gw[j] for j in range(len(w))]
        b = b - alpha / math.sqrt(Gb + eps) * gb
        hist.append(cross_entropy(X, w, b, Y))
    return w, b, hist


def adam(X, Y, w0, b0, alpha, b1, b2, eps, max_it):
    w, b = list(w0), b0
    m = [0.0] * len(w)
    v = [0.0] * len(w)
    mb, vb = 0.0, 0.0
    hist = [cross_entropy(X, w, b, Y)]
    for t in range(1, max_it + 1):
        gw, gb = grad(X, w, b, Y)
        m = [b1 * m[j] + (1 - b1) * gw[j] for j in range(len(w))]
        v = [b2 * v[j] + (1 - b2) * gw[j] * gw[j] for j in range(len(w))]
        mb = b1 * mb + (1 - b1) * gb
        vb = b2 * vb + (1 - b2) * gb * gb
        mh = [m[j] / (1 - b1 ** t) for j in range(len(w))]
        vh = [v[j] / (1 - b2 ** t) for j in range(len(w))]
        w = [w[j] - alpha * mh[j] / (math.sqrt(vh[j]) + eps) for j in range(len(w))]
        mbh = mb / (1 - b1 ** t)
        vbh = vb / (1 - b2 ** t)
        b = b - alpha * mbh / (math.sqrt(vbh) + eps)
        hist.append(cross_entropy(X, w, b, Y))
    return w, b, hist


def worked_gd_sgd(out, lg):
    X = [[1.0, 2.0], [3.0, 4.0]]
    Y = [0.0, 1.0]
    w0, b0, alpha = [0.5, 0.5], 1.0, 0.01
    lg(f"Logistic worked example X={X} Y={Y} w0={w0} b0={b0} alpha={alpha}")
    w, b = list(w0), b0
    for k in range(1, 3):
        P = predict(X, w, b)
        E = [p - y for p, y in zip(P, Y)]
        gw, gb = grad(X, w, b, Y)
        lg(f"GD iter {k}: P={P} E={E} loss={cross_entropy(X, w, b, Y):.6f} gw={gw} gb={gb}")
        w = sub(w, scale(alpha, gw))
        b = b - alpha * gb
        lg(f"GD iter {k}: w={w} b={b}")
    w_s, b_s, rows = sgd_fixed(X, Y, w0, b0, alpha, [0, 1])
    pw, pb = w0, b0
    for i, p, e, wn, bn in rows:
        lg(f"SGD pick row {i}: p={p:.6f} e={e:.6f} loss={cross_entropy(X, pw, pb, Y):.6f} w={wn} b={bn:.6f}")
        pw, pb = wn, bn


def worked_adagrad(out, lg):
    X = [[1.0, 0.2], [3.0, 0.01]]
    Y = [0.0, 1.0]
    w0, b0 = [0.5, 0.5], 1.0
    alpha, eps = 0.01, 1e-8
    lg(f"Adagrad worked example X={X} Y={Y} w0={w0} b0={b0} alpha={alpha} eps={eps}")

    def trace(k, gw, gb, G, Gb, w, b):
        lg(f"Adagrad iter {k}: gw={gw} gb={gb:.8f} G={G} Gb={Gb:.8f} w={w} b={b:.8f}")

    adagrad(X, Y, w0, b0, alpha, eps, 2, trace=trace)


def synthetic_sparse(N, d, seed):
    random.seed(seed)
    true_w = [random.uniform(-2, 2) for _ in range(d)]
    X = []
    for _ in range(N):
        row = [random.gauss(0, 1) if random.random() < 0.2 else 0.0 for _ in range(d)]
        X.append(row)
    Y = []
    for x in X:
        z = dot(x, true_w) + 0.3
        Y.append(1.0 if random.random() < sigmoid(z) else 0.0)
    return X, Y, true_w


def compare_optimizers(out, lg):
    N, d = 300, 6
    X, Y, _ = synthetic_sparse(N, d, cfg.SEED)
    w0, b0 = [0.0] * d, 0.0
    runs = {
        "GD alpha=0.5": gd(X, Y, w0, b0, 0.5, 300),
        "Adagrad alpha=0.5": adagrad(X, Y, w0, b0, 0.5, 1e-8, 300),
        "RMSProp alpha=0.01": rmsprop(X, Y, w0, b0, 0.01, 0.9, 1e-8, 300),
        "Adam alpha=0.01": adam(X, Y, w0, b0, 0.01, 0.9, 0.999, 1e-8, 300),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (w, b, hist) in runs.items():
        lg(f"{name}: final loss={hist[-1]:.6f} w={w} b={b:.6f}")
        ax.plot(range(len(hist)), hist, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Logistic regression: GD vs adaptive methods (synthetic sparse data)")
    ax.legend()
    save_plot(fig, out, "l4_logistic_optimizers.png")


def main(out):
    log_path = os.path.join(out, "log.txt")
    set_seed(cfg.SEED, log_path)

    def lg(msg):
        log(log_path, msg)

    worked_gd_sgd(out, lg)
    worked_adagrad(out, lg)
    compare_optimizers(out, lg)


if __name__ == "__main__":
    main(run_dir())
