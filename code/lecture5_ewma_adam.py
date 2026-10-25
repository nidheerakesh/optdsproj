import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

from common import dot, scale, mat_vec, transpose, log, set_seed, save_plot, run_dir
import config as cfg


def ewma(xs, beta):
    out = []
    m = 0.0
    for x in xs:
        m = beta * x + (1 - beta) * m
        out.append(m)
    return out


def ewma_weights(beta, t):
    return [beta * (1 - beta) ** k for k in range(t)]


def adam_fn(f, g, x0, alpha, b1, b2, eps, max_it, lg=None):
    x = list(x0)
    n = len(x)
    M = [0.0] * n
    G = [0.0] * n
    hist = [f(x)]
    for k in range(1, max_it + 1):
        gr = g(x)
        M = [b1 * M[j] + (1 - b1) * gr[j] for j in range(n)]
        G = [b2 * G[j] + (1 - b2) * gr[j] * gr[j] for j in range(n)]
        Mh = [M[j] / (1 - b1 ** k) for j in range(n)]
        Gh = [G[j] / (1 - b2 ** k) for j in range(n)]
        x = [x[j] - alpha * Mh[j] / (math.sqrt(Gh[j]) + eps) for j in range(n)]
        hist.append(f(x))
        if lg is not None:
            lg(f"Adam f-example iter {k}: grad={gr} M={M} G={G} Mhat={Mh} Ghat={Gh} x={x} f={hist[-1]:.8f}")
    return x, hist


def rmsprop_fn(f, g, x0, alpha, beta, eps, max_it, lg=None):
    x = list(x0)
    n = len(x)
    G = [0.0] * n
    hist = [f(x)]
    for k in range(1, max_it + 1):
        gr = g(x)
        G = [beta * G[j] + (1 - beta) * gr[j] * gr[j] for j in range(n)]
        x = [x[j] - alpha * gr[j] / (math.sqrt(G[j]) + eps) for j in range(n)]
        hist.append(f(x))
        if lg is not None:
            lg(f"RMSProp f-example iter {k}: grad={gr} G={G} x={x} f={hist[-1]:.8f}")
    return x, hist


def gd_fn(f, g, x0, alpha, max_it):
    x = list(x0)
    hist = [f(x)]
    for _ in range(max_it):
        gr = g(x)
        x = [x[j] - alpha * gr[j] for j in range(len(x))]
        hist.append(f(x))
    return x, hist


def f_ex(x):
    return x[0] ** 2 + 2 * x[1] ** 2


def g_ex(x):
    return [2 * x[0], 4 * x[1]]


def linreg_loss_grad(X, Y, w, b):
    N = len(Y)
    E = [dot(x, w) + b - y for x, y in zip(X, Y)]
    s = 0.0
    se = 0.0
    for e in E:
        s += e * e
        se += e
    loss = s / (2 * N)
    gw = scale(1.0 / N, mat_vec(transpose(X), E))
    gb = se / N
    return loss, gw, gb


def adam_linreg(X, Y, w0, b0, alpha, b1, b2, eps, max_it, lg=None):
    w, b = list(w0), b0
    n = len(w)
    M = [0.0] * n
    G = [0.0] * n
    Mb, Gb = 0.0, 0.0
    hist = []
    for k in range(1, max_it + 1):
        loss, gw, gb = linreg_loss_grad(X, Y, w, b)
        hist.append(loss)
        M = [b1 * M[j] + (1 - b1) * gw[j] for j in range(n)]
        G = [b2 * G[j] + (1 - b2) * gw[j] * gw[j] for j in range(n)]
        Mb = b1 * Mb + (1 - b1) * gb
        Gb = b2 * Gb + (1 - b2) * gb * gb
        Mh = [M[j] / (1 - b1 ** k) for j in range(n)]
        Gh = [G[j] / (1 - b2 ** k) for j in range(n)]
        Mbh = Mb / (1 - b1 ** k)
        Gbh = Gb / (1 - b2 ** k)
        w = [w[j] - alpha * Mh[j] / (math.sqrt(Gh[j]) + eps) for j in range(n)]
        b = b - alpha * Mbh / (math.sqrt(Gbh) + eps)
        if lg is not None:
            lg(f"Adam linreg iter {k}: loss={loss:.6f} gw={gw} gb={gb:.6f} w={w} b={b:.6f}")
    final_loss, _, _ = linreg_loss_grad(X, Y, w, b)
    hist.append(final_loss)
    return w, b, hist


def ewma_demo(out, lg):
    random.seed(cfg.SEED)
    t = 60
    true = [math.sin(0.15 * i) for i in range(t)]
    xs = [true[i] + random.gauss(0, 0.4) for i in range(t)]
    lg(f"EWMA demo t={t} seed={cfg.SEED}")
    for beta in [0.1, 0.3, 0.9]:
        lg(f"beta={beta}: first weights={[round(w, 4) for w in ewma_weights(beta, 4)]} sum_{t}={sum(ewma_weights(beta, t)):.6f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ".", color="0.6", label="noisy x_t")
    for beta in [0.1, 0.3, 0.9]:
        ax.plot(ewma(xs, beta), label=f"EWMA beta={beta}")
    ax.set_xlabel("t")
    ax.set_title("EWMA smoothing")
    ax.legend()
    save_plot(fig, out, "l5_ewma_smoothing.png")


def adam_examples(out, lg):
    x0 = [1.0, -1.0]
    lg(f"Adam example f=x1^2+2x2^2 x0={x0} alpha=0.01 b1=0.9 b2=0.99 eps=1e-8, 2 iterations")
    adam_fn(f_ex, g_ex, x0, 0.01, 0.9, 0.99, 1e-8, 2, lg=lg)
    X = [[1.0, 2.0], [2.0, 5.0]]
    Y = [3.0, 7.0]
    lg(f"Adam linreg example X={X} Y={Y} w0=[0,0] b0=1 alpha=0.01 b1=0.9 b2=0.99 eps=1e-8, 2 iterations")
    adam_linreg(X, Y, [0.0, 0.0], 1.0, 0.01, 0.9, 0.99, 1e-8, 2, lg=lg)


def adam_vs_gd(out, lg):
    x0 = [1.0, -1.0]
    xa, ha = adam_fn(f_ex, g_ex, x0, 0.05, 0.9, 0.99, 1e-8, 300)
    xg, hg = gd_fn(f_ex, g_ex, x0, 0.1, 300)
    lg(f"f=x1^2+2x2^2 after 300 iters: Adam f={ha[-1]:.3e} x={xa}; GD f={hg[-1]:.3e} x={xg}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(ha)), [max(v, 1e-300) for v in ha], label="Adam alpha=0.05")
    ax.semilogy(range(len(hg)), [max(v, 1e-300) for v in hg], label="GD alpha=0.1")
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x)")
    ax.legend()
    save_plot(fig, out, "l5_adam_vs_gd.png")


def rmsprop_examples(out, lg):
    x0 = [1.0, -1.0]
    lg(f"RMSProp example f=x1^2+2x2^2 x0={x0} alpha=0.01 beta=0.9 eps=1e-8, 2 iterations (slide gives no numbers for RMSProp)")
    rmsprop_fn(f_ex, g_ex, x0, 0.01, 0.9, 1e-8, 2, lg=lg)
    xr, hr = rmsprop_fn(f_ex, g_ex, x0, 0.05, 0.9, 1e-8, 300)
    xa, ha = adam_fn(f_ex, g_ex, x0, 0.05, 0.9, 0.99, 1e-8, 300)
    lg(f"RMSProp f=x1^2+2x2^2 after 300 iters alpha=0.05 beta=0.9: f={hr[-1]:.3e} x={xr}; Adam same alpha: f={ha[-1]:.3e} x={xa}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(hr)), [max(v, 1e-300) for v in hr], label="RMSProp alpha=0.05")
    ax.semilogy(range(len(ha)), [max(v, 1e-300) for v in ha], label="Adam alpha=0.05")
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x)")
    ax.legend()
    save_plot(fig, out, "l5_rmsprop_vs_adam.png")


def main(out):
    log_path = os.path.join(out, "log.txt")
    set_seed(cfg.SEED, log_path)

    def lg(msg):
        log(log_path, msg)

    ewma_demo(out, lg)
    adam_examples(out, lg)
    adam_vs_gd(out, lg)
    rmsprop_examples(out, lg)


if __name__ == "__main__":
    main(run_dir())
