import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

from common import add, sub, scale, norm, log, set_seed, save_plot, run_dir, solve
import config as cfg
from lecture2_gradient_descent import rosen, rosen_grad, gd_fixed


def f1(p):
    x, y = p
    return x * x + 2 * x * y + 3 * y * y - 6 * x - 12 * y


def g1(p):
    x, y = p
    return [2 * x + 2 * y - 6, 2 * x + 6 * y - 12]


def h1(p):
    return [[2.0, 2.0], [2.0, 6.0]]


def f2(p):
    x, y = p
    return x * x + 2 * x * y + y * y


def g2(p):
    x, y = p
    return [2 * x + 2 * y, 2 * x + 2 * y]


def h2(p):
    return [[2.0, 2.0], [2.0, 2.0]]


def rosen_hess(p):
    x, y = p
    return [[2 - 400 * (y - x * x) + 800 * x * x, -400 * x], [-400 * x, 200.0]]


def newton(f, g, h, x0, lam, max_it, eps=cfg.EPS):
    x = list(x0)
    hist = [f(x)]
    path = [list(x)]
    n = len(x)
    for _ in range(max_it):
        gr = g(x)
        if norm(gr) <= eps:
            break
        H = h(x)
        Hd = [[H[i][j] + (lam if i == j else 0.0) for j in range(n)] for i in range(n)]
        d = solve(Hd, scale(-1.0, gr))
        x = add(x, d)
        hist.append(f(x))
        path.append(list(x))
    return x, path, hist


def cond(H):
    a, b, c, d = H[0][0], H[0][1], H[1][0], H[1][1]
    tr = a + d
    det = a * d - b * c
    disc = (tr * tr / 4 - det) ** 0.5
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    return max(abs(l1), abs(l2)) / min(abs(l1), abs(l2))


def root_1d(f, fp, x0, max_it=50, eps=cfg.EPS):
    x = x0
    hist = [x]
    for _ in range(max_it):
        fx = f(x)
        if abs(fx) <= eps:
            break
        x = x - fx / fp(x)
        hist.append(x)
    return x, hist


def opt_1d(fp, fpp, x0, max_it=50, eps=cfg.EPS):
    x = x0
    hist = [x]
    for _ in range(max_it):
        g = fp(x)
        if abs(g) <= eps:
            break
        x = x - g / fpp(x)
        hist.append(x)
    return x, hist


def oned(out, lg):
    xr, hr = root_1d(lambda x: x * x - 2, lambda x: 2 * x, 1.0)
    lg(f"1D Newton root f=x^2-2 from x0=1: iters={len(hr)-1} root={xr} f={xr*xr-2:.3e} iterates={hr}")
    xo, ho = opt_1d(lambda x: 4 * x ** 3 - 9 * x * x, lambda x: 12 * x * x - 18 * x, 3.0)
    lg(f"1D Newton optimum f=x^4-3x^3+2 from x0=3: iters={len(ho)-1} x*={xo} f(x*)={xo**4-3*xo**3+2:.6f} iterates={ho}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(hr)), hr, "o-", label="root of x^2-2")
    ax.plot(range(len(ho)), ho, "s-", label="optimum of x^4-3x^3+2")
    ax.axhline(2 ** 0.5, ls=":", c="gray")
    ax.axhline(2.25, ls=":", c="gray")
    ax.set_xlabel("iteration")
    ax.set_ylabel("x")
    ax.legend()
    ax.set_title("1D Newton iterates")
    save_plot(fig, out, "l7_newton_1d.png")


def worked(out, lg):
    x1, p1, h1s = newton(f1, g1, h1, [0.0, 0.0], 0.0, 1)
    lg(f"Example 1: f=x^2+2xy+3y^2-6x-12y, Newton from (0,0): x1={x1} f(x1)={f1(x1):.6f}")
    x2, p2, h2s = newton(f2, g2, h2, [1.0, 0.0], 1.0, 1)
    lg(f"Example 2: f=x^2+2xy+y^2 singular H, damped lambda=1 from (1,0): x1={x2} f(x1)={f2(x2):.6f}")
    H_ill = [[1.0, 0.0], [0.0, 0.001]]
    H_damp = [[1.1, 0.0], [0.0, 0.101]]
    lg(f"Conditioning: cond(diag(1,0.001))={cond(H_ill):.2f}; cond(H+0.1I)={cond(H_damp):.2f}")


def rosenbrock(out, lg):
    x0 = cfg.L2["rosen_x0"]
    xn, pn, hn = newton(rosen, rosen_grad, rosen_hess, x0, 0.0, 200)
    lg(f"Newton (lambda=0) on Rosenbrock from {x0}: iters={len(hn)-1} final_x={xn} f={rosen(xn):.3e}")
    xd, pd, hd = newton(rosen, rosen_grad, rosen_hess, x0, 1.0, 2000)
    lg(f"Damped Newton (lambda=1) on Rosenbrock from {x0}: iters={len(hd)-1} final_x={xd} f={rosen(xd):.3e}")
    xg, pg, hg = gd_fixed(rosen, rosen_grad, x0, cfg.L2["rosen_alpha"], max_it=cfg.MAX_IT)
    lg(f"GD fixed alpha={cfg.L2['rosen_alpha']} on Rosenbrock: iters={len(hg)-1} final_f={hg[-1]:.3e}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(hn)), [max(v, 1e-16) for v in hn], label="Newton lambda=0")
    ax.semilogy(range(len(hd)), [max(v, 1e-16) for v in hd], label="damped Newton lambda=1")
    ax.semilogy(range(len(hg)), [max(v, 1e-16) for v in hg], label="GD fixed alpha")
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x)")
    ax.set_xlim(0, 300)
    ax.legend()
    ax.set_title("Rosenbrock: Newton vs damped Newton vs GD")
    save_plot(fig, out, "l7_newton_rosenbrock.png")


def main(out):
    log_path = os.path.join(out, "log.txt")
    set_seed(cfg.SEED, log_path)

    def lg(msg):
        log(log_path, msg)

    worked(out, lg)
    oned(out, lg)
    rosenbrock(out, lg)


if __name__ == "__main__":
    main(run_dir())
