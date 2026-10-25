import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

from common import dot, add, sub, scale, norm, log, set_seed, save_plot, write_table
import config as cfg


def quad(a, b):
    def f(x):
        return 0.5 * (a * x[0] ** 2 + b * x[1] ** 2)

    def g(x):
        return [a * x[0], b * x[1]]

    return f, g


def sq_cross(x):
    return 4 * x[0] ** 2 + x[1] ** 2 - 2 * x[0] * x[1]


def sq_cross_grad(x):
    return [8 * x[0] - 2 * x[1], 2 * x[1] - 2 * x[0]]


def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return [
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
        200 * (x[1] - x[0] ** 2),
    ]


def gd_fixed(f, g, x0, alpha, eps=cfg.EPS, max_it=cfg.MAX_IT):
    x = list(x0)
    path, vals = [list(x)], [f(x)]
    k = 0
    while norm(g(x)) > eps and k < max_it:
        x = sub(x, scale(alpha, g(x)))
        path.append(list(x))
        vals.append(f(x))
        k += 1
    return x, path, vals


def golden(phi, lo, hi, tol=1e-10):
    r = (5 ** 0.5 - 1) / 2
    c = hi - r * (hi - lo)
    d = lo + r * (hi - lo)
    while abs(hi - lo) > tol:
        if phi(c) < phi(d):
            hi = d
        else:
            lo = c
        c = hi - r * (hi - lo)
        d = lo + r * (hi - lo)
    return (lo + hi) / 2


def exact_alpha(f, x, d):
    def phi(a):
        return f(add(x, scale(a, d)))

    return golden(phi, 0.0, 1.0)


def gd_exact(f, g, x0, eps=cfg.EPS, max_it=cfg.MAX_IT):
    x = list(x0)
    path, vals = [list(x)], [f(x)]
    k = 0
    while norm(g(x)) > eps and k < max_it:
        d = scale(-1.0, g(x))
        a = exact_alpha(f, x, d)
        x = add(x, scale(a, d))
        path.append(list(x))
        vals.append(f(x))
        k += 1
    return x, path, vals


def backtrack(f, g, x, d, alpha_init, beta, c1):
    a = alpha_init
    gd = dot(g(x), d)
    while f(add(x, scale(a, d))) > f(x) + c1 * a * gd:
        a = beta * a
    return a


def gd_armijo(f, g, x0, eps=cfg.EPS, max_it=cfg.MAX_IT):
    x = list(x0)
    path, vals = [list(x)], [f(x)]
    k = 0
    while norm(g(x)) > eps and k < max_it:
        d = scale(-1.0, g(x))
        a = backtrack(f, g, x, d, cfg.L2["alpha_init"], cfg.L2["beta"], cfg.L2["c1"])
        x = add(x, scale(a, d))
        path.append(list(x))
        vals.append(f(x))
        k += 1
    return x, path, vals


def momentum(f, g, x0, alpha, gamma, eps=cfg.EPS, max_it=cfg.MAX_IT):
    x = list(x0)
    v = [0.0] * len(x)
    path, vals = [list(x)], [f(x)]
    k = 0
    while norm(g(x)) > eps and k < max_it:
        v = add(scale(gamma, v), scale(alpha, g(x)))
        x = sub(x, v)
        path.append(list(x))
        vals.append(f(x))
        k += 1
    return x, path, vals


def nesterov(f, g, x0, alpha, gamma, eps=cfg.EPS, max_it=cfg.MAX_IT):
    x = list(x0)
    v = [0.0] * len(x)
    path, vals = [list(x)], [f(x)]
    k = 0
    while norm(g(x)) > eps and k < max_it and norm(x) < 1e50:
        look = sub(x, scale(gamma, v))
        v = add(scale(gamma, v), scale(alpha, g(look)))
        x = sub(x, v)
        path.append(list(x))
        vals.append(f(x))
        k += 1
    return x, path, vals


def run_conditioning(out, lg):
    rows = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in cfg.L2["conds"]:
        f, g = quad(a, 1.0)
        alpha = cfg.L2["alphas"][int(a)]
        xs, path, vals = gd_fixed(f, g, cfg.L2["x0"], alpha)
        lg(f"quad a={a:g} b=1 alpha={alpha} iters={len(vals)-1} final_x={xs} final_f={vals[-1]:.3e}")
        rows.append([a, alpha, len(vals) - 1, vals[-1]])
        ax.semilogy(range(len(vals)), vals, label=f"a={int(a)}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x)")
    ax.set_title("GD on f=0.5(a x1^2 + x2^2), fixed alpha")
    ax.legend()
    save_plot(fig, out, "l2_conditioning_convergence.png")
    write_table(os.path.join(out, "l2_conditioning.csv"), ["a", "alpha", "iters", "final_f"], rows)


def run_methods(out, lg):
    fig, ax = plt.subplots(figsize=(6, 4))
    x0 = cfg.L2["x0"]
    runs = [
        (f"GD fixed alpha={cfg.L2['fixed_alpha']}", gd_fixed(sq_cross, sq_cross_grad, x0, cfg.L2["fixed_alpha"])),
        ("GD exact line search", gd_exact(sq_cross, sq_cross_grad, x0)),
        ("GD Armijo backtracking", gd_armijo(sq_cross, sq_cross_grad, x0)),
    ]
    for name, (xs, path, vals) in runs:
        lg(f"4x1^2+x2^2-2x1x2 {name}: iters={len(vals)-1} final_x={xs} final_f={vals[-1]:.3e}")
        ax.semilogy(range(len(vals)), [v - vals[-1] + 1e-16 for v in vals], label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x) - f(x*)")
    ax.legend()
    save_plot(fig, out, "l2_line_search_comparison.png")


def run_rosen(out, lg):
    x0 = cfg.L2["rosen_x0"]
    xs, path, vals = gd_fixed(rosen, rosen_grad, x0, cfg.L2["rosen_alpha"], max_it=cfg.MAX_IT)
    st = "converged" if norm(rosen_grad(xs)) <= cfg.EPS else "hit max_it"
    lg(f"Rosenbrock GD fixed alpha={cfg.L2['rosen_alpha']}: {st} iters={len(vals)-1} final_x={xs} final_f={vals[-1]:.3e}")
    xs2, path2, vals2 = gd_armijo(rosen, rosen_grad, x0, max_it=cfg.MAX_IT)
    st2 = "converged" if norm(rosen_grad(xs2)) <= cfg.EPS else "hit max_it"
    lg(f"Rosenbrock GD Armijo: {st2} iters={len(vals2)-1} final_x={xs2} final_f={vals2[-1]:.3e}")
    fig, ax = plt.subplots(figsize=(6, 5))
    xg = [-1.5 + 0.02 * i for i in range(101)]
    yg = [-0.5 + 0.03 * j for j in range(101)]
    z = [[rosen([xv, yv]) for xv in xg] for yv in yg]
    ax.contour(xg, yg, z, levels=[0.5, 2, 5, 10, 25, 60, 150, 400], cmap="viridis")
    ax.plot([p[0] for p in path], [p[1] for p in path], "r.-", ms=3, label="GD fixed")
    ax.plot([p[0] for p in path2], [p[1] for p in path2], "b.-", ms=3, label="GD Armijo")
    ax.plot(1, 1, "k*", ms=12, label="minimizer")
    ax.legend()
    ax.set_title("Rosenbrock: path of iterates")
    save_plot(fig, out, "l2_rosenbrock_paths.png")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(range(len(vals)), vals, label="GD fixed")
    ax2.semilogy(range(len(vals2)), vals2, label="GD Armijo")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("f(x)")
    ax2.legend()
    save_plot(fig2, out, "l2_rosenbrock_convergence.png")


def run_momentum(out, lg):
    x0 = cfg.L2["x0"]
    f, g = quad(100.0, 1.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, fn in [
        ("GD", lambda: gd_fixed(f, g, x0, cfg.L2["alphas"][100])),
        ("Momentum", lambda: momentum(f, g, x0, cfg.L2["alphas"][100], cfg.L2["gamma"])),
        ("Nesterov", lambda: nesterov(f, g, x0, cfg.L2["alphas"][100], cfg.L2["gamma"])),
    ]:
        xs, path, vals = fn()
        status = "diverged" if norm(xs) >= 1e50 else "converged"
        lg(f"a=100 {name}: {status} iters={len(vals)-1} final_x={xs} final_f={vals[-1]:.3e}")
        ax.semilogy(range(len(vals)), vals, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x)")
    ax.set_title("GD vs momentum vs Nesterov, a=100")
    ax.legend()
    save_plot(fig, out, "l2_momentum_comparison.png")


def main(out):
    log_path = os.path.join(out, "log.txt")
    set_seed(cfg.SEED, log_path)

    def lg(msg):
        log(log_path, msg)

    run_conditioning(out, lg)
    run_methods(out, lg)
    run_rosen(out, lg)
    run_momentum(out, lg)


if __name__ == "__main__":
    from common import run_dir
    main(run_dir())
