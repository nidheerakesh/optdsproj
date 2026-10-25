import os
import sys
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)

from common import dot, add, sub, scale, norm, mat_vec, log, set_seed, save_plot, write_table, solve, run_dir
import config as cfg

L1 = {
    "a": [1.0, 2.0],
    "b": [4.0, 6.0],
    "vec": [3.0, -4.0, -6.0],
    "unit_src": [3.0, 4.0],
    "dots": [
        ([2.0, 3.0, 4.0], [4.0, 1.0, 2.0]),
        ([2.0, 3.0, 4.0], [2.0, 4.0, 2.0]),
        ([1.0, 2.0, -3.0], [-2.0, 1.0, 2.0]),
        ([1.0, 2.0, 0.0], [-2.0, 1.0, 1.0]),
    ],
    "grad_pts": [[2.0, 3.0], [3.0, 1.0]],
    "dd_pt": [1.0, 2.0],
    "dd_dir": [3.0, 4.0],
    "sweep_n": 720,
    "fd_h": 1e-6,
    "hess_h": 1e-4,
    "ball_delta": 0.3,
    "ball_n": 500,
    "knap_w": [10.0, 20.0, 30.0],
    "knap_v": [60.0, 100.0, 120.0],
    "knap_cap": 50.0,
    "cons_lo": -2.0,
    "cons_hi": 2.0,
    "cons_m": 201,
    "peaks_lo": -3.0,
    "peaks_hi": 3.0,
    "peaks_m": 121,
    "peaks_surf_m": 61,
    "peaks_starts": [
        [-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0],
        [0.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [0.5, -1.5],
    ],
    "gd_step": 0.02,
    "gd_eps": 1e-7,
    "gd_max_it": 20000,
    "taylor_x": 1.0,
    "taylor_orders": [1, 3, 5],
    "taylor_alphas": [0.1, 0.05, 0.025, 0.0125],
}


def l1n(a):
    s = 0.0
    for v in a:
        s += abs(v)
    return s


def l1d(x, y):
    return l1n(sub(x, y))


def l2d(x, y):
    return norm(sub(x, y))


def grad_fd(f, x, h):
    g = []
    for i in range(len(x)):
        xp = list(x)
        xm = list(x)
        xp[i] += h
        xm[i] -= h
        g.append((f(xp) - f(xm)) / (2 * h))
    return g


def shifted(x, i, si, j, sj, h):
    y = list(x)
    y[i] += si * h
    y[j] += sj * h
    return y


def hess_fd(f, x, h):
    n = len(x)
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                yp = shifted(x, i, 1, i, 0, h)
                ym = shifted(x, i, -1, i, 0, h)
                H[i][i] = (f(yp) - 2 * f(x) + f(ym)) / (h * h)
            else:
                pp = shifted(x, i, 1, j, 1, h)
                pm = shifted(x, i, 1, j, -1, h)
                mp = shifted(x, i, -1, j, 1, h)
                mm = shifted(x, i, -1, j, -1, h)
                H[i][j] = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * h * h)
    return H


def dir_deriv_fd(f, x, d, h):
    return (f(add(x, scale(h, d))) - f(x)) / h


def eig2_closed(H):
    tr = H[0][0] + H[1][1]
    det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    disc = (tr * tr - 4 * det) ** 0.5
    return sorted([(tr - disc) / 2, (tr + disc) / 2])


def eig_sym(A, tol=1e-14, max_sweep=100):
    n = len(A)
    a = [[float(v) for v in row] for row in A]
    for _ in range(max_sweep):
        off = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    off += a[i][j] * a[i][j]
        if off < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if a[p][q] == 0.0:
                    continue
                th = (a[q][q] - a[p][p]) / (2 * a[p][q])
                sg = 1.0 if th >= 0 else -1.0
                t = sg / (abs(th) + (th * th + 1) ** 0.5)
                c = 1 / (t * t + 1) ** 0.5
                s = t * c
                for k in range(n):
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
    return sorted(a[i][i] for i in range(n))


def classify(eigs, tol=1e-9):
    if all(abs(e) < tol for e in eigs):
        return "zero matrix (PSD and NSD)"
    if all(e > tol for e in eigs):
        return "positive definite (strictly convex)"
    if all(e >= -tol for e in eigs):
        return "positive semi-definite (convex)"
    if all(e < -tol for e in eigs):
        return "negative definite (strictly concave)"
    if all(e <= tol for e in eigs):
        return "negative semi-definite (concave)"
    return "indefinite (saddle)"


def quad_form(H, x):
    return dot(x, mat_vec(H, x))


def fmt(v):
    return "[" + ", ".join(f"{t:.6g}" for t in v) + "]"


def f_grad_ex(x):
    return 3 * x[0] ** 2 + 2 * x[0] * x[1] + x[1] ** 2


def g_grad_ex(x):
    return [6 * x[0] + 2 * x[1], 2 * x[0] + 2 * x[1]]


def f_sq(x):
    return x[0] ** 2 + x[1] ** 2


def g_sq(x):
    return [2 * x[0], 2 * x[1]]


def f_cubic(x):
    return x[0] ** 3 - 3 * x[0] * x[1] ** 2


def g_cubic(x):
    return [3 * x[0] ** 2 - 3 * x[1] ** 2, -6 * x[0] * x[1]]


def h_cubic(x):
    return [[6 * x[0], -6 * x[1]], [-6 * x[1], -6 * x[0]]]


def peaks(x, lit=False):
    x1, x2 = x[0], x[1]
    t1 = 3 * (1 - x1) ** 2 * math.exp(-(x1 ** 2) - (x2 + 1) ** 2)
    s = (x1 ** 2 - x2 ** 2) if lit else (x1 ** 2 + x2 ** 2)
    t2 = -10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * math.exp(-s)
    t3 = -(1.0 / 3.0) * math.exp(-((x1 + 1) ** 2) - x2 ** 2)
    return t1 + t2 + t3


def run_norms(out, lg):
    a, b = L1["a"], L1["b"]
    lg(f"norm: x={a}, y={b}, L1 distance={l1d(a, b):.6g}, L2 distance={l2d(a, b):.6g} (slide: 7 and 5)")
    v = L1["vec"]
    lg(f"norm: a={v}, ||a||_1={l1n(v):.6g} (slide: 13), ||a||_2={norm(v):.6g} (sqrt(61))")
    u = scale(1.0 / norm(L1["unit_src"]), L1["unit_src"])
    lg(f"unit vector of {L1['unit_src']} = {fmt(u)} (slide: (0.6, 0.8))")
    ts = [2 * math.pi * i / 400 for i in range(401)]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([math.cos(t) for t in ts], [math.sin(t) for t in ts], label="L2 unit ball")
    xs = [-1 + 2 * i / 200 for i in range(201)]
    ax.plot(xs, [1 - abs(x) for x in xs], "r", label="L1 unit ball")
    ax.plot(xs, [-(1 - abs(x)) for x in xs], "r")
    ax.set_aspect("equal")
    ax.set_title("Unit balls of L1 and L2 norms")
    ax.legend()
    save_plot(fig, out, "l1_l2_unit_balls.png")


def run_dot_cosine(lg):
    for a, b in L1["dots"]:
        d = dot(a, b)
        cs = d / (norm(a) * norm(b))
        th = math.degrees(math.acos(max(-1.0, min(1.0, cs))))
        lg(f"dot({a}, {b}) = {d:.6g}, cosine similarity = {cs:.6g}, angle = {th:.4g} deg")


def run_single_var(lg):
    a, b, x = 2.0, 1.0, 3.0
    lg(f"f=ax+b, a=2, b=1 at x=3: f={a * x + b:.6g}")
    a2, b2 = 1.0, -2.0
    f2 = lambda t: a2 * t * t + b2 * t
    lg(f"f=ax^2+bx, a=1, b=-2 at x=3: f={f2(x):.6g}, FD f'={(f2(x + 1e-6) - f2(x - 1e-6)) / 2e-6:.6g} (analytic 2ax+b = {2 * a2 * x + b2:.6g})")
    fm = lambda t: max(0.0, t)
    lg(f"f=max(0,x): f(-1)={fm(-1.0):.6g}, f(2)={fm(2.0):.6g}, right derivative at 2 = {(fm(2.0 + 1e-6) - fm(2.0)) / 1e-6:.6g}")


def run_gradients(out, lg):
    h = L1["fd_h"]
    for p in L1["grad_pts"]:
        lg(f"grad f=3x^2+2xy+y^2 at {p}: analytic={fmt(g_grad_ex(p))}, FD={fmt(grad_fd(f_grad_ex, p, h))}")
    x = L1["dd_pt"]
    d = L1["dd_dir"]
    u = scale(1.0 / norm(d), d)
    g = g_sq(x)
    lg(f"f=x^2+y^2 at {x}: grad={fmt(g)}, unit d={fmt(u)}")
    lg(f"directional derivative analytic grad.u = {dot(g, u):.6g} (slide: 4.4), FD = {dir_deriv_fd(f_sq, x, u, h):.6g}")
    n = L1["sweep_n"]
    angs = [2 * math.pi * i / n for i in range(n)]
    dd = [dir_deriv_fd(f_sq, x, [math.cos(t), math.sin(t)], h) for t in angs]
    k = max(range(n), key=lambda i: dd[i])
    lg(f"sweep over {n} directions: max directional derivative {dd[k]:.6g} at angle {angs[k]:.5f} rad; "
       f"grad direction angle {math.atan2(g[1], g[0]):.5f} rad, |grad|={norm(g):.6g}")
    dsc = scale(-1.0, g)
    lg(f"descent check: grad.(-grad) = {dot(g, dsc):.6g} < 0, so -grad is a descent direction")
    fig, ax = plt.subplots(figsize=(6, 5))
    xg = [-2 + 4 * i / 60 for i in range(61)]
    z = [[f_sq([xv, yv]) for xv in xg] for yv in xg]
    ax.contour(xg, xg, z, levels=12)
    ax.quiver(x[0], x[1], g[0], g[1], color="r", angles="xy", scale_units="xy", scale=1, label="grad f(1,2)")
    ax.quiver(x[0], x[1], u[0], u[1], color="b", angles="xy", scale_units="xy", scale=1, label="unit d")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("f=x^2+y^2: gradient (red) and direction d (blue)")
    save_plot(fig, out, "l1_directional_derivative_contour.png")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(angs, dd, label="D_u f(x) by finite difference")
    ax2.axhline(norm(g), color="r", ls="--", label="|grad f| (max)")
    ax2.set_xlabel("direction angle (rad)")
    ax2.set_ylabel("directional derivative")
    ax2.legend()
    ax2.set_title("Directional derivative peaks along the gradient")
    save_plot(fig2, out, "l1_directional_ascent_sweep.png")


def run_hessians(out, lg):
    H = L1["hess_h"]
    cubic_h = lambda p: h_cubic(p)
    ex = [
        ("3x1^2+5x2^2 (PD example)", lambda x: 3 * x[0] ** 2 + 5 * x[1] ** 2, lambda p: [[6.0, 0.0], [0.0, 10.0]], [1.0, 1.0]),
        ("6x1^2 (PSD example, slide H=diag(6,0))", lambda x: 6 * x[0] ** 2, lambda p: [[12.0, 0.0], [0.0, 0.0]], [1.0, 1.0]),
        ("-x1^2-x2^2 (ND example)", lambda x: -x[0] ** 2 - x[1] ** 2, lambda p: [[-2.0, 0.0], [0.0, -2.0]], [1.0, 1.0]),
        ("-(x1+x2)^2 (NSD example)", lambda x: -(x[0] + x[1]) ** 2, lambda p: [[-2.0, -2.0], [-2.0, -2.0]], [1.0, 1.0]),
        ("x1^2-x2^2 (indefinite example)", lambda x: x[0] ** 2 - x[1] ** 2, lambda p: [[2.0, 0.0], [0.0, -2.0]], [1.0, 1.0]),
        ("2x^2+3xy+y^2 (quadratic form example)", lambda x: 2 * x[0] ** 2 + 3 * x[0] * x[1] + x[1] ** 2, lambda p: [[4.0, 3.0], [3.0, 2.0]], [1.0, 1.0]),
        ("5x1^2+3x2^2 (convexity example)", lambda x: 5 * x[0] ** 2 + 3 * x[1] ** 2, lambda p: [[10.0, 0.0], [0.0, 6.0]], [1.0, 1.0]),
        ("3x^2+2xy+y^2 (gradient example)", f_grad_ex, lambda p: [[6.0, 2.0], [2.0, 2.0]], [2.0, 3.0]),
        ("x^3-3xy^2 at (1,1)", f_cubic, cubic_h, [1.0, 1.0]),
        ("x^3-3xy^2 at (0,1)", f_cubic, cubic_h, [0.0, 1.0]),
        ("x^3-3xy^2 at (0,0)", f_cubic, cubic_h, [0.0, 0.0]),
    ]
    rows = []
    for name, f, Hf, p in ex:
        Ha = Hf(p)
        Hn = hess_fd(f, p, H)
        err = max(abs(Ha[i][j] - Hn[i][j]) for i in range(2) for j in range(2))
        eigs = eig_sym(Ha)
        cl = classify(eigs)
        lg(f"{name} at {p}: H={fmt(Ha[0])}{fmt(Ha[1])}, FD-vs-analytic max err={err:.2e}, eig={fmt(eigs)}, {cl}")
        if abs(eig_sym(Ha)[0] - eig2_closed(Ha)[0]) > 1e-9:
            lg("  WARNING: Jacobi and closed-form eigenvalues disagree")
        if name.startswith("2x^2+3xy") or name.startswith("x1^2-x2^2") or name.startswith("5x1"):
            for v in ([1.0, 1.0], [1.0, -1.0]):
                lg(f"  Q(x)=x^T H x at x={v} = {quad_form(Ha, v):.6g}")
        rows.append([name, fmt(p), f"{eigs[0]:.6g}", f"{eigs[1]:.6g}", cl])
    write_table(os.path.join(out, "l1_hessian_classification.csv"), ["example", "point", "eig_min", "eig_max", "class"], rows)
    const_ex = ex[:8]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    ts = [2 * math.pi * i / 200 for i in range(201)]
    for ax, (name, f, Hf, p) in zip(axes.flat, const_ex):
        Ha = Hf(p)
        qs = [quad_form(Ha, [math.cos(t), math.sin(t)]) for t in ts]
        ax.plot(ts, qs)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("angle of unit vector")
        ax.set_ylabel("x^T H x")
    save_plot(fig, out, "l1_quadratic_forms_unit_circle.png")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    xg = [-2 + 4 * i / 80 for i in range(81)]
    z = [[f_cubic([xv, yv]) for xv in xg] for yv in xg]
    ax2.contour(xg, xg, z, levels=20)
    ax2.plot([0], [0], "r*", ms=12, label="(0,0): H=0")
    ax2.plot([1], [1], "bo", label="(1,1): H eig +-6 sqrt2")
    ax2.plot([0], [1], "go", label="(0,1): H eig +-6")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8)
    ax2.set_title("f=x^3-3xy^2 (monkey saddle, 3 branches)")
    save_plot(fig2, out, "l1_cubic_saddle_contour.png")


def taylor_sin(x, order):
    s = 0.0
    k = 0
    while 2 * k + 1 <= order:
        p = 2 * k + 1
        fact = 1.0
        for j in range(2, p + 1):
            fact *= j
        s += ((-1) ** k) * x ** p / fact
        k += 1
    return s


def run_taylor(out, lg):
    x = L1["taylor_x"]
    ref = math.sin(x)
    for o in L1["taylor_orders"]:
        t = taylor_sin(x, o)
        lg(f"Taylor sin(x) order {o} at x={x}: {t:.8f}, error={abs(t - ref):.3e} (sin={ref:.8f})")
    xs = [-math.pi + 2 * math.pi * i / 300 for i in range(301)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, [math.sin(t) for t in xs], "k", lw=2, label="sin x")
    for o in L1["taylor_orders"]:
        ax.plot(xs, [taylor_sin(t, o) for t in xs], label=f"order {o}")
    ax.set_ylim(-2, 2)
    ax.legend()
    ax.set_title("Taylor approximation of sin x at 0")
    save_plot(fig, out, "l1_taylor_sin.png")
    p = [1.0, 1.0]
    d = [1.0, 0.0]
    g = g_cubic(p)
    Hp = h_cubic(p)
    fp = f_cubic(p)
    lg(f"f=x^3-3xy^2 at p=(1,1): f={fp:.6g}, grad={fmt(g)}, H={fmt(Hp[0])}{fmt(Hp[1])}, d=(1,0)")
    al = []
    e1s = []
    e2s = []
    for a in L1["taylor_alphas"]:
        exact = f_cubic(add(p, scale(a, d)))
        first = fp + a * dot(g, d)
        second = first + 0.5 * a * a * quad_form(Hp, d)
        e1 = abs(exact - first)
        e2 = abs(exact - second)
        al.append(a)
        e1s.append(e1)
        e2s.append(e2)
        lg(f"  alpha={a}: exact={exact:.8f}, 1st-order err={e1:.3e}, 2nd-order err={e2:.3e}")
    for i in range(1, len(al)):
        lg(f"  slope step {i}: 1st-order log-ratio={math.log(e1s[i] / e1s[i - 1]) / math.log(al[i] / al[i - 1]):.3f} (expect ~2), "
           f"2nd-order log-ratio={math.log(e2s[i] / e2s[i - 1]) / math.log(al[i] / al[i - 1]):.3f} (expect ~3)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.loglog(al, e1s, "o-", label="1st-order error")
    ax2.loglog(al, e2s, "s-", label="2nd-order error")
    ax2.set_xlabel("alpha")
    ax2.set_ylabel("|f(p+alpha d) - approx|")
    ax2.legend()
    ax2.set_title("Taylor error vs step, f=x^3-3xy^2 at (1,1)")
    save_plot(fig2, out, "l1_taylor_error_vs_alpha.png")


def run_optimality(lg):
    A = [[4.0, 1.0], [1.0, 6.0]]
    b = [0.0, 0.0]
    xs = solve(A, b)
    H = [[4.0, 1.0], [1.0, 6.0]]
    f = lambda x: 2 * x[0] ** 2 + 3 * x[1] ** 2 + x[0] * x[1]
    lg(f"FONC: grad f=0 for f=2x^2+3y^2+xy gives x*={fmt(xs)}, f(x*)={f(xs):.6g}, eig(H)={fmt(eig_sym(H))}, {classify(eig_sym(H))}")


def run_constrained(lg):
    f = lambda x: 2 * x[0] ** 2 + 3 * x[1] ** 2 + x[0] * x[1]
    g1 = lambda x: x[0] + x[1] - 1
    g2 = lambda x: -x[0]
    feas = lambda x: g1(x) <= 0 and g2(x) <= 0
    lg(f"constrained: g1(0,0)={g1([0.0, 0.0]):.6g}, g2(0,0)={g2([0.0, 0.0]):.6g}, feasible={feas([0.0, 0.0])}")
    lo, hi, m = L1["cons_lo"], L1["cons_hi"], L1["cons_m"]
    best = None
    for i in range(m):
        for j in range(m):
            x = [lo + (hi - lo) * i / (m - 1), lo + (hi - lo) * j / (m - 1)]
            if feas(x) and (best is None or f(x) < best[0]):
                best = (f(x), x)
    lg(f"constrained grid search over [{lo},{hi}]^2 step {(hi - lo) / (m - 1):.3g}: best feasible f={best[0]:.6g} at {fmt(best[1])}")
    lg(f"constrained: (1,1) feasible? {feas([1.0, 1.0])} (g1=1 > 0)")


def run_knapsack(lg):
    w, v, cap = L1["knap_w"], L1["knap_v"], L1["knap_cap"]
    best = None
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                x = [float(x1), float(x2), float(x3)]
                wt = dot(w, x)
                val = dot(v, x)
                ok = wt <= cap
                lg(f"knapsack x={x} weight={wt:.0f} value={val:.0f} feasible={ok}")
                if ok and (best is None or val > best[0]):
                    best = (val, x, wt)
    lg(f"knapsack optimum (brute force): x={best[1]}, value={best[0]:.0f}, weight={best[2]:.0f}")


def is_local_min(f, x, delta, n):
    fx = f(x)
    for _ in range(n):
        r = delta * random.random() ** 0.5
        th = 2 * math.pi * random.random()
        y = [x[0] + r * math.cos(th), x[1] + r * math.sin(th)]
        if f(y) < fx - 1e-12:
            return False
    return True


def grid_min(f, lo, hi, m):
    best = None
    for i in range(m):
        for j in range(m):
            x = [lo + (hi - lo) * i / (m - 1), lo + (hi - lo) * j / (m - 1)]
            v = f(x)
            if best is None or v < best[0]:
                best = (v, x)
    return best


def grad_desc_fd(f, x0, step, eps, max_it, h):
    x = list(x0)
    k = 0
    while k < max_it:
        g = grad_fd(f, x, h)
        if norm(g) < eps:
            break
        x = sub(x, scale(step, g))
        k += 1
    gn = norm(grad_fd(f, x, h))
    return x, f(x), k, gn, gn < eps


def run_peaks(out, lg):
    lo, hi, m = L1["peaks_lo"], L1["peaks_hi"], L1["peaks_m"]
    gv, gx = grid_min(peaks, lo, hi, m)
    lg(f"peaks grid min over [{lo},{hi}]^2 (step {(hi - lo) / (m - 1):.3g}): f={gv:.6f} at {fmt(gx)}")
    lv, lx = grid_min(lambda x: peaks(x, lit=True), lo, hi, m)
    lg(f"peaks with slide's literal sign -(x1^2-x2^2): grid min f={lv:.6e} at {fmt(lx)} (no finite minimum, unbounded below; see report)")
    minima = []
    for s in L1["peaks_starts"]:
        x, v, k, gn, conv = grad_desc_fd(peaks, s, L1["gd_step"], L1["gd_eps"], L1["gd_max_it"], L1["fd_h"])
        if not conv:
            lg(f"start {fmt(s)}: not converged after {k} iters, |grad|={gn:.2e} (flat region), skipped")
            continue
        if any(norm(sub(x, mm[0])) < 0.05 for mm in minima):
            continue
        minima.append((x, v, k, gn))
    rows = []
    for x, v, k, gn in minima:
        eigs = eig_sym(hess_fd(peaks, x, L1["hess_h"]))
        loc = is_local_min(peaks, x, L1["ball_delta"], L1["ball_n"])
        cl = classify(eigs)
        lg(f"stationary point x={fmt(x)} f={v:.6f} iters={k} |grad|={gn:.2e} eig(H)={fmt(eigs)} {cl}; "
           f"local-min ball test (delta={L1['ball_delta']}, {L1['ball_n']} samples)={loc}")
        rows.append([fmt(x), f"{v:.6f}", k, f"{gn:.2e}", cl, loc])
    write_table(os.path.join(out, "l1_peaks_minimizers.csv"), ["x", "f", "iters", "grad_norm", "hessian_class", "local_min_ball_test"], rows)
    n = L1["peaks_surf_m"]
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    Z = [[peaks([xv, yv]) for xv in xs] for yv in xs]
    X, Y = np.meshgrid(np.array(xs), np.array(xs))
    Zn = np.array(Z)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Zn, cmap="viridis", alpha=0.9)
    for x, v, k, gn in minima:
        ax.scatter([x[0]], [x[1]], [v], color="r", s=40)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f")
    ax.set_title("Peaks function with stationary points (red)")
    save_plot(fig, out, "l1_peaks_surface.png")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    cs = ax2.contour(xs, xs, Z, levels=30)
    for x, v, k, gn in minima:
        ax2.plot(x[0], x[1], "r*", ms=12)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_title("Peaks contour with stationary points")
    save_plot(fig2, out, "l1_peaks_contour.png")


def main(out):
    log_path = os.path.join(out, "log.txt")
    set_seed(cfg.SEED, log_path)

    def lg(msg):
        log(log_path, msg)

    lg(f"seed={cfg.SEED} (cfg.SEED), gradient FD step={L1['fd_h']}, Hessian FD step={L1['hess_h']}")
    run_norms(out, lg)
    run_dot_cosine(lg)
    run_single_var(lg)
    run_gradients(out, lg)
    run_hessians(out, lg)
    run_taylor(out, lg)
    run_optimality(lg)
    run_constrained(lg)
    run_knapsack(lg)
    run_peaks(out, lg)


if __name__ == "__main__":
    main(run_dir())
