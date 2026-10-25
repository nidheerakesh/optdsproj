import os
import sys
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

from common import dot, add, scale, norm, solve, log, set_seed, save_plot, write_table
import config as cfg

L8A = {
    "starts": 200,
    "box": 3.0,
    "tol": 1e-10,
    "max_it": 50,
    "close": 1e-6,
    "match": 1e-5,
    "newton_start": [1.0, 0.5, 1.0],
}

S2 = math.sqrt(2.0)


def zero2(x):
    return [[0.0, 0.0], [0.0, 0.0]]


def vs(v, sep=", "):
    return sep.join(f"{a:.6f}" for a in v)


def close_to(a, b, tol):
    return len(a) == len(b) and all(abs(a[i] - b[i]) < tol for i in range(len(a)))


def mat_t_vec(M, v):
    n = len(M[0]) if M else 0
    out = [0.0] * n
    for j in range(len(M)):
        for i in range(n):
            out[i] += M[j][i] * v[j]
    return out


def rank(M, tol=1e-9):
    if not M:
        return 0
    A = [list(row) for row in M]
    m, k = len(A), len(A[0])
    r = 0
    for c in range(k):
        p, best = None, tol
        for i in range(r, m):
            if abs(A[i][c]) > best:
                best, p = abs(A[i][c]), i
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        for i in range(m):
            if i != r:
                t = A[i][c] / A[r][c]
                A[i] = [A[i][j] - t * A[r][j] for j in range(k)]
        r += 1
    return r


# ---------- problems: f, grad f, Hess f, and equality constraints (h, grad h, Hess h) ----------

def prob_xy():
    def h(x):
        return x[0] ** 2 / 8 + x[1] ** 2 / 2 - 1

    def gh(x):
        return [x[0] / 4, x[1]]

    def Hh(x):
        return [[0.25, 0.0], [0.0, 1.0]]

    return {
        "n": 2,
        "f": lambda x: x[0] * x[1],
        "gf": lambda x: [x[1], x[0]],
        "Hf": lambda x: [[0.0, 1.0], [1.0, 0.0]],
        "eq": [(h, gh, Hh)],
    }


def prob_sum_circle():
    def h(x):
        return x[0] ** 2 + x[1] ** 2 - 2

    def gh(x):
        return [2 * x[0], 2 * x[1]]

    def Hh(x):
        return [[2.0, 0.0], [0.0, 2.0]]

    return {
        "n": 2,
        "f": lambda x: x[0] + x[1],
        "gf": lambda x: [1.0, 1.0],
        "Hf": zero2,
        "eq": [(h, gh, Hh)],
    }


def prob_two_eq():
    def h1(x):
        return x[0] + x[1] - 1

    def g1(x):
        return [1.0, 1.0]

    def h2(x):
        return x[0] ** 2 + x[1] ** 2 - 1

    def g2(x):
        return [2 * x[0], 2 * x[1]]

    def H2(x):
        return [[2.0, 0.0], [0.0, 2.0]]

    return {
        "n": 2,
        "f": lambda x: x[0] ** 2 + x[1] ** 2,
        "gf": lambda x: [2 * x[0], 2 * x[1]],
        "Hf": H2,
        "eq": [(h1, g1, zero2), (h2, g2, H2)],
    }


def prob_diy():
    def h1(x):
        return x[1] - x[0] ** 2

    def g1(x):
        return [-2 * x[0], 1.0]

    def H1(x):
        return [[-2.0, 0.0], [0.0, 0.0]]

    def h2(x):
        return x[1] - 2

    def g2(x):
        return [0.0, 1.0]

    return {
        "n": 2,
        "f": lambda x: x[0] ** 2 + x[1] ** 2,
        "gf": lambda x: [2 * x[0], 2 * x[1]],
        "Hf": lambda x: [[2.0, 0.0], [0.0, 2.0]],
        "eq": [(h1, g1, H1), (h2, g2, zero2)],
    }


def eq_pairs(P):
    return [(h, gh) for h, gh, _ in P["eq"]]


# ---------- KKT system for equality constraints: F(x, lam) = 0 and its Jacobian ----------

def kkt_F(P, x, lam):
    r = list(P["gf"](x))
    hs = []
    for j, (h, gh, _) in enumerate(P["eq"]):
        r = add(r, scale(lam[j], gh(x)))
        hs.append(h(x))
    return r + hs


def kkt_J(P, x, lam):
    n, l = P["n"], len(P["eq"])
    H = [list(row) for row in P["Hf"](x)]
    for j, (_, _, Hh) in enumerate(P["eq"]):
        Hj = Hh(x)
        for a in range(n):
            for b in range(n):
                H[a][b] += lam[j] * Hj[a][b]
    M = [[0.0] * (n + l) for _ in range(n + l)]
    for a in range(n):
        for b in range(n):
            M[a][b] = H[a][b]
    for j, (_, gh, _) in enumerate(P["eq"]):
        g = gh(x)
        for a in range(n):
            M[a][n + j] = g[a]
            M[n + j][a] = g[a]
    return M


def newton_kkt(P, z0, tol, max_it):
    n = P["n"]
    z = list(z0)
    hist = []
    for k in range(max_it + 1):
        x, lam = z[:n], z[n:]
        r = kkt_F(P, x, lam)
        rn = norm(r)
        hist.append(rn)
        if rn < tol:
            return z, hist, True
        if k == max_it or not math.isfinite(rn) or rn > 1e8:
            break
        try:
            d = solve(kkt_J(P, x, lam), scale(-1.0, r))
        except ZeroDivisionError:
            break
        if not all(math.isfinite(v) for v in d):
            break
        z = add(z, d)
    return z, hist, False


def find_stationary(P, starts, box, tol, max_it):
    m = P["n"] + len(P["eq"])
    found, conv = [], 0
    for _ in range(starts):
        z0 = [random.uniform(-box, box) for _ in range(m)]
        z, _, ok = newton_kkt(P, z0, tol, max_it)
        if not ok:
            continue
        conv += 1
        if not any(close_to(z, w, L8A["close"]) for w in found):
            found.append(z)
    return found, conv


def same_set(found, n, expected, tol):
    if len(found) != len(expected):
        return False
    for x_ex, lam_ex in expected:
        hit = False
        for z in found:
            if close_to(z[:n], x_ex, tol) and (lam_ex is None or close_to(z[n:], lam_ex, tol)):
                hit = True
        if not hit:
            return False
    return True


def run_newton_case(lg, rows, name, P, expected):
    n = P["n"]
    found, conv = find_stationary(P, L8A["starts"], L8A["box"], L8A["tol"], L8A["max_it"])
    lg(f"{name}: converged starts={conv}/{L8A['starts']}, distinct Lagrange points={len(found)}")
    for z in found:
        x, lam = z[:n], z[n:]
        res = norm(kkt_F(P, x, lam))
        lg(f"  x=({vs(x)}) lam=({vs(lam)}) f={P['f'](x):.6f} kkt_residual={res:.1e}")
        rows.append([name, vs(x, ";"), vs(lam, ";"), f"{P['f'](x):.6f}", f"{res:.1e}"])
    lg(f"  slide values (x, lam): {expected}")
    lg(f"  matches slide: {same_set(found, n, expected, L8A['match'])}")
    return found


# ---------- analytic cases (degenerate or constraint-qualification examples) ----------

def mult_at_point(gf, J, tol):
    l = len(J)
    rk = rank(J)
    if rk == l:
        A = [[dot(J[i], J[j]) for j in range(l)] for i in range(l)]
        b = [-dot(J[i], gf) for i in range(l)]
        lam = solve(A, b)
        return lam, norm(add(gf, mat_t_vec(J, lam)))
    if all(abs(v) < tol for row in J for v in row):
        return None, norm(gf)
    return None, None


def run_sphere_case(lg):
    lg("Example 1 f=x^2+y^2, g=x^2+y^2-1 (lam = -1 case, analytic)")
    lam = -1.0
    for t in [0.0, 0.7, math.pi / 2, math.pi]:
        x = [math.cos(t), math.sin(t)]
        res = norm(add([2 * x[0], 2 * x[1]], scale(lam, [2 * x[0], 2 * x[1]])))
        lg(f"  point=({vs(x)}) lam={lam} stationarity_residual={res:.1e} f={x[0]**2 + x[1]**2:.6f}")
    lg("  for lam != -1, stationarity forces x=y=0, which violates g=0, so no other candidates")


def run_cq_cases(lg):
    tol = L8A["tol"]
    pt = [0.0, 0.0]
    cases = [
        ("Example 2 f=x+y, h=x^2+y^2", [1.0, 1.0], lambda x: [2 * x[0], 2 * x[1]]),
        ("Without CQ case 1 f=2x+3y, h=x^2+y^2", [2.0, 3.0], lambda x: [2 * x[0], 2 * x[1]]),
        ("Without CQ case 2 f=x^2, h=y^2", [0.0, 0.0], lambda x: [0.0, 2 * x[1]]),
    ]
    for name, gf, gh in cases:
        J = [gh(pt)]
        lam, res = mult_at_point(gf, J, tol)
        if lam is not None:
            status = f"lam=({vs(lam)})"
        elif res is None:
            status = "J rank-deficient, multiplier not computed"
        elif res < tol:
            status = "stationarity holds for ANY lam (multiplier not unique)"
        else:
            status = f"NO lam satisfies stationarity, min residual={res:.4f}"
        lg(f"{name} at (0,0): grad f=({vs(gf)}) rank(J)={rank(J)} -> {status}")


def run_licq(lg):
    for name, J in [("h1=x1-1, h2=x2-1", [[1.0, 0.0], [0.0, 1.0]]),
                    ("h1=x1+x2-1, h2=2x1+2x2-2", [[1.0, 1.0], [2.0, 2.0]])]:
        r = rank(J)
        lg(f"LICQ {name}: rank={r}, active constraints={len(J)}, LICQ {'holds' if r == len(J) else 'fails'}")


def kkt_check(f_gf, ineq, eq, x, lam, mu, tol):
    st = list(f_gf(x))
    for i, (_, gg) in enumerate(ineq):
        st = add(st, scale(lam[i], gg(x)))
    for j, (_, gh) in enumerate(eq):
        st = add(st, scale(mu[j], gh(x)))
    stat_ok = norm(st) < tol
    prim_ok = all(g(x) <= tol for g, _ in ineq) and all(abs(h(x)) <= tol for h, _ in eq)
    dual_ok = all(v >= -tol for v in lam)
    comp_ok = all(abs(lam[i] * ineq[i][0](x)) <= tol for i in range(len(ineq)))
    return stat_ok, prim_ok, dual_ok, comp_ok


def run_kkt_checks(lg, P):
    tol = 1e-9
    for pt, mu in [([1.0, 0.0], [0.0, -1.0]), ([0.0, 1.0], [0.0, -1.0])]:
        r = kkt_check(P["gf"], [], eq_pairs(P), pt, [], mu, tol)
        lg(f"KKT equality check Example C at ({vs(pt)}) mu=({vs(mu)}): stationarity, primal, dual, comp = {r}")
    f_gf = lambda x: [2 * x[0], 2 * x[1]]
    ineq = [(lambda x: 1 - x[0] - x[1], lambda x: [-1.0, -1.0])]
    r1 = kkt_check(f_gf, ineq, [], [0.5, 0.5], [1.0], [], tol)
    lg(f"KKT inequality sanity (not a slide example) min x^2+y^2 s.t. 1-x-y<=0 at (0.5,0.5) lam=1: {r1}")
    r2 = kkt_check(f_gf, ineq, [], [1.0, 1.0], [0.0], [], tol)
    lg(f"KKT inequality sanity at (1,1) lam=0 (should fail stationarity): {r2}")


def run_newton_convergence(out, lg):
    P = prob_xy()
    z, hist, ok = newton_kkt(P, L8A["newton_start"], L8A["tol"], L8A["max_it"])
    lg(f"Newton KKT from start {L8A['newton_start']}: ok={ok} iters={len(hist)-1} final=({vs(z)})")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(hist)), hist, "o-")
    ax.set_xlabel("iteration")
    ax.set_ylabel("||F(x, lam)||")
    ax.set_title("Newton on KKT system, f=xy on ellipse")
    save_plot(fig, out, "l8a_kkt_newton_convergence.png")


# ---------- plots ----------

def grid(fn, xr, yr, m=121):
    xs = [xr[0] + (xr[1] - xr[0]) * i / (m - 1) for i in range(m)]
    ys = [yr[0] + (yr[1] - yr[0]) * j / (m - 1) for j in range(m)]
    zs = [[fn(xv, yv) for xv in xs] for yv in ys]
    return xs, ys, zs


def mark(ax, found, n):
    for z in found:
        ax.plot(z[0], z[1], "r*", ms=12)
        ax.annotate(f"({z[0]:.2f}, {z[1]:.2f})", (z[0], z[1]), textcoords="offset points", xytext=(6, 6))


def plot_xy(out, found):
    fig, ax = plt.subplots(figsize=(6, 5))
    xs, ys, zs = grid(lambda a, b: a * b, (-3, 3), (-3, 3))
    ax.contour(xs, ys, zs, levels=[-4, -2, -1, 0, 1, 2, 4])
    ts = [2 * math.pi * k / 400 for k in range(401)]
    ax.plot([math.sqrt(8) * math.cos(t) for t in ts], [math.sqrt(2) * math.sin(t) for t in ts], "k-", label="h=0")
    mark(ax, found, 2)
    ax.set_title("f=xy on x^2/8+y^2/2=1 (stars: Lagrange points)")
    ax.legend()
    save_plot(fig, out, "l8a_xy_ellipse.png")


def plot_sum_circle(out, found):
    fig, ax = plt.subplots(figsize=(6, 5))
    xs, ys, zs = grid(lambda a, b: a + b, (-2, 2), (-2, 2))
    ax.contour(xs, ys, zs, levels=[-2, -1, 0, 1, 2])
    ts = [2 * math.pi * k / 400 for k in range(401)]
    ax.plot([S2 * math.cos(t) for t in ts], [S2 * math.sin(t) for t in ts], "k-", label="h=0")
    mark(ax, found, 2)
    ax.set_title("f=x1+x2 on x1^2+x2^2=2")
    ax.legend()
    save_plot(fig, out, "l8a_sum_on_circle.png")


def plot_two_eq(out, found):
    fig, ax = plt.subplots(figsize=(6, 5))
    xs, ys, zs = grid(lambda a, b: a * a + b * b, (-2, 2), (-2, 2))
    ax.contour(xs, ys, zs, levels=[0.5, 1, 2, 4, 8])
    ax.plot([-2, 2], [3, -1], "g-", label="h: x+y-1=0")
    ts = [2 * math.pi * k / 400 for k in range(401)]
    ax.plot([math.cos(t) for t in ts], [math.sin(t) for t in ts], "k-", label="g: x^2+y^2-1=0")
    mark(ax, found, 2)
    ax.set_title("f=x^2+y^2 with two equality constraints")
    ax.legend()
    save_plot(fig, out, "l8a_two_equality.png")


def plot_diy(out, found):
    fig, ax = plt.subplots(figsize=(6, 5))
    xs, ys, zs = grid(lambda a, b: a * a + b * b, (-2.5, 2.5), (-1, 3))
    ax.contour(xs, ys, zs, levels=[1, 2, 4, 6, 8, 10])
    ax.plot(xs, [xv * xv for xv in xs], "r-", label="h: y-x^2=0")
    ax.plot(xs, [2.0 for _ in xs], "g-", label="g: y-2=0")
    mark(ax, found, 2)
    ax.set_title("DIY: f=x^2+y^2 on parabola and line")
    ax.legend()
    save_plot(fig, out, "l8a_diy_parabola_line.png")


# ---------- main ----------

def main(out):
    log_path = os.path.join(out, "log.txt")
    set_seed(cfg.SEED, log_path)

    def lg(msg):
        log(log_path, msg)

    lg(f"L8A settings: {L8A}")
    rows = []
    expect = {
        "A f=xy h=x^2/8+y^2/2-1": [([2, 1], None), ([-2, -1], None), ([2, -1], None), ([-2, 1], None)],
        "B f=x1+x2 h=x1^2+x2^2-2": [([1, 1], None), ([-1, -1], None)],
        "C two equality h=x+y-1 g=x^2+y^2-1": [([0, 1], [0, -1]), ([1, 0], [0, -1])],
        "D DIY f=x^2+y^2 h=y-x^2 g=y-2": [([S2, 2], [1, -3]), ([-S2, 2], [-1, -1])],
    }
    pa, pb, pc, pd = prob_xy(), prob_sum_circle(), prob_two_eq(), prob_diy()
    names = list(expect.keys())
    fa = run_newton_case(lg, rows, names[0], pa, expect[names[0]])
    fb = run_newton_case(lg, rows, names[1], pb, expect[names[1]])
    fc = run_newton_case(lg, rows, names[2], pc, expect[names[2]])
    fd = run_newton_case(lg, rows, names[3], pd, expect[names[3]])
    run_sphere_case(lg)
    run_cq_cases(lg)
    run_licq(lg)
    run_kkt_checks(lg, pc)
    run_newton_convergence(out, lg)
    plot_xy(out, fa)
    plot_sum_circle(out, fb)
    plot_two_eq(out, fc)
    plot_diy(out, fd)
    write_table(os.path.join(out, "l8a_candidates.csv"), ["case", "x", "lam", "f", "kkt_residual"], rows)


if __name__ == "__main__":
    from common import run_dir
    main(run_dir())
