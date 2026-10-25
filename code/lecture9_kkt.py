import csv
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-9


def f1(x):
    return (x[0] - 2) ** 2 + 2 * (x[1] - 1) ** 2


def f2(x):
    return x[0] ** 2 + x[1] ** 2 + 1


def f3(x):
    return (x[0] - 3) ** 2 + (x[1] - 2) ** 2


PROBLEMS = [
    {
        "name": "Example 1",
        "f": f1,
        "n": 2,
        "Q": [[2.0, 0.0], [0.0, 4.0]],
        "c": [-4.0, -4.0],
        "ineq": [([1.0, 4.0], -3.0), ([-1.0, 1.0], 0.0)],
        "ineq_names": ["g1", "g2"],
        "eq": [],
        "eq_names": [],
    },
    {
        "name": "Example 2",
        "f": f2,
        "n": 2,
        "Q": [[2.0, 0.0], [0.0, 2.0]],
        "c": [0.0, 0.0],
        "ineq": [([-1.0, -1.0], 1.0)],
        "ineq_names": ["g1"],
        "eq": [([1.0, -1.0], -2.0)],
        "eq_names": ["g2"],
    },
    {
        "name": "Example 3",
        "f": f3,
        "n": 2,
        "Q": [[2.0, 0.0], [0.0, 2.0]],
        "c": [-6.0, -4.0],
        "ineq": [([1.0, 0.0], 0.0)],
        "ineq_names": ["g"],
        "eq": [([1.0, 1.0], -4.0)],
        "eq_names": ["h"],
    },
]


def run_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ROOT, "results", ts)
    os.makedirs(path, exist_ok=True)
    return path


def log(log_path, msg):
    with open(log_path, "a") as fh:
        fh.write(msg + "\n")
    print(msg)


def save_plot(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def matvec(M, v):
    return [dot(row, v) for row in M]


def aff(con, x):
    return dot(con[0], x) + con[1]


def solve_lin(A, b):
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(M[r][c]) > abs(M[p][c]):
                p = r
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def det(A):
    n = len(A)
    M = [list(r) for r in A]
    d = 1.0
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(M[r][c]) > abs(M[p][c]):
                p = r
        if abs(M[p][c]) < 1e-14:
            return 0.0
        if p != c:
            M[c], M[p] = M[p], M[c]
            d = -d
        d *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for j in range(c, n):
                M[r][j] -= f * M[c][j]
    return d


def nullspace(R, n):
    M = [list(r) for r in R]
    piv = []
    row = 0
    for c in range(n):
        p = None
        best = 1e-12
        for r in range(row, len(M)):
            if abs(M[r][c]) > best:
                best = abs(M[r][c])
                p = r
        if p is None:
            continue
        M[row], M[p] = M[p], M[row]
        pv = M[row][c]
        M[row] = [v / pv for v in M[row]]
        for r in range(len(M)):
            if r != row and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [M[r][j] - f * M[row][j] for j in range(n)]
        piv.append(c)
        row += 1
        if row == len(M):
            break
    basis = []
    for fc in range(n):
        if fc in piv:
            continue
        v = [0.0] * n
        v[fc] = 1.0
        for k, pc in enumerate(piv):
            v[pc] = -M[k][fc]
        basis.append(v)
    return basis


def leading_minors_pd(M):
    k = len(M)
    minors = [det([r[:p] for r in M[:p]]) for p in range(1, k + 1)]
    return minors, all(v > TOL for v in minors)


def build_system(prob, S):
    n, Q, c = prob["n"], prob["Q"], prob["c"]
    ine, eqs = prob["ineq"], prob["eq"]
    m = n + len(S) + len(eqs)
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for k in range(n):
        for j in range(n):
            A[k][j] = Q[k][j]
        for s, i in enumerate(S):
            A[k][n + s] = ine[i][0][k]
        for t, e in enumerate(eqs):
            A[k][n + len(S) + t] = e[0][k]
        b[k] = -c[k]
    row = n
    for con in [ine[i] for i in S] + eqs:
        for j in range(n):
            A[row][j] = con[0][j]
        b[row] = -con[1]
        row += 1
    return A, b


def solve_case(prob, S):
    n, ine, eqs = prob["n"], prob["ineq"], prob["eq"]
    label = "{" + ", ".join(prob["ineq_names"][i] for i in S) + "}"
    cs = {"label": label, "S": S, "kkt": False}
    A, b = build_system(prob, S)
    sol = solve_lin(A, b)
    if sol is None:
        cs["status"] = "singular KKT system"
        return cs
    x = sol[:n]
    lam = {i: sol[n + s] for s, i in enumerate(S)}
    mu = sol[n + len(S):]
    gv = [aff(g, x) for g in ine]
    hv = [aff(h, x) for h in eqs]
    fails = []
    for i, v in enumerate(gv):
        if v > TOL:
            fails.append(f"primal {prob['ineq_names'][i]}={v:.4g} > 0")
    for j, v in enumerate(hv):
        if abs(v) > TOL:
            fails.append(f"primal {prob['eq_names'][j]}={v:.4g} != 0")
    for i in S:
        if lam[i] < -TOL:
            fails.append(f"dual lambda_{prob['ineq_names'][i]}={lam[i]:.4g} < 0")
    comp = max([abs(lam.get(i, 0.0) * gv[i]) for i in range(len(ine))])
    if comp > TOL:
        fails.append(f"complementary slackness {comp:.4g} != 0")
    mult = [(prob["ineq_names"][i], lam[i]) for i in S]
    mult += [(prob["eq_names"][j], mu[j]) for j in range(len(eqs))]
    cs.update(
        x=x, lam=lam, mult=mult, gv=gv, hv=hv, f=prob["f"](x), comp=comp,
        kkt=not fails, status="KKT point" if not fails else "; ".join(fails),
    )
    return cs


def enumerate_cases(prob):
    m_in = len(prob["ineq"])
    return [
        solve_case(prob, [i for i in range(m_in) if (mask >> i) & 1])
        for mask in range(1 << m_in)
    ]


def second_order(prob, cs):
    n, Q = prob["n"], prob["Q"]
    rows = [e[0] for e in prob["eq"]]
    rows += [prob["ineq"][i][0] for i in cs["S"] if cs["lam"][i] > TOL]
    degenerate = [prob["ineq_names"][i] for i in cs["S"] if abs(cs["lam"][i]) <= TOL]
    Z = nullspace(rows, n)
    if not Z:
        return {"dim": 0, "minors": [], "ok": True, "degenerate": degenerate}
    M = [[dot(Z[i], matvec(Q, Z[j])) for j in range(len(Z))] for i in range(len(Z))]
    minors, ok = leading_minors_pd(M)
    return {"dim": len(Z), "minors": minors, "ok": ok, "degenerate": degenerate}


def fmt_vec(v):
    return "(" + ", ".join(f"{t:.6g}" for t in v) + ")"


def describe(cs):
    mult = ", ".join(f"{nm}={val:.6g}" for nm, val in cs["mult"]) or "none"
    if "x" not in cs:
        return f"active {cs['label']}: {cs['status']}"
    g = ", ".join(f"{t:.4g}" for t in cs["gv"])
    return (
        f"active {cs['label']}: x={fmt_vec(cs['x'])} f={cs['f']:.6g} "
        f"mult[{mult}] g=[{g}] -> {cs['status']}"
    )


def plot_ex1(out, kx, lg_path):
    xs = [-1.0 + 0.05 * i for i in range(91)]
    ys = [-1.0 + 0.05 * j for j in range(71)]
    feas = [[1.0 if (x + 4 * y - 3 <= 0 and y - x <= 0) else 0.0 for x in xs] for y in ys]
    fv = [[f1([x, y]) for x in xs] for y in ys]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xs, ys, feas, levels=[0.5, 1.5], colors=["#cfe8cf"])
    ax.contour(xs, ys, fv, levels=[0.5, 1, 2, 4, 6, 9], cmap="viridis")
    ax.plot(xs, [(3 - x) / 4 for x in xs], "g-", lw=1, label="g1 = 0")
    ax.plot(xs, xs, "m-", lw=1, label="g2 = 0")
    ax.plot(2, 1, "bs", label="unconstrained min (2, 1)")
    ax.plot(kx[0], kx[1], "r*", ms=14, label=f"KKT point ({kx[0]:.3f}, {kx[1]:.3f})")
    ax.set_xlim(-1.0, 3.5)
    ax.set_ylim(-1.0, 2.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Example 1: feasible region (green), f contours, KKT point")
    ax.legend(loc="upper right", fontsize=8)
    path = save_plot(fig, out, "l9_example1_feasible_region.png")
    log(lg_path, f"plot saved: {path}")


def main(out):
    log_path = os.path.join(out, "log.txt")
    random.seed(cfg.SEED)
    log(log_path, f"seed={cfg.SEED}")
    rows = []
    ex1_kkt = None
    for prob in PROBLEMS:
        Qm, pd = leading_minors_pd(prob["Q"])
        log(log_path, f"== {prob['name']}: Hessian of f PD={pd}, affine constraints, "
                      f"convex problem={pd}")
        cases = enumerate_cases(prob)
        for cs in cases:
            log(log_path, f"  case {describe(cs)}")
            rows.append([prob["name"], cs["label"],
                         ";".join(f"{v:.6g}" for v in cs["x"]) if "x" in cs else "",
                         ";".join(f"{nm}={val:.6g}" for nm, val in cs.get("mult", [])),
                         cs["status"]])
        kkts = [cs for cs in cases if cs["kkt"]]
        for cs in kkts:
            so = second_order(prob, cs)
            log(log_path, f"  KKT {cs['label']} second order: dim(critical cone)={so['dim']} "
                          f"leading minors of Z^T H Z={[round(v, 6) for v in so['minors']]} "
                          f"sufficient={so['ok']} degenerate_ineq={so['degenerate']}")
        if prob["name"] == "Example 1" and kkts:
            ex1_kkt = kkts[0]["x"]
    if ex1_kkt is not None:
        plot_ex1(out, ex1_kkt, log_path)
    with open(os.path.join(out, "l9_kkt_cases.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["example", "active_set", "x", "multipliers", "status"])
        w.writerows(rows)


if __name__ == "__main__":
    main(run_dir())
