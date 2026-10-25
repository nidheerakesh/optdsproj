import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

from common import dot, add, sub, scale, mat_vec, norm, log, set_seed, save_plot, run_dir
import config as cfg
from lecture2_gradient_descent import rosen, rosen_grad, gd_armijo, backtrack, quad


def outer(a, b):
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


def matmul(A, B):
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for t in range(k):
                C[i][j] += A[i][t] * B[t][j]
    return C


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def madd(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A))] for i in range(len(A))]


def mscale(c, A):
    return [[c * A[i][j] for j in range(len(A))] for i in range(len(A))]


def bfgs_inverse_update(H, delta, y):
    sy = dot(y, delta)
    if sy <= 0:
        return H, None, False
    rho = 1.0 / sy
    n = len(delta)
    I = eye(n)
    V = sub_mat(I, mscale(rho, outer(delta, y)))
    Vt = [[V[j][i] for j in range(n)] for i in range(n)]
    term1 = matmul(matmul(V, H), Vt)
    term2 = mscale(rho, outer(delta, delta))
    return madd(term1, term2), rho, True


def sub_mat(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(A))]


def bfgs(f, g, x0, max_it, eps=cfg.EPS):
    x = list(x0)
    H = eye(len(x))
    gr = g(x)
    hist = [f(x)]
    path = [list(x)]
    skipped = 0
    for _ in range(max_it):
        if norm(gr) <= eps:
            break
        p = scale(-1.0, mat_vec(H, gr))
        a = backtrack(f, g, x, p, cfg.L2["alpha_init"], cfg.L2["beta"], cfg.L2["c1"])
        x_new = add(x, scale(a, p))
        g_new = g(x_new)
        delta = sub(x_new, x)
        y = sub(g_new, gr)
        H, _, ok = bfgs_inverse_update(H, delta, y)
        if not ok:
            skipped += 1
        x, gr = x_new, g_new
        hist.append(f(x))
        path.append(list(x))
    return x, path, hist, skipped


def check_worked(lg):
    H0 = eye(2)
    delta = [1.0, 1.0]
    y = [2.0, 3.0]
    H1, rho, ok = bfgs_inverse_update(H0, delta, y)
    lg(f"Worked BFGS: H0=I delta={delta} y={y} y.delta={dot(y, delta)} rho={rho:.6f} updated={ok}")
    lg(f"  H1={[[round(v, 6) for v in r] for r in H1]} expected [[0.92, -0.28], [-0.28, 0.52]]")
    lg(f"  secant check H1*y={mat_vec(H1, y)} delta={delta}")
    x_k = [2.0, 1.0]
    g_k = [2.0, 3.0]
    p = scale(-1.0, mat_vec(H1, g_k))
    x_next = add(x_k, scale(1.0, p))
    lg(f"  p=-H1 g={p} x_next={x_next} expected p=(-1,-1), x_next=(1,0)")


def check_slide37(lg):
    H0 = eye(2)
    delta = [1.0, 2.0]
    y = [2.0, 3.0]
    H1, rho, ok = bfgs_inverse_update(H0, delta, y)
    lg(f"Slide 37 BFGS: H0=I delta={delta} y={y} y.delta={dot(y, delta)} rho={rho:.6f} updated={ok}")
    lg(f"  H1={[[round(v, 6) for v in r] for r in H1]} expected [[0.828125, -0.21875], [-0.21875, 0.8125]]")
    lg(f"  secant check H1*y={mat_vec(H1, y)} delta={delta}")


def check_skip_cases(lg):
    cases = [
        ("(i)", [1.0, 1.0], [2.0, 1.0]),
        ("(ii)", [1.0, 0.0], [0.0, 2.0]),
        ("(iii)", [1.0, 0.0], [-1.0, 2.0]),
        ("final", [1.0, 2.0], [2.0, 3.0]),
    ]
    for name, d, y in cases:
        sy = dot(y, d)
        _, rho, ok = bfgs_inverse_update(eye(2), d, y)
        decision = "perform" if ok else "skip"
        lg(f"Secant pair {name}: delta={d} y={y} y.delta={sy} decision={decision} rho={rho}")


def run_compare(out, lg):
    x0 = cfg.L2["rosen_x0"]
    xb, pb, hb, sk = bfgs(rosen, rosen_grad, x0, 2000)
    lg(f"BFGS on Rosenbrock from {x0}: iters={len(hb)-1} skipped_updates={sk} final_x={xb} f={rosen(xb):.3e}")
    xg, pg, hg = gd_armijo(rosen, rosen_grad, x0, max_it=cfg.MAX_IT)
    lg(f"GD Armijo on Rosenbrock: iters={len(hg)-1} final_f={hg[-1]:.3e}")
    f, gq = quad(100.0, 1.0)
    xq, pq, hq, skq = bfgs(f, gq, cfg.L2["x0"], 500)
    lg(f"BFGS on a=100 quadratic from {cfg.L2['x0']}: iters={len(hq)-1} final_f={hq[-1]:.3e}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(hb)), [max(v, 1e-16) for v in hb], label="BFGS (Rosenbrock)")
    ax.semilogy(range(len(hg)), [max(v, 1e-16) for v in hg], label="GD Armijo (Rosenbrock)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.set_title("BFGS vs GD with Armijo")
    save_plot(fig, out, "l8_bfgs_vs_gd_rosenbrock.png")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(range(len(hq)), [max(v, 1e-300) for v in hq], label="BFGS on a=100 quadratic")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("f(x)")
    ax2.legend()
    save_plot(fig2, out, "l8_bfgs_quadratic.png")


def main(out):
    log_path = os.path.join(out, "log.txt")
    set_seed(cfg.SEED, log_path)

    def lg(msg):
        log(log_path, msg)

    check_worked(lg)
    check_slide37(lg)
    check_skip_cases(lg)
    run_compare(out, lg)


if __name__ == "__main__":
    main(run_dir())
