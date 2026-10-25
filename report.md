# Optimization for Data Science: Implementation Report

**Authors:** [Author 1 Name], [Author 2 Name]
**Course:** Optimization for Data Science
**Date:** [Date]

## 1. Abstract

All ten lecture PDFs in the Opt DS folder (Lectures 1, 2, 3, 4, 5, 6, 7, 8, 8A, 9) were implemented from their definitions in plain Python, with matplotlib used only for figures. One full run (`code/run_all.py`) completed with exit code 0 and wrote 32 figures, 6 tables and a seeded log to `results/20261025_143100/`. The main findings are the convergence behaviour of gradient descent variants, the effect of conditioning and step size, the Adam and RMSProp comparison, BFGS versus gradient descent, and the KKT points of the worked constrained examples. Several worked examples in the slides contain arithmetic or sign errors; the code follows the stated formulas and each discrepancy is listed in Section 5.

## 2. Introduction

The lecture PDFs cover unconstrained optimization (gradient descent and its variants, second-order methods), machine learning losses (linear and logistic regression, lasso, hinge loss), non-smooth optimization (subgradients), and constrained optimization (Lagrange multipliers and KKT conditions). These methods are the basis of training most data science models, so each one is implemented and checked against the slide examples.

## 3. Methods

Algorithms are implemented from their definitions using basic Python lists and loops. Vector and matrix helpers, the linear solver, and the results folder are shared through `code/common.py`. Matplotlib is used only for plotting.

| Lecture | File | Algorithms and examples | Status |
|---|---|---|---|
| 1 | `code/lecture1_intro.py` | Norms, dot products, cosine similarity, gradients (analytic vs finite difference), directional derivative, Hessians and eigenvalues, definiteness of 8 examples, Taylor series of sin, FONC example, knapsack brute force, peaks function minimization | Implemented |
| 2 | `code/lecture2_gradient_descent.py` | Fixed-step GD, exact line search, Armijo backtracking, GD with Armijo, ill-conditioned quadratics, Rosenbrock, momentum, Nesterov | Implemented and checked |
| 3 | `code/lecture3_linear_regression.py` | Linear regression gradient, GD, SGD, mini-batch SGD, worked example, normal equation | Implemented and checked |
| 4 | `code/lecture4_logistic_adaptive.py` | Logistic regression gradient, GD and SGD worked example, Adagrad (worked), RMSProp, Adam | Implemented and checked |
| 5 | `code/lecture5_ewma_adam.py` | EWMA with weight formula, RMSProp, Adam with bias correction, Adam examples (function and linear regression) | Implemented and checked |
| 6 | `code/lecture6_subgradient.py` | Subgradient checks (abs, hinge), subgradient method with best-iterate tracking, lasso, hinge-loss SVM | Implemented and checked |
| 7 | `code/lecture7_newton.py` | Newton method (2D worked examples), damped Newton, 1D Newton for roots and optima, Rosenbrock comparison | Implemented and checked |
| 8 | `code/lecture8_bfgs.py` | Secant pairs, inverse BFGS update, curvature check, four secant-pair decisions, BFGS on Rosenbrock and a quadratic | Implemented and checked |
| 8A | `code/lecture8a_constrained.py` | KKT system with Newton solve, multi-start search, equality-constrained examples, LICQ checks, KKT checker | Implemented |
| 9 | `code/lecture9_kkt.py` | Active-set KKT solver for the three worked examples, second-order sufficiency | Implemented |

### 3.1 Experiment setup

- Random seed: `SEED = 42` in `code/config.py`. Every run logs its seed to `log.txt`.
- Configuration: the Lecture 2 and Lecture 3 parameters are in `code/config.py`. Lectures 1, 8A and 9 keep their constants in a local dictionary at the top of each file (see Section 6).
- Results folder: `results/<timestamp>/`, one folder per run.
- Run everything: `python3 code/run_all.py` from the worktree root.

## 4. Results

Numbers below are copied from the run in `results/20261025_143100/log.txt`.

### 4.1 Lecture 2: gradient descent variants

- Ill-conditioned quadratic `f = 0.5(a x1^2 + x2^2)`, start (4, 3): with fixed step sizes 0.5, 0.18, 0.019 for a = 1, 10, 100, GD converges in 23, 79 and 778 iterations. The step size must shrink as conditioning worsens (`l2_conditioning_convergence.png`).
- `4x1^2 + x2^2 - 2x1x2`: fixed step 0.1 takes 104 iterations, exact line search 19, Armijo backtracking 39 (`l2_line_search_comparison.png`).
- Rosenbrock from (-1.2, 1): fixed step 0.001 and Armijo GD both hit the 10000-iteration cap, with final f = 5.9e-5 and 4.2e-10. The log now reports "hit max_it" for these runs (`l2_rosenbrock_paths.png`, `l2_rosenbrock_convergence.png`).
- Momentum (gamma = 0.9) on the a = 100 quadratic converges in 375 iterations, against 778 for plain GD. Nesterov diverges at iteration 155 with the slide's step size 0.019 and the same gamma. The slide does not give gamma for this example, so this is a parameter-choice result, not a bug (`l2_momentum_comparison.png`).

### 4.2 Lecture 3: linear regression

- Worked example (X = [[1,2],[3,4]], Y = (5,7), alpha = 0.01): GD after 2 iterations gives W = (1.0465, 0.59175, 0.63825) with bias first, and loss 2.319 at iteration 2. The first-iteration gradient is (-2.5, -5.0, -7.5).
- SGD picking row 0 then row 1 gives W = (1.047, 0.591, 0.638) with the bias first.
- Synthetic data (N = 200, d = 3, seed 42): final loss GD 0.005657, SGD 0.005705, mini-batch (16) 0.005685, normal equation 0.005657. GD matches the normal equation (`l3_linreg_loss_comparison.png`, `l3_gd_loss.csv`).

### 4.3 Lecture 4: logistic regression and adaptive steps

- Logistic GD on X = [[1,2],[3,4]], Y = (0,1): loss 1.294969 at iteration 1 and 1.282781 at iteration 2.
- Adagrad on X = [[1,0.2],[3,0.01]], Y = (0,1), alpha = 0.01, eps = 1e-8: after 2 iterations W = (0.48300, 0.48294), b = 0.98296.
- 300-iteration comparison on synthetic sparse data: final loss GD (alpha 0.5) 0.571721, Adagrad 0.570896, RMSProp 0.570901, Adam 0.573895 (`l4_logistic_optimizers.png`).

### 4.4 Lecture 5: EWMA and Adam

- EWMA weights for beta = 0.1 are 0.1, 0.09, 0.081, 0.0729, and they sum to 0.998 over 60 points (`l5_ewma_smoothing.png`).
- Adam on f = x1^2 + 2x2^2 from (1, -1), alpha = 0.01, beta1 = 0.9, beta2 = 0.99: after iteration 1, x = (0.99, -0.99), matching the slide.
- Adam on the linear regression example (X = [[1,2],[2,5]], Y = (3,7), b0 = 1): loss 10.000 at iteration 1 and 9.722 at iteration 2.
- RMSProp, implemented from the slide formula: 2 iterations give f = 2.81326 and 2.68354. Over 300 iterations at alpha = 0.05, RMSProp reaches f = 1.9e-3 and Adam reaches 1.2e-13 (`l5_adam_vs_gd.png`, `l5_rmsprop_vs_adam.png`).

### 4.5 Lecture 6: subgradients

- The subgradient inequality holds for g = 0.5 at x = 0 for f = |x|. For the hinge loss at x = 1, g in [-1, 0] is valid and g = 0.5 is not.
- ReLU subgradients at x = 0 are valid for g in {0, 0.5, 1} and invalid for -0.1 and 1.5.
- Subgradient method on f = |x1| + 2|x2| from (3, -2): best f = 0.036510 after 400 iterations.
- Lasso (synthetic, N = 150, d = 8, seed 42): best loss 0.069 for lambda = 0.01, 0.641 for 0.1, 2.880 for 0.5. Five weights stay below 1e-2 in every case, but the subgradient method does not set them exactly to zero (`l6_lasso_convergence.png`).
- Hinge-loss SVM on 120 synthetic 2D points: objective 0.183, training accuracy 1.000 (`l6_svm_hinge_boundary.png`).

### 4.6 Lecture 7: Newton's method

- Example 2 (f = x^2 + 2xy + y^2, damped with lambda = 1, from (1, 0)): x1 = (0.6, -0.4), f = 0.04, matching the slide.
- Example 1 (f = x^2 + 2xy + 3y^2 - 6x - 12y, from (0, 0)): the code gives x1 = (1.5, 1.5), f = -13.5. The slide gives (2, -1), f = -7, which does not follow from its own function (see Section 5).
- 1D Newton: root of x^2 - 2 from 1 in 4 iterations to 1.41421356; optimum of x^4 - 3x^3 + 2 from 3 in 5 iterations to x = 2.25, f = -6.542969.
- Rosenbrock from (-1.2, 1): Newton with lambda = 0 converges in 6 iterations, damped Newton with lambda = 1 takes 46, and fixed-step GD takes 10000 (`l7_newton_rosenbrock.png`, `l7_newton_1d.png`).

### 4.7 Lecture 8: BFGS

- Worked update H = I, delta = (1, 1), y = (2, 3): rho = 1/5, H' = [[0.92, -0.28], [-0.28, 0.52]], and H'y = delta holds. The search direction is (-1, -1), so x1 = (1, 0), matching the slide.
- Secant pairs: (1,1),(2,1) perform (rho = 1/3); (1,0),(0,2) skip (y.delta = 0); (1,0),(-1,2) skip (y.delta = -1); (1,2),(2,3) perform (rho = 1/8).
- BFGS on Rosenbrock from (-1.2, 1) converges in 36 iterations with one skipped update, final f = 3e-18. GD with Armijo hits the cap (f = 4.2e-10). BFGS on the a = 100 quadratic takes 3 iterations (`l8_bfgs_vs_gd_rosenbrock.png`, `l8_bfgs_quadratic.png`).

### 4.8 Lecture 8A: constrained optimization

- Equality-constrained problems solved by Newton on the KKT system: f = xy on x^2/8 + y^2/2 = 1 gives 4 stationary points with f = +-2; x1 + x2 on the unit circle gives 2 points with f = +-2; two equality constraints give (0, 1) and (1, 0) with lambda = (0, -1), f = 1.
- Multi-start search: 200 of 200 starts converge to the 4 stationary points, with KKT residual below 1e-11.
- Lagrange multipliers for the DIY example at (sqrt 2, 2) and (-sqrt 2, 2) are lambda = (1, -5) in the code. See Section 5 for the slide's different value.
- The inequality KKT check is tested on a problem made for this report, labelled as such in the log, because Lecture 8A has no numeric inequality example.

### 4.9 Lecture 9: KKT conditions

- Example 1: minimize (x-2)^2 + 2(y-1)^2 subject to x + 4y <= 3 and y <= x. The only KKT point is {g1} active at x = (5/3, 1/3), lambda1 = 2/3, lambda2 = 0, f = 1. The second-order check passes (critical cone dimension 1, Z^T H Z = 36 > 0). The problem is convex, so this is the global minimum (`l9_example1_feasible_region.png`, `l9_kkt_cases.csv`).
- Example 2: the only KKT point is x = (3/2, -1/2), lambda = 1, mu = -2, f = 7/2. The critical cone is {0}, so the second-order condition holds trivially.
- Example 3: the only KKT point is x = (0, 4), lambda = 10, mu = -4, f = 13. The unconstrained minimum (2.5, 1.5) violates x <= 0.

### 4.10 Lecture 1: definitions and examples

- Norm and distance examples: L1 distance 7, L2 distance 5, L1 norm 13. Cosine similarities 0.7699, 0.9097, -0.5345, 0 (vectors as set in `code/lecture1_intro.py`).
- Directional derivative 4.4. Knapsack optimum 220 at (0, 1, 1).
- Peaks function: global minimum -6.551133 at (0.2283, -1.6255); a local minimum -3.049849 at (-1.3474, 0.2045) (`l1_peaks_surface.png`, `l1_peaks_minimizers.csv`).
- Taylor series of sin(1): order 1 error 1.6e-1, order 3 error 8.1e-3, order 5 error 2.0e-4 (`l1_taylor_sin.png`).

## 5. Discussion

### 5.1 Slide discrepancies

The code follows the stated formulas. Where the slide's arithmetic or notation is wrong, the code gives the corrected value and the report records the difference.

- Lecture 1: the PSD example f = 6x1^2 has Hessian diag(12, 0), not diag(6, 0). A label "lambda1=6, lambda1=10" should read lambda2 = 10. The NSD example pairs (1,1) with eigenvalue 0, but (1,1) has eigenvalue -4 and (1,-1) has 0. The cubic x^3 - 3xy^2 has d2f/dy2 = -6x, not -6y. The peaks formula uses exp(-(x1^2 - x2^2)) on the slide; with this sign the function is unbounded below, so the code uses +.
- Lecture 3: the GD gradient middle entry is -5.0, not -4.5, which carries into all later GD values. The SGD iteration 1 update is (1.025, 0.525, 0.55) from its own gradient, not (0.9875, 0.375, 0.475), and the iteration 2 values follow from that error.
- Lecture 5: the loss is written as 1/2 ||E||^2 = 20 on the slide, but the stated 1/(2N) formula gives 10. The slide's gradient (-8, -14, -34) omits 1/N. One Y-hat coefficient is 1 where it should be 2, and the second beta is labelled beta1 but is beta2 = 0.99.
- Lecture 7: Example 1's gradient (2x - 4, 6y + 6) and Hessian do not match the stated function. The correct result is (1.5, 1.5), f = -13.5. The slide's (2, -1), f = -7 gives f(2, -1) = 3 from the stated function.
- Lecture 8A: the DIY slide gives lambda = (1, -3) and (-1, -1); the code and a hand check both give lambda = (1, -5).
- Lecture 9: the stationarity lines show 2x - 2 and 4y - 1, which should be 2(x - 2) and 4(y - 1). Example 1's Lagrangian writes lambda2 g1 where it should be lambda2 g2, and Example 2's writes mu g1 where it should be mu g2. Example 2's KKT point shows x = 3/5, but the equations give x = 3/2.

### 5.2 Gaps and limitations

- Lecture 1: Pearson, Jaccard and Mahalanobis similarities are not implemented because the slides give no formulas or data for them.
- Lecture 7: the secant method from the slide's recall section is not implemented.
- Lecture 8: the slide 29 positive and negative curvature illustrations (y = (1,2) and y = (-2,1)) are not implemented as separate cases.
- Lecture 6: the lasso worked example on the slide (lambda = 0.5, alpha = 0.1) is not run as a separate check.
- Lecture 8A: penalty and gradient-projection methods are not in this lecture, so they are not implemented.
- Nesterov at the slide's parameters diverges (Section 4.1). Its gamma is not given on the slide.

## 6. Conclusion

All ten lecture PDFs are implemented in plain Python and run together through `code/run_all.py`. Gradient descent needs a step size matched to the conditioning. Momentum and Nesterov help but depend on their parameters. Adaptive methods (Adagrad, RMSProp, Adam) converge faster than fixed-step GD on the sparse logistic problem. BFGS needs far fewer iterations than GD on Rosenbrock. The KKT examples give the points listed above, once the slide's sign and arithmetic errors are corrected. Several worked examples on the slides contain errors, and these are listed in Section 5.1.

## 7. Reproducibility

- Install: `matplotlib` only; no other libraries are needed for the core algorithms.
- Run everything: `python3 code/run_all.py` from the worktree root. Each run writes to a new `results/<timestamp>/` folder.
- Seed: `SEED = 42` (in `code/config.py`), logged at the top of each `log.txt`.
- Known remaining issues:
  - Lectures 1, 8A and 9 keep some constants in local dictionaries rather than `code/config.py`, and Lecture 9 duplicates helper functions that exist in `common.py`. Both are against prompt.md (rules 1 and 5 for helpers and the one-config rule).
  - Lecture 6 still has an unused helper `sub_abs`.
  - Earlier failed runs left folders in `results/`; they were not deleted.

## References

[Add references in the form: Title, Authors, Year. Do not add references that have not been checked.]
