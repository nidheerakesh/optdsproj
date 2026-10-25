SEED = 42

EPS = 1e-6
MAX_IT = 10000

L2 = {
    "x0": [4.0, 3.0],
    "alphas": {1: 0.5, 10: 0.18, 100: 0.019},
    "conds": [1.0, 10.0, 100.0],
    "fixed_alpha": 0.1,
    "beta": 0.4,
    "c1": 1e-4,
    "alpha_init": 1.0,
    "gamma": 0.9,
    "rosen_x0": [-1.2, 1.0],
    "rosen_alpha": 0.001,
}

L3 = {
    "X": [[1.0, 2.0], [3.0, 4.0]],
    "W": [0.5, 0.5],
    "b": 1.0,
    "Y": [5.0, 7.0],
    "alpha": 0.01,
    "gd_iters": 2,
    "sgd_pick": [0, 1],
    "batch_size": 1,
}
