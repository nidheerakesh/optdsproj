import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import run_dir
import lecture1_intro
import lecture2_gradient_descent
import lecture3_linear_regression
import lecture4_logistic_adaptive
import lecture5_ewma_adam
import lecture6_subgradient
import lecture7_newton
import lecture8_bfgs
import lecture8a_constrained
import lecture9_kkt

MODULES = [
    lecture1_intro,
    lecture2_gradient_descent,
    lecture3_linear_regression,
    lecture4_logistic_adaptive,
    lecture5_ewma_adam,
    lecture6_subgradient,
    lecture7_newton,
    lecture8_bfgs,
    lecture8a_constrained,
    lecture9_kkt,
]


def main():
    out = run_dir()
    print(f"results folder: {out}")
    for m in MODULES:
        print(f"== {m.__name__}")
        m.main(out)


if __name__ == "__main__":
    main()
