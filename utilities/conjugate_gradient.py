"""Module containing the conjugate gradient method. This code is based
on https://github.com/aravindr93/mjrl/blob/master/mjrl/utils/cg_solve.py
written by Aravind Rajeswaran on May 6, 2018.

:Date: 2019-03-11
:Version: 1
:Authors:
    - Cedric Derstroff
    - Janosch Moos
    - Kay Hansel
"""

import numpy as np


def conjugate_gradient(ax, b, cg_iter: int = 10, residual_tol: float = 1e-10):
    """ This conjugate gradient method solves the system of linear
    equations Ax=b for the vector x without inverting the matrix A
    directly

    :param ax: The matrix product Ax
    :type ax: array_like

    :param b: The solution of the system of linear equations b
    :type b: array_like

    :param cg_iter: The number of conjugate gradient iterations
    :type cg_iter: int

    :param residual_tol: The termination criterion to exit the loop as
        soon as the residual is sufficiently small
    :type residual_tol: float

    :return: Solution vector x
    :rtype: array_like
    """

    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    r_dot_r = r.T.dot(r)

    for i in range(cg_iter):
        z = ax(p)
        v = r_dot_r / p.T.dot(z)
        x += v * p
        r -= v * z
        new_r_dot_r = r.T.dot(r)
        mu = new_r_dot_r / r_dot_r
        p = r + mu * p

        r_dot_r = new_r_dot_r
        if r_dot_r < residual_tol:
            break
    return x
