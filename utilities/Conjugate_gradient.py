import numpy as np

#######################################
# CG_solve copied from
# https://github.com/aravindr93/mjrl/blob/master/mjrl/utils/cg_solve.py
#######################################


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    x = np.zeros_like(b)  # if x_0 is None else x_0
    r = b.copy()  # if x_0 is None else b-f_Ax(x_0)
    p = r.copy()
    rdotr = r.T.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.T.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.T.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x