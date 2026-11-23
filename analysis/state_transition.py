"""State transition matrix and analytical responses."""
import numpy as np
from scipy.linalg import expm


def phi_matrix(A, t):
    return expm(A*t)


def step_response(A, B, x0, u, t):
    """Constant-input analytical response and its zero-input/zero-state parts."""
    zi = np.column_stack([expm(A*ti)@x0 for ti in t])
    zs = np.zeros((2, len(t)))
    for k, ti in enumerate(t):
        if ti == 0:
            continue
        # The augmented exponential is exact for constant input.
        M = np.block([[A, B*np.asarray(u).reshape(-1)[0]],
                      [np.zeros((1,2)), np.zeros((1,1))]])
        augmented = expm(M*ti)@np.r_[np.asarray(x0).reshape(-1), 0.0]
        zs[:,k] = augmented[:2]-expm(A*ti)@np.asarray(x0).reshape(-1)
    return zi, zs, zi+zs
