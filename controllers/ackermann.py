"""Independent Ackermann formula implementation."""
import numpy as np


def ackermann_gain(A, B, desired_poles):
    n = A.shape[0]
    Mc = np.hstack([B, A@B])
    if np.linalg.matrix_rank(Mc) < n:
        raise ValueError("Ackermann formula requires a controllable SISO system")
    phi_A = np.eye(n)
    for pole in desired_poles:
        phi_A = (A-pole*np.eye(n))@phi_A
    selector = np.zeros((1,n)); selector[0,-1] = 1.0
    K = selector @ np.linalg.solve(Mc, phi_A)
    return K
