"""Full-order observers using pole placement, Ackermann and dual LQR."""
import numpy as np
from scipy.signal import place_poles
from scipy.linalg import solve_continuous_are


def observer_pole_gain(A, C, poles):
    result = place_poles(A.T, C.T, poles)
    L = result.gain_matrix.T
    return L, np.linalg.eigvals(A-L@C)


def observer_ackermann(A, C, desired_poles):
    n = A.shape[0]
    Mo = np.vstack([C, C@A])
    if np.linalg.matrix_rank(Mo) < n:
        raise ValueError("Dual Ackermann requires an observable system")
    Mcd = np.hstack([C.T, A.T@C.T])
    phi_A = np.eye(n)
    for pole in desired_poles:
        phi_A = (A.T-pole*np.eye(n))@phi_A
    selector = np.zeros((1,n)); selector[0,-1] = 1.0
    Kdual = selector @ np.linalg.solve(Mcd, phi_A)
    L = Kdual.T
    return L, np.linalg.eigvals(A-L@C)


def observer_lqr(A, C, Q=None, R=None):
    Q = np.diag([10.0, 100.0]) if Q is None else np.asarray(Q)
    R = np.array([[1.0]]) if R is None else np.asarray(R)
    P = solve_continuous_are(A.T, C.T, Q, R)
    L = np.linalg.solve(R, C@P).T
    residual = A@P+P@A.T-P@C.T@np.linalg.solve(R, C)@P+Q
    return L, np.linalg.eigvals(A-L@C), residual
