"""LQR as Lyapunov/Riccati state-feedback controller."""
import numpy as np
from scipy.linalg import solve_continuous_are


def lqr_gain(A, B, Q=None, R=None):
    Q = np.diag([10.0, 100.0]) if Q is None else np.asarray(Q)
    # R=100 limits yaw-rate feedback and avoids saturation chattering in the
    # nonlinear observer-feedback closed loop. R remains configurable.
    R = np.array([[100.0]]) if R is None else np.asarray(R)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T@P)
    residual = A.T@P+P@A-P@B@np.linalg.solve(R, B.T)@P+Q
    Acl = A-B@K
    lyap_residual = Acl.T@P+P@Acl+(Q+K.T@R@K)
    return K, P, residual, lyap_residual


def reference_precompensator(A, B, C, K, measured_yaw_only=False):
    """Calculate Nbar for r_ref using dc_gain * Nbar = 1."""
    Acl = A-B@K
    Ck = np.array([[0.0, 1.0]]) if measured_yaw_only else C
    # Nbar = -1 / [C (A-BK)^(-1) B], evaluated with linear solves.
    response = np.linalg.solve(Acl, B)
    dc = Ck@response
    gain = -float(dc[0,0])
    if abs(gain) < 1e-12:
        raise ValueError("Reference precompensator is singular")
    return float(1.0/gain)
