"""Genuine one-state reduced-order observer."""
import numpy as np


def reduced_order_gain(A, B, desired_pole=-12.0):
    A11, A12 = A[0,0], A[0,1]
    A21, A22 = A[1,0], A[1,1]
    if abs(A21) < 1e-12:
        raise ValueError("Reduced-order gain is undefined: A21 is zero")
    L = (A11-desired_pole)/A21
    return float(L), A11-L*A21


def reduced_order_rhs(A, B, L, z, r, u):
    """Compute dz/dt for z = vy_hat-L*r."""
    z=float(np.asarray(z).reshape(-1)[0])
    r=float(np.asarray(r).reshape(-1)[0])
    u=float(np.asarray(u).reshape(-1)[0])
    A11, A12 = A[0,0], A[0,1]
    A21, A22 = A[1,0], A[1,1]
    B1, B2 = B[0,0], B[1,0]
    zdot = (A11-L*A21)*z + ((A11-L*A21)*L+A12-L*A22)*r + (B1-L*B2)*u
    return float(zdot)
