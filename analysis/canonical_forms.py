"""Controllable, observable and Jordan canonical realizations."""
import numpy as np
from scipy.linalg import expm
import scipy.linalg as la


def controllable_canonical(A, B, C):
    Mc = np.hstack([B, A@B])
    if np.linalg.matrix_rank(Mc) < 2:
        raise ValueError("System is not controllable")
    coeff = np.poly(np.linalg.eigvals(A))
    a1, a2 = float(coeff[1]), float(coeff[2])
    # T = [B, AB].  This yields Ac = T^-1 A T and Bc = [1, 0]^T
    # for the companion form [[0, -a2],[1, -a1]].
    T = Mc
    if abs(np.linalg.det(T)) < 1e-10:
        raise ValueError("Controllable transformation is ill-conditioned")
    Ac = np.linalg.solve(T, A@T)
    Bc = np.linalg.solve(T, B)
    Cc = C@T
    residual = np.linalg.norm(A@T-T@Ac)+np.linalg.norm(B-T@Bc)
    return {"T":T, "A":Ac, "B":Bc, "C":Cc, "rank":np.linalg.matrix_rank(Mc),
            "residual":residual, "condition":np.linalg.cond(T)}


def observable_canonical(A, B, C):
    Mo = np.vstack([C, C@A])
    if np.linalg.matrix_rank(Mo) < 2:
        raise ValueError("System is not observable")
    To = Mo.T
    Ao = np.linalg.solve(To, A@To)
    Bo = np.linalg.solve(To, B)
    Co = C@To
    residual = np.linalg.norm(A@To-To@Ao)+np.linalg.norm(B-To@Bo)
    return {"T":To, "A":Ao, "B":Bo, "C":Co,
            "residual":residual, "condition":np.linalg.cond(To)}


def jordan_real(A):
    vals, V = la.eig(A)
    if np.all(np.isreal(vals)):
        T = np.real(V)
        Aj = la.solve(T, A@T)
        return vals, T, Aj
    index = int(np.argmin(vals.imag))
    v = V[:, index]
    T = np.column_stack([np.real(v), np.imag(v)])
    if abs(np.linalg.det(T)) < 1e-12:
        raise ValueError("Real modal transformation is singular")
    Aj = la.solve(T, A@T)
    return vals, T, Aj


def complex_modal(A):
    vals, V = la.eig(A)
    Aj = la.solve(V, A@V)
    residual = np.linalg.norm(A@V-V@Aj)
    return vals, V, Aj, residual
