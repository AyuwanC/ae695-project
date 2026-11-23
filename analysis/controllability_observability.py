"""Controllability, observability and index calculations."""
import numpy as np


def controllability(A, B):
    Mc = np.hstack([B, A@B])
    rank = np.linalg.matrix_rank(Mc)
    singular = np.linalg.svd(Mc, compute_uv=False)
    Wc = __import__("scipy").linalg.solve_continuous_lyapunov(A, -B@B.T)
    # Return negative? solve_continuous_lyapunov gives P for A P + P A.T = -BBT.
    Wc = -Wc
    return Mc, rank, singular, Wc


def observability(A, C):
    Mo = np.vstack([C, C@A])
    rank = np.linalg.matrix_rank(Mo)
    singular = np.linalg.svd(Mo, compute_uv=False)
    Wo = __import__("scipy").linalg.solve_continuous_lyapunov(A.T, -C.T@C)
    Wo = -Wo
    return Mo, rank, singular, Wo
