"""Eigenvalue, eigenvector and stability analysis."""
import numpy as np


def eigen_analysis(A):
    vals, vecs = np.linalg.eig(A)
    rows = []
    for lam, vec in zip(vals, vecs.T):
        if abs(lam.imag) > 1e-10:
            wn = abs(lam)
            zeta = -lam.real/wn
            wd = abs(lam.imag)
            classification = "stable oscillatory"
        else:
            wn = abs(lam.real); zeta = 1.0 if lam.real < 0 else 0.0
            wd = 0.0
            classification = "stable" if lam.real < 0 else ("unstable" if lam.real > 0 else "marginal")
        residual = np.linalg.norm(A@vec-lam*vec)
        rows.append({"real": lam.real, "imag": lam.imag, "magnitude": abs(lam),
                     "natural_frequency": wn, "damped_frequency": wd,
                     "damping_ratio": zeta, "classification": classification,
                     "eigenvector_residual": residual})
    char_coeff = np.poly(vals)
    return vals, vecs, rows, char_coeff
