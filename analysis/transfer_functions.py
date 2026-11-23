"""Numerical transfer functions and symbolic derivation."""
import numpy as np
import sympy as sp
from scipy.signal import tf2ss
from scipy import signal


def transfer_functions(A, B, C, D):
    numvy = []; numr = []
    for row, C_row in enumerate(C):
        num, den = signal.ss2tf(A, B, C_row.reshape(1, -1), D[row:row+1])[0:2]
        (numvy if row == 0 else numr).append(np.asarray(num).real)
    numvy = np.asarray(numvy)[0]
    numr = np.asarray(numr)[0]
    den = np.asarray(np.real_if_close(den)).ravel()
    return numvy, numr, den


def symbolic_transfer(A, B, C):
    s = sp.symbols("s")
    As = s*sp.eye(A.shape[0])-sp.Matrix(A)
    G = sp.simplify(sp.Matrix(C)*(As.inv()*sp.Matrix(B)))
    return G, s
