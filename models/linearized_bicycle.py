"""Analytical, symbolic and numerical linearization of the bicycle model."""
import numpy as np
from scipy.optimize import approx_fprime
import sympy as sp
from config import VehicleParameters
from models.nonlinear_bicycle import nonlinear_dynamics


def analytical_matrices(params: VehicleParameters):
    """Construct A and B using axle cornering stiffness linearization."""
    m, Iz, a, b, vx = params.m, params.Iz, params.a, params.b, params.vx
    Cf, Cr = params.Cf, params.Cr
    A11 = -(Cf + Cr) / (m * vx)
    A12 = (b * Cr - a * Cf) / (m * vx) - vx
    A21 = (b * Cr - a * Cf) / (Iz * vx)
    A22 = -(a*a*Cf + b*b*Cr) / (Iz * vx)
    B1 = Cf / m
    B2 = a * Cf / Iz
    A = np.array([[A11, A12], [A21, A22]], dtype=float)
    B = np.array([[B1], [B2]], dtype=float)
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = np.zeros((2, 1))
    Cm = np.array([[0.0, 1.0]])
    Dm = np.zeros((1, 1))
    return A, B, C, D, Cm, Dm


def numerical_jacobians(params: VehicleParameters, x=np.zeros(2), u=0.0,
                        eps=1e-6):
    def fun_x(z):
        return nonlinear_dynamics(0.0, z, u, params)
    def fun_u(z):
        return nonlinear_dynamics(0.0, x, z[0], params)
    # Central differences use an adaptive per-variable perturbation.
    A = np.zeros((2, 2))
    for j in range(2):
        h = eps * max(1.0, abs(x[j]))
        xp = x.copy(); xm = x.copy()
        xp[j] += h; xm[j] -= h
        A[:, j] = (fun_x(xp) - fun_x(xm)) / (2*h)
    h = eps * max(1.0, abs(u))
    B = ((fun_u(np.array([u+h])) - fun_u(np.array([u-h]))) / (2*h)).reshape(2, 1)
    return A, B


def symbolic_matrices(params: VehicleParameters):
    """Return SymPy expressions for the linearized matrices."""
    vy, r, d, vx, a, b, m, Iz, Cf, Cr = sp.symbols(
        "vy r d vx a b m Iz Cf Cr", real=True)
    alpha_f = d - sp.atan((vy+a*r)/vx)
    alpha_r = -sp.atan((vy-b*r)/vx)
    # The evaluated-zero Pacejka derivatives are the axle cornering stiffnesses.
    Fyf = Cf*alpha_f
    Fyr = Cr*alpha_r
    f1 = (Fyf + Fyr)/m - vx*r
    f2 = (a*Fyf-b*Fyr)/Iz
    f = sp.Matrix([f1, f2])
    A = f.jacobian([vy, r]).subs({vy:0, r:0, d:0})
    B = f.jacobian([d]).subs({vy:0, r:0, d:0})
    subst = {vx:params.vx, a:params.a, b:params.b, m:params.m,
             Iz:params.Iz, Cf:params.Cf, Cr:params.Cr}
    return sp.N(A.subs(subst), 12), sp.N(B.subs(subst), 12)
