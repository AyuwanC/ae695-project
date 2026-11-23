"""Reusable fourth-order Runge-Kutta integration helpers."""
import numpy as np


def rk4_step(t, x, dt, rhs, u):
    """Advance one state interval using RK4 with a held input u."""
    x = np.asarray(x, dtype=float).reshape(-1)
    k1 = np.asarray(rhs(t, x, u), dtype=float).reshape(-1)
    k2 = np.asarray(rhs(t + 0.5*dt, x + 0.5*dt*k1, u),
                   dtype=float).reshape(-1)
    k3 = np.asarray(rhs(t + 0.5*dt, x + 0.5*dt*k2, u),
                   dtype=float).reshape(-1)
    k4 = np.asarray(rhs(t + dt, x + dt*k3, u),
                   dtype=float).reshape(-1)
    return x + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
