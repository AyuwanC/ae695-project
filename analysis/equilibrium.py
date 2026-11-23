"""Numerical equilibrium calculation for the nonlinear bicycle model."""
import numpy as np
from scipy.optimize import root
from models.nonlinear_bicycle import nonlinear_dynamics, slip_angles, tire_forces


def find_equilibrium(params, steering_deg=0.0):
    """Solve f(xe, ue)=0 and reject nonphysical or unconverged solutions."""
    ue = np.deg2rad(steering_deg)
    sol = root(lambda x: nonlinear_dynamics(0.0, x, ue, params),
               np.zeros(2), method="hybr")
    residual = np.linalg.norm(nonlinear_dynamics(0.0, sol.x, ue, params))
    if not sol.success or residual > 1e-7 or not np.all(np.isfinite(sol.x)):
        raise RuntimeError(f"Equilibrium solve failed at {steering_deg} deg")
    af, ar = slip_angles(sol.x, ue, params)
    fyf, fyr = tire_forces(sol.x, ue, params)
    if abs(sol.x[1]) > 10.0 or abs(sol.x[0]) > 100.0:
        raise RuntimeError("Equilibrium is physically implausible")
    return {"steering_deg": steering_deg, "vy": sol.x[0], "r": sol.x[1],
            "alpha_f_deg": np.rad2deg(af), "alpha_r_deg": np.rad2deg(ar),
            "Fyf": fyf, "Fyr": fyr, "residual": residual}
