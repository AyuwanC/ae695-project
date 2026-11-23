"""Nonlinear dynamic bicycle model with a Pacejka tire-force formulation."""
import numpy as np
from config import VehicleParameters
from models.pacejka_tire import pacejka_lateral_force


def slip_angles(x, delta_f: float, params: VehicleParameters):
    vy, r = np.asarray(x).reshape(2)
    alpha_f = delta_f - np.arctan((vy + params.a * r) / params.vx)
    alpha_r = -np.arctan((vy - params.b * r) / params.vx)
    return float(alpha_f), float(alpha_r)


def tire_forces(x, delta_f: float, params: VehicleParameters):
    alpha_f, alpha_r = slip_angles(x, delta_f, params)
    Fyf = float(pacejka_lateral_force(alpha_f, params.Bf, params.pacejka_C,
                                      params.Df, params.pacejka_E))
    Fyr = float(pacejka_lateral_force(alpha_r, params.Br, params.pacejka_C,
                                      params.Dr, params.pacejka_E))
    return Fyf, Fyr


def nonlinear_dynamics(t, x, u, params: VehicleParameters):
    """Return [dvy/dt, dr/dt] for state x and front steering u."""
    vy, r = np.asarray(x).reshape(2)
    delta_f = float(np.asarray(u).reshape(-1)[0])
    Fyf, Fyr = tire_forces([vy, r], delta_f, params)
    dvy = (Fyf * np.cos(delta_f) + Fyr) / params.m - params.vx * r
    dr = (params.a * Fyf * np.cos(delta_f) - params.b * Fyr) / params.Iz
    return np.array([dvy, dr], dtype=float)

