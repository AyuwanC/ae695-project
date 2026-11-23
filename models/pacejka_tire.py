"""Pacejka simplified Magic Formula lateral tire force model."""
import numpy as np


def pacejka_lateral_force(alpha, B: float, C: float, D: float, E: float):
    """Return lateral force for scalar or array slip angle (radians)."""
    for name, value in (("B", B), ("C", C), ("D", D), ("E", E)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    alpha_array = np.asarray(alpha, dtype=float)
    if not np.all(np.isfinite(alpha_array)):
        raise ValueError("alpha contains non-finite values")
    B_alpha = B * alpha_array
    inner = B_alpha - E * (B_alpha - np.arctan(B_alpha))
    return D * np.sin(C * np.arctan(inner))

