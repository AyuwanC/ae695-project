"""Nonlinear and linear open-loop response calculations."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from models.nonlinear_bicycle import nonlinear_dynamics, slip_angles, tire_forces


def simulate_responses(params, A, B, t, u_fun, x0):
    """Simulate nonlinear and linear models with the same input and initial state."""
    def nl(ti, x):
        return nonlinear_dynamics(ti, x, u_fun(ti), params)
    nl_sol = solve_ivp(nl, (t[0], t[-1]), np.asarray(x0), t_eval=t,
                       rtol=1e-9, atol=1e-11)
    if not nl_sol.success:
        raise RuntimeError(nl_sol.message)

    linear = np.zeros((len(t), 2))
    linear[0] = x0
    for k in range(1, len(t)):
        dt = t[k]-t[k-1]
        # Exact zero-order-hold propagation is used for piecewise steering inputs.
        M = np.vstack([np.hstack([A, B]),
                       np.hstack([np.zeros((1, 2)), np.zeros((1, 1))])])
        ad = expm(M*dt)
        augmented = np.array([linear[k-1,0], linear[k-1,1], u_fun((t[k-1]+t[k])/2)])
        linear[k] = (ad @ augmented)[:2]

    extra = {"alpha_f": [], "alpha_r": [], "Fyf": [], "Fyr": []}
    for k, x in enumerate(nl_sol.y.T):
        af, ar = slip_angles(x, u_fun(t[k]), params)
        fyf, fyr = tire_forces(x, u_fun(t[k]), params)
        extra["alpha_f"].append(np.rad2deg(af))
        extra["alpha_r"].append(np.rad2deg(ar))
        extra["Fyf"].append(fyf); extra["Fyr"].append(fyr)
    for key in extra:
        extra[key] = np.asarray(extra[key])
    return nl_sol.y.T, linear, extra
