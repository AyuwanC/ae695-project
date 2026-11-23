"""Closed-loop nonlinear controller simulations."""
import numpy as np
from scipy.integrate import solve_ivp
from models.nonlinear_bicycle import nonlinear_dynamics


def simulate_nonlinear_controller(params, A, B, K, Nbar, t, r_ref,
                                  x0=None, steering_limit=np.deg2rad(20)):
    x0 = np.zeros(2) if x0 is None else np.asarray(x0)
    controls = np.zeros(len(t))
    states = np.zeros((len(t), 2))
    states[0] = x0

    def rhs(ti, x):
        k = min(np.searchsorted(t, ti, side="right"), len(t)-1)
        ref = r_ref(ti) if callable(r_ref) else r_ref[k]
        u_unsat = -K@x+Nbar*ref
        u = float(np.clip(np.asarray(u_unsat).reshape(-1)[0],
                          -steering_limit, steering_limit))
        return nonlinear_dynamics(ti, x, u, params)

    sol = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t,
                    rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    states = sol.y.T
    for k, ti in enumerate(t):
        ref = r_ref(ti) if callable(r_ref) else r_ref[k]
        u_unsat = -K@states[k]+Nbar*ref
        controls[k] = np.clip(np.asarray(u_unsat).reshape(-1)[0],
                              -steering_limit, steering_limit)
    return states, controls
