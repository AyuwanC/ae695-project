"""Combined nonlinear controller-observer closed-loop simulations."""
import numpy as np
from models.nonlinear_bicycle import nonlinear_dynamics
from observers.reduced_order import reduced_order_rhs
from simulations.rk4 import rk4_step


def simulate_combined(params, A, B, K, Nbar, r_ref, t, observer="full",
                      L=None, Lr=None, noise_sigma=0.0, seed=695,
                      x0=None, xhat0=None, steering_limit=np.deg2rad(20),
                      return_diagnostics=False):
    rng=np.random.default_rng(seed)
    x0=np.zeros(2) if x0 is None else np.asarray(x0)
    xhat0=np.array([0.0,0.0]) if xhat0 is None else np.asarray(xhat0)
    states=np.zeros((len(t),2)); states[0]=x0
    estimates=np.zeros((len(t),2)); estimates[0]=xhat0
    controls=np.zeros(len(t)); measured=np.zeros(len(t))
    r_initial=x0[1]+rng.normal(0.0,noise_sigma)
    measured[0]=r_initial
    z=float(xhat0[0]-Lr*r_initial) if observer=="reduced" else 0.0
    estimates[0]=xhat0
    C=np.array([[0.0,1.0]])
    observer_errors=np.zeros((len(t),2))
    observer_errors[0]=states[0]-estimates[0]
    references=np.zeros(len(t))
    references[0]=r_ref(t[0]) if callable(r_ref) else r_ref[0]
    for k in range(1,len(t)):
        dt=t[k]-t[k-1]
        ti=t[k-1]
        ref=r_ref(ti) if callable(r_ref) else r_ref[k-1]
        references[k-1]=ref
        r_meas=states[k-1,1]+rng.normal(0,noise_sigma)
        measured[k-1]=r_meas
        if observer=="full":
            xh=estimates[k-1]
            u_unsat=-K@xh+Nbar*ref
            u=float(np.clip(np.asarray(u_unsat).reshape(-1)[0],
                            -steering_limit,steering_limit))
            def observer_rhs(ti, xh_inner, u_inner):
                return A@xh_inner+B[:,0]*u_inner+L[:,0]*(r_meas-(C@xh_inner)[0])
            estimates[k]=rk4_step(ti,xh,dt,observer_rhs,u)
        else:
            u_current=-float(K[0,0]*estimates[k-1,0]+K[0,1]*r_meas)+Nbar*ref
            u=float(np.clip(u_current,-steering_limit,steering_limit))
            def reduced_rhs(ti,z_inner,u_inner):
                z_value=float(np.asarray(z_inner).reshape(-1)[0])
                return np.array([reduced_order_rhs(A,B,Lr,z_value,r_meas,u_inner)])
            z=float(rk4_step(ti,np.array([z]),dt,reduced_rhs,u)[0])
            estimates[k]=np.array([z+Lr*r_meas,r_meas])
        # If initial estimate branch commanded at k=1, recompute state with u.
        def plant_rhs(ti_inner,x_inner,u_inner):
            return nonlinear_dynamics(ti_inner,x_inner,u_inner,params)
        states[k]=rk4_step(ti,states[k-1],dt,plant_rhs,u)
        controls[k-1]=u
        if observer=="full":
            observer_errors[k]=states[k]-estimates[k]
        else:
            observer_errors[k]=states[k]-estimates[k]
    measured[-1]=states[-1,1]+rng.normal(0,noise_sigma)
    ref_last=r_ref(t[-1]) if callable(r_ref) else r_ref[-1]
    references[-1]=ref_last
    controls[-1]=controls[-2]
    observer_errors[-1]=states[-1]-estimates[-1]
    if return_diagnostics:
        return states, estimates, controls, measured, references, observer_errors
    return states, estimates, controls, measured
