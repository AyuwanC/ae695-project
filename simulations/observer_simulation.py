"""Full- and reduced-order observer simulations."""
import numpy as np
from models.nonlinear_bicycle import nonlinear_dynamics
from observers.reduced_order import reduced_order_rhs
from simulations.rk4 import rk4_step


def simulate_full_observer(params, A, B, L, C, u_fun, t, x0, xhat0,
                           noisy_sigma=0.0, seed=695):
    rng = np.random.default_rng(seed)
    truth = np.zeros((len(t),2)); truth[0]=np.asarray(x0)
    for k in range(1,len(t)):
        dt=t[k]-t[k-1]; u=u_fun(t[k-1])
        def plant_rhs(ti_inner,x_inner,u_inner):
            return nonlinear_dynamics(ti_inner,x_inner,u_inner,params)
        truth[k]=rk4_step(t[k-1],truth[k-1],dt,plant_rhs,u)
    xhat = np.zeros((len(t),2)); xhat[0]=xhat0
    measured = truth[:,1]+rng.normal(0,noisy_sigma,len(t))
    for k in range(1,len(t)):
        dt=t[k]-t[k-1]
        y=measured[k-1]
        xh=xhat[k-1]
        u=u_fun(t[k-1])
        def observer_rhs(ti_inner,x_inner,u_inner):
            return A@x_inner+B[:,0]*u_inner+L[:,0]*(y-(C@x_inner)[0])
        xhat[k]=rk4_step(t[k-1],xh,dt,observer_rhs,u)
    return truth, xhat, measured


def simulate_reduced_observer(params, A, B, L, u_fun, t, x0, z0,
                              noisy_sigma=0.0, seed=695):
    rng = np.random.default_rng(seed)
    truth = np.zeros((len(t),2)); truth[0]=np.asarray(x0)
    for k in range(1,len(t)):
        dt=t[k]-t[k-1]; u=u_fun(t[k-1])
        def plant_rhs(ti_inner,x_inner,u_inner):
            return nonlinear_dynamics(ti_inner,x_inner,u_inner,params)
        truth[k]=rk4_step(t[k-1],truth[k-1],dt,plant_rhs,u)
    z = np.zeros(len(t)); z[0]=z0
    measured = truth[:,1]+rng.normal(0,noisy_sigma,len(t))
    A11,A12=A[0,0],A[0,1]; A21,A22=A[1,0],A[1,1]
    B1,B2=B[0,0],B[1,0]
    for k in range(1,len(t)):
        dt=t[k]-t[k-1]; r=measured[k-1]; u=u_fun(t[k-1])
        def reduced_rhs(ti_inner,z_inner,u_inner):
            return np.array([reduced_order_rhs(A,B,L,z_inner[0],r,u_inner)])
        z[k]=rk4_step(t[k-1],np.array([z[k-1]]),dt,reduced_rhs,u)[0]
    vyhat=z+L*measured
    xhat=np.column_stack([vyhat, measured])
    return truth, xhat, measured
