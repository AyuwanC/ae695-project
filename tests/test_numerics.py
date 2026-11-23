import numpy as np
from config import DEFAULT_PARAMS
from models.linearized_bicycle import analytical_matrices
from models.nonlinear_bicycle import nonlinear_dynamics
from analysis.state_transition import phi_matrix,step_response
from simulations.rk4 import rk4_step


def test_state_transition():
    A,B,*_=analytical_matrices(DEFAULT_PARAMS)
    assert np.allclose(phi_matrix(A,0.0),np.eye(2),rtol=0,atol=1e-12)
    zi,zs,total=step_response(A,B,np.array([1.0,-0.1]),np.deg2rad(2),np.linspace(0,2,21))
    assert zi.shape==(2,21) and zs.shape==(2,21) and total.shape==(2,21)
    assert np.all(np.isfinite(total))


def test_rk4_converges():
    p=DEFAULT_PARAMS
    def rhs(t,x,u):
        return nonlinear_dynamics(t,x,u,p)
    x0=np.array([0.0,0.0]); u=np.deg2rad(0.5)
    dt=0.02
    coarse=rk4_step(0.0,x0,dt,rhs,u)
    fine=rk4_step(0.0,x0,dt/2,rhs,u)
    fine=rk4_step(dt/2,fine,dt/2,rhs,u)
    assert np.max(np.abs(coarse-fine))<1e-4


def test_rk4_plant_converges_at_smaller_step():
    p=DEFAULT_PARAMS
    def rhs(t,x,u):
        return nonlinear_dynamics(t,x,u,p)
    x0=np.array([0.0,0.0]); u=np.deg2rad(10.0)
    coarse=rk4_step(0.0,x0,0.02,rhs,u)
    fine=rk4_step(0.0,x0,0.01,rhs,u)
    fine=rk4_step(0.01,fine,0.01,rhs,u)
    assert np.max(np.abs(coarse-fine))<1e-5
