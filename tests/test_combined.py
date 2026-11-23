import numpy as np
from config import DEFAULT_PARAMS
from models.linearized_bicycle import analytical_matrices
from controllers.pole_placement import pole_placement_gain
from controllers.lyapunov_lqr import lqr_gain, reference_precompensator
from observers.full_order import observer_pole_gain
from observers.reduced_order import reduced_order_gain
from simulations.combined_simulation import simulate_combined


def test_combined_pole_full_tracks_positive_step():
    p=DEFAULT_PARAMS
    A,B,C,D,Cm,Dm=analytical_matrices(p)
    K,poles=pole_placement_gain(A,B,np.array([-8.0,-10.0]))
    C_yaw=np.array([[0.0,1.0]])
    N=reference_precompensator(A,B,C_yaw,K)
    L,opoles=observer_pole_gain(A,Cm,np.array([-16.0,-18.0]))
    t=np.linspace(0,10,501)
    ref=lambda ti: np.deg2rad(2.0)
    states,estimates,controls,measured,refs,errors=simulate_combined(
        p,A,B,K,N,ref,t,"full",L=L,return_diagnostics=True)
    rmse=np.sqrt(np.mean((refs-states[:,1])**2))
    assert rmse<0.2
    assert states[-1,1]>np.deg2rad(1.8)
    assert np.max(np.abs(controls))<=np.deg2rad(20)+1e-12
    assert np.all(np.isfinite(states)) and np.all(np.isfinite(estimates))


def test_combined_reduced_tracks_positive_step():
    p=DEFAULT_PARAMS
    A,B,C,D,Cm,Dm=analytical_matrices(p)
    K,poles=pole_placement_gain(A,B,np.array([-8.0,-10.0]))
    N=reference_precompensator(A,B,np.array([[0.0,1.0]]),K)
    Lr,rpole=reduced_order_gain(A,B,-12.0)
    t=np.linspace(0,10,501)
    ref=lambda ti: np.deg2rad(2.0)
    states,estimates,controls,measured,refs,errors=simulate_combined(
        p,A,B,K,N,ref,t,"reduced",Lr=Lr,return_diagnostics=True)
    rmse=np.sqrt(np.mean((refs-states[:,1])**2))
    assert rmse<0.2
    assert states[-1,1]>np.deg2rad(1.8)
    assert np.max(np.abs(controls))<=np.deg2rad(20)+1e-12


def test_combined_sinusoid_and_saturation_dimensions():
    p=DEFAULT_PARAMS
    A,B,*_=analytical_matrices(p)
    K,poles=pole_placement_gain(A,B,np.array([-8.0,-10.0]))
    N=reference_precompensator(A,B,np.array([[0.0,1.0]]),K)
    L,opoles=observer_pole_gain(A,np.array([[0.0,1.0]]),np.array([-16.0,-18.0]))
    t=np.linspace(0,10,501)
    ref=lambda ti: np.deg2rad(2.0*np.sin(0.5*ti))
    states,estimates,controls,measured,refs,errors=simulate_combined(
        p,A,B,K,N,ref,t,"full",L=L,return_diagnostics=True)
    assert states.shape==(501,2) and estimates.shape==(501,2)
    assert controls.shape==(501,) and measured.shape==(501,)
    assert np.all(np.isfinite(states))
