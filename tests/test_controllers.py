import numpy as np
from config import DEFAULT_PARAMS
from models.linearized_bicycle import analytical_matrices
from controllers.pole_placement import pole_placement_gain
from controllers.ackermann import ackermann_gain
from controllers.lyapunov_lqr import lqr_gain, reference_precompensator


def test_controller_methods():
    A,B,*_=analytical_matrices(DEFAULT_PARAMS)
    desired=np.array([-8,-10])
    K,Kp=pole_placement_gain(A,B,desired)
    Ka=ackermann_gain(A,B,desired)
    assert np.allclose(np.sort_complex(Kp),np.sort_complex(desired))
    assert np.allclose(K,Ka,rtol=1e-6,atol=1e-6)
    Kl,P,res,lyap=lqr_gain(A,B)
    assert np.max(np.abs(res))<1e-7
    assert np.max(np.abs(lyap))<1e-7
    C=np.array([[0.0,1.0]])
    N=reference_precompensator(A,B,C,K)
    dc=C@np.linalg.solve(A-B@K,B)
    assert abs(1+N*float(dc[0,0]))<1e-8


def test_lqr_reference_precompensator_steady_state():
    A,B,*_=analytical_matrices(DEFAULT_PARAMS)
    K,P,res,lyap=lqr_gain(A,B)
    C=np.array([[0.0,1.0]])
    N=reference_precompensator(A,B,C,K)
    dc=C@np.linalg.solve(-(A-B@K),B)*N
    assert np.isclose(dc[0,0],1.0,rtol=1e-8,atol=1e-8)
