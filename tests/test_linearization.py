import numpy as np
from config import DEFAULT_PARAMS
from models.linearized_bicycle import analytical_matrices,numerical_jacobians
from models.nonlinear_bicycle import nonlinear_dynamics


def test_straight_equilibrium():
    p=DEFAULT_PARAMS
    derivative=nonlinear_dynamics(0,np.zeros(2),0,p)
    assert np.allclose(derivative,np.zeros(2),atol=1e-10)


def test_matrices_benchmark_and_jacobians():
    p=DEFAULT_PARAMS
    A,B,*_=analytical_matrices(p)
    assert np.allclose(A,[[ -5.333333333,-18.933333333],[0.64,-6.4]],rtol=1e-5)
    assert np.allclose(B,[[53.333333333],[38.4]],rtol=1e-5)
    An,Bn=numerical_jacobians(p)
    assert np.allclose(An,A,rtol=1e-5,atol=1e-5)
    assert np.allclose(Bn,B,rtol=1e-5,atol=1e-5)
