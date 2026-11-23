import numpy as np
from config import DEFAULT_PARAMS
from models.linearized_bicycle import analytical_matrices
from observers.full_order import observer_pole_gain,observer_ackermann
from observers.reduced_order import reduced_order_gain


def test_observers():
    A,B,C,D,Cm,Dm=analytical_matrices(DEFAULT_PARAMS)
    desired=np.array([-16,-18])
    L,poles=observer_pole_gain(A,Cm,desired)
    La,apoles=observer_ackermann(A,Cm,desired)
    assert np.all(np.sort_complex(poles).real<0)
    assert np.allclose(np.sort_complex(poles),np.sort_complex(apoles))
    assert np.allclose(L,La,rtol=1e-5,atol=1e-5)
    Lr,errpole=reduced_order_gain(A,B,-12)
    assert abs(errpole+12)<1e-10


def test_reduced_observer_coordinate_initialization():
    A,B,C,D,Cm,Dm=analytical_matrices(DEFAULT_PARAMS)
    Lr,errpole=reduced_order_gain(A,B,-12)
    vyhat0=0.0
    r0=-0.1
    z0=vyhat0-Lr*r0
    vyhat=z0+Lr*r0
    assert np.isclose(vyhat,vyhat0)
    assert abs(errpole+12)<1e-10
