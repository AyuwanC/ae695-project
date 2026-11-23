import numpy as np
from config import DEFAULT_PARAMS
from models.pacejka_tire import pacejka_lateral_force


def test_tire_zero_and_sign():
    p=DEFAULT_PARAMS
    F=lambda a: pacejka_lateral_force(a,p.Bf,p.pacejka_C,p.Df,p.pacejka_E)
    assert abs(F(0.0)) < 1e-12
    assert F(0.05)>0 and F(-0.05)<0
    assert abs(F(0.2)+F(-0.2)) < 1e-9


def test_small_slip_gradient():
    p=DEFAULT_PARAMS
    D=p.pacejka_C*p.Bf*p.Df
    assert np.isclose(D,p.Cf,rtol=1e-9)


def test_force_saturation():
    p=DEFAULT_PARAMS
    large=pacejka_lateral_force(5.0,p.Bf,p.pacejka_C,p.Df,p.pacejka_E)
    assert abs(large) <= p.Df*1.01
