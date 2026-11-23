import numpy as np
from config import DEFAULT_PARAMS
from models.linearized_bicycle import analytical_matrices
from analysis.controllability_observability import controllability,observability
from analysis.eigen_analysis import eigen_analysis
from analysis.canonical_forms import controllable_canonical,complex_modal,jordan_real


def test_properties():
    A,B,C,D,Cm,Dm=analytical_matrices(DEFAULT_PARAMS)
    vals,vecs,rows,coeff=eigen_analysis(A)
    assert np.all(vals.real<0)
    _,rc,_,_=controllability(A,B)
    _,ro,_,_=observability(A,Cm)
    assert rc==2 and ro==2
    cc=controllable_canonical(A,B,Cm)
    assert np.allclose(np.linalg.eigvals(cc["A"]),np.linalg.eigvals(A))
    vals,V,Aj,residual=complex_modal(A)
    assert np.max(np.abs(A@V-V@Aj))<1e-10
    assert residual<1e-10
    assert np.linalg.matrix_rank(V)==2
    vals,T,Ar=jordan_real(A)
    assert np.max(np.abs(A@T-T@Ar))<1e-10
