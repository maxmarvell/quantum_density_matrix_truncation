import numpy as np
import pytest
from src.utils.geometry import *
from src.uniform_mps import UniformMps
from src.cost import HilbertSchmidt
from ncon import ncon

class DummyProjector(IdentityProjector):
    pass

def test_grassman_projector():

    d, p = 5, 2
    A = UniformMps.new(d, p)
    B = UniformMps.new(d, p)

    P = GrassmanProjector()
    f = HilbertSchmidt(A, B, 5)

    D = f.derivative()
    G = P.project(B.tensor, D)

    assert np.allclose(ncon((B.conj, G), ((1, 2, -1), (1, 2, -2))), np.zeros((d, d)))

# def test_grassman_euclidian_retraction():

#     d, p = 5, 2
#     A = UniformMps.new(d, p)
#     B = UniformMps.new(d, p)

#     P = GrassmanProjector()
#     f = HilbertSchmidt(A, B, 5)
#     R = Retraction(P, False)

#     D = f.derivative()
#     B_new = R.update(B.tensor, D, 0.1)
#     assert np.allclose(ncon((np.conj(B_new), B_new), ((1, 2, -1), (1, 2, -2))), np.eye(d, dtype=np.complex128), rtol=1e-12)


def test():
    d, p = 5, 2
    A = UniformMps.new(d, p)
    B = UniformMps.new(d, p)

    P = GrassmanProjector()
    f = HilbertSchmidt(A, B, 5)
    R = TestRetraction(P, True)

    D = f.derivative()
    B_new = R.update(B.tensor, D, 0.1, f.rB.tensor)
    assert np.allclose(ncon((np.conj(B_new), B_new), ((1, 2, -1), (1, 2, -2))), np.eye(d, dtype=np.complex128), rtol=1e-12)
