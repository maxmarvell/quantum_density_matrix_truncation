
from tests.helpers.cost import EvolvedHilbertSchmidt as Naive
from qdmt.cost import EvolvedHilbertSchmidt as Efficient
from qdmt.model import TransverseFieldIsing
from qdmt.fixed_point import RightFixedPoint
from qdmt.uniform_mps import UniformMps
import numpy as np
import pytest

@pytest.fixture
def A() -> UniformMps:
    return UniformMps.new(5, 2)

@pytest.fixture
def B() -> UniformMps:
    return UniformMps.new(5, 2)

@pytest.fixture(params=[2, 4, 8])
def L(request: pytest.FixtureRequest) -> int:
    return request.param

# def test(A, B, L):

#     model = TransverseFieldIsing(0.1, 0.1)
#     rB = RightFixedPoint.from_mps(B)

#     f = Efficient(A, model, L, trotterization_order=1)
#     f_naive = Naive(A, model, L, trotterization_order=2)

#     assert np.allclose(f.cost(B, rB), f_naive.cost(B, rB))
#     assert np.allclose(f.derivative(B, rB), f_naive.derivative(B, rB))


def test_second_trotterization(A, B, L):

    model = TransverseFieldIsing(0.1, 0.1)
    rB = RightFixedPoint.from_mps(B)

    f = Efficient(A, model, L, trotterization_order=2)
    f_naive = Naive(A, model, L, trotterization_order=2)

    assert np.allclose(f.cost(B, rB), f_naive.cost(B, rB))
    assert np.allclose(f.derivative(B, rB), f_naive.derivative(B, rB))

