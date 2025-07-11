
from tests.helpers.cost import EvolvedHilbertSchmidt as Naive
from src.cost import EvolvedHilbertSchmidt as Efficient
from src.model import TransverseFieldIsing
from src.uniform_mps import UniformMps
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

def test(A, B, L):

    model = TransverseFieldIsing(0.1, 0.1)
    f = Efficient(A, B, model, L, trotterization_order=1)
    f_naive = Naive(A, B, model, L)

    assert np.allclose(f.cost(), f_naive.cost())
    assert np.allclose(f.derivative(), f_naive.derivative())


def test_second_trotterization(A, B, L):

    model = TransverseFieldIsing(0.1, 0.1)

    f = Efficient(A, B, model, L)
    f_naive = Naive(A, B, model, L, order=2)

    assert np.allclose(f.cost(), f_naive.cost())
    assert np.allclose(f.derivative(), f_naive.derivative())

