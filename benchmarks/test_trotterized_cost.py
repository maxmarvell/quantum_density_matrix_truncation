import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.cost import EvolvedHilbertSchmidt
from qdmt.model import TransverseFieldIsing

@pytest.fixture(params=[2, 4, 8, 16])
def L(request):
    return request.param

@pytest.fixture(params=[2, 4, 6, 8])
def f(request, L):
    A = UniformMps.new(request.param, 2)
    tfim = TransverseFieldIsing(0.2, 0.1)
    return EvolvedHilbertSchmidt(A, A, tfim, L, trotterization_order=2)

def test_cost(benchmark, f):
    benchmark(f.cost)