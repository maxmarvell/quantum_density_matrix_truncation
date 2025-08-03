import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.cost import EvolvedHilbertSchmidt
from qdmt.model import TransverseFieldIsing
from qdmt.fixed_point import RightFixedPoint
from helpers.cost import EvolvedHilbertSchmidt

FIXED_L = 10
FIXED_D = 5

@pytest.fixture(params=[2, 4, 6])
def L(request):
    return request.param

@pytest.fixture(params=[2, 4, 6, 8, 10, 12])
def f(request, L):
    A = UniformMps.new(request.param, 2)
    tfim = TransverseFieldIsing(0.2, 0.1)
    return EvolvedHilbertSchmidt(A, tfim, L, trotterization_order=2)

def test_compute_cost(benchmark, f):
    B = f.A
    rB = RightFixedPoint.from_mps(B)
    benchmark(f.cost, B, rB)

def test_compute_derivative(benchmark, f):
    B = f.A
    rB = RightFixedPoint.from_mps(B)
    benchmark(f.derivative, B, rB)