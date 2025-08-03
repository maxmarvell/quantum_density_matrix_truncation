import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrix import SecondOrderTrotterizedTransferMatrix as SecondOrder
from qdmt.model import TransverseFieldIsing

TFIM = TransverseFieldIsing(.2, .1)
U1, U2 = TFIM.trotter_second_order()

@pytest.fixture(params=[i for i in range(1, 29)])
def A(request):
    return UniformMps.new(request.param, 2)

@pytest.fixture()
def E(A):
    return SecondOrder.new(A, A, U1, U2)

def test_transfer_matrix_pow(benchmark, E):
    benchmark(E.__pow__, 16)
