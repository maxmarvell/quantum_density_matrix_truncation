import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrix import SecondOrderTrotterizedTransferMatrix as SecondOrder
from qdmt.model import TransverseFieldIsing

TFIM = TransverseFieldIsing(.2, .1)
U1, U2 = TFIM.trotter_second_order()

@pytest.fixture()
def A():
    return UniformMps.new(8, 2)

@pytest.fixture()
def E(A):
    return SecondOrder.new(A, A, U1, U2)

@pytest.mark.parametrize("L", [i for i in range(4, 100, 4)])
def test_transfer_matrix_pow(benchmark, E, L):
    benchmark(E.__pow__, L)
