import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrix import SecondOrderTrotterizedTransferMatrix as SecondOrder
from helpers.transfer_matrix import SecondOrderTrotterizedTransferMatrix as NaiveSecondOrder
from qdmt.model import TransverseFieldIsing

FIXED_D = 5
FIXED_N = 10
FIXED_G = .2
FIXED_DT = .1

TFIM = TransverseFieldIsing(.2, .1)
U1, U2 = TFIM.trotter_second_order()

@pytest.fixture(params=[2, 4, 6, 8])
def A(request):
    return UniformMps.new(request.param, 2)

def test_new(benchmark, A):
    benchmark(SecondOrder.new, A, A, U1, U2)

@pytest.fixture()
def E(A):
    return SecondOrder.new(A, A, U1, U2)

@pytest.mark.parametrize("L", [2, 4, 8, 16])
def test_transfer_matrix_pow(benchmark, E, L):
    benchmark(E.__pow__, L)

@pytest.mark.parametrize("L", [2, 4, 8, 16])
def test_transfer_matrix_derivative(benchmark, E, L):
    benchmark(lambda: E.__pow__(L).derivative())

# @pytest.fixture()
# def naive_E(A):
#     return NaiveSecondOrder.new(A, A, U1, U2)

# @pytest.mark.parametrize("L", [2, 4, 8, 16])
# def test_naive_transfer_matrix_derivative(benchmark, naive_E, L):
#     benchmark(naive_E.derivative_of_power, L)

# @pytest.mark.parametrize("n", [2, 10, 50, 100])
# def test_transfer_matrix_pow_scaling_with_n(benchmark, n):
#     """
#     Benchmarks __pow__ scaling with the exponent `n` for a fixed dimension `d`.
#     """

#     tfim = TransverseFieldIsing(FIXED_G, FIXED_DT)
#     U1, U2 = tfim.trotter_second_order()

#     transfer_matrix = SecondOrder.new(
#         UniformMps.new(d=FIXED_D, p=2),
#         UniformMps.new(d=FIXED_D, p=2),
#         U1,
#         U2
#     )
#     benchmark(transfer_matrix.__pow__, n)


# @pytest.mark.parametrize("d", [2, 5, 8, 10])
# def test_transfer_matrix_pow_scaling_with_d(benchmark, d):
#     """
#     Benchmarks __pow__ scaling with the dimension `d` for a fixed exponent `n`.
#     """
#     tfim = TransverseFieldIsing(FIXED_G, FIXED_DT)
#     U1, U2 = tfim.trotter_second_order()

#     transfer_matrix = SecondOrder.new(
#         UniformMps.new(d=d, p=2),
#         UniformMps.new(d=d, p=2),
#         U1,
#         U2
#     )
#     benchmark(transfer_matrix.__pow__, FIXED_N)


# @pytest.mark.parametrize("n", [2, 10, 50, 100])
# def test_transfer_matrix_derivative_scaling_with_n(benchmark, n):
#     """
#     Benchmarks __pow__ scaling with the exponent `n` for a fixed dimension `d`.
#     """
#     tfim = TransverseFieldIsing(FIXED_G, FIXED_DT)
#     U1, U2 = tfim.trotter_second_order()

#     transfer_matrix = SecondOrder.new(
#         UniformMps.new(d=FIXED_D, p=2),
#         UniformMps.new(d=FIXED_D, p=2),
#         U1,
#         U2
#     )

#     E = transfer_matrix.__pow__(n)
#     benchmark(E.derivative)


# @pytest.mark.parametrize("d", [2, 5, 8, 10])
# def test_transfer_matrix_derivative_scaling_with_d(benchmark, d):
#     """
#     Benchmarks __pow__ scaling with the exponent `n` for a fixed dimension `d`.
#     """
#     tfim = TransverseFieldIsing(FIXED_G, FIXED_DT)
#     U1, U2 = tfim.trotter_second_order()

#     transfer_matrix = SecondOrder.new(
#         UniformMps.new(d=d, p=2),
#         UniformMps.new(d=d, p=2),
#         U1,
#         U2
#     )

#     E = transfer_matrix.__pow__(FIXED_N)
#     benchmark(E.derivative)
