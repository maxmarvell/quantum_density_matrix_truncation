import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.cost import EvolvedHilbertSchmidt
from qdmt.model import TransverseFieldIsing
from qdmt.transfer_matrix import (
    SecondOrderTrotterizedTransferMatrix as SecondOrder,
)

FIXED_L = 10
FIXED_D = 5

@pytest.fixture(params=[2, 4, 8, 16])
def L(request):
    return request.param

@pytest.fixture(params=[2, 4, 6, 8])
def f(request, L):
    A = UniformMps.new(request.param, 2)
    tfim = TransverseFieldIsing(0.2, 0.1)
    return EvolvedHilbertSchmidt(A, A, tfim, L, trotterization_order=2)

def test_compute_derivative_rho_A_rho_B(benchmark, f):
    benchmark(f._compute_derivative_rho_A_rho_B)

def test_compute_derivative_rho_B_rho_B(benchmark, f):
    benchmark(f._compute_derivative_rho_B_rho_B)

def test_contract_rhoA_rhoB_central_derivative(benchmark, f):
    ABdag = SecondOrder.new(f.A, f.B, f.U1, f.U2)
    BAdag = SecondOrder.new(f.B, f.A, f.U1.conj().transpose(2, 3, 0, 1), f.U2.conj().transpose(2, 3, 0, 1))
    T_BA = BAdag.__pow__(f.L // 2)
    T_AB = ABdag.__pow__(f.L // 2)
    D = T_AB.derivative()
    benchmark(f._contract_rhoA_rhoB_central_derivative, D, T_AB, T_BA)

def test_contract_rhoA_rhoB_fixed_point_derivative(benchmark, f):
    ABdag = SecondOrder.new(f.A, f.B, f.U1, f.U2)
    BAdag = SecondOrder.new(f.B, f.A, f.U1.conj().transpose(2, 3, 0, 1), f.U2.conj().transpose(2, 3, 0, 1))
    T_BA = BAdag.__pow__(f.L // 2)
    T_AB = ABdag.__pow__(f.L // 2)
    benchmark(f._contract_rhoA_rhoB_fixed_point_derivative, T_AB, T_BA)