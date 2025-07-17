import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrix import TransferMatrix

FIXED_D = 5
FIXED_N = 10

@pytest.mark.parametrize("L", [2, 10, 50, 100])
def test_transfer_matrix_pow_scaling_with_n(benchmark, n):
    """
    Benchmarks __pow__ scaling with the exponent `n` for a fixed dimension `d`.
    """
    A=UniformMps.new(d=FIXED_D, p=2)
    B=UniformMps.new(d=FIXED_D, p=2)
    benchmark(transfer_matrix.__pow__, n)
