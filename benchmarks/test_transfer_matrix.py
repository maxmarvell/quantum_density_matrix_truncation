import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrix import TransferMatrix

# A constant value for the dimension `d` when testing `n`
FIXED_D = 5

# A constant value for the exponent `n` when testing `d`
FIXED_N = 10

@pytest.mark.parametrize("n", [2, 10, 50, 100])
def test_transfer_matrix_pow_scaling_with_n(benchmark, n):
    """
    Benchmarks __pow__ scaling with the exponent `n` for a fixed dimension `d`.
    """
    transfer_matrix = TransferMatrix.new(
        A=UniformMps.new(d=FIXED_D, p=2),
        B=UniformMps.new(d=FIXED_D, p=2)
    )
    benchmark(transfer_matrix.__pow__, n)


@pytest.mark.parametrize("d", [2, 5, 10, 15])
def test_transfer_matrix_pow_scaling_with_d(benchmark, d):
    """
    Benchmarks __pow__ scaling with the dimension `d` for a fixed exponent `n`.
    """
    transfer_matrix = TransferMatrix.new(
        A=UniformMps.new(d=d, p=2),
        B=UniformMps.new(d=d, p=2)
    )
    benchmark(transfer_matrix.__pow__, FIXED_N)


@pytest.mark.parametrize("n", [2, 10, 50, 100])
def test_transfer_matrix_derivative_scaling_with_n(benchmark, n):
    """
    Benchmarks __pow__ scaling with the exponent `n` for a fixed dimension `d`.
    """
    transfer_matrix = TransferMatrix.new(
        A=UniformMps.new(d=FIXED_D, p=2),
        B=UniformMps.new(d=FIXED_D, p=2)
    )

    E = transfer_matrix.__pow__(n)
    benchmark(E.derivative)


@pytest.mark.parametrize("d", [2, 5, 10, 15])
def test_transfer_matrix_derivative_scaling_with_d(benchmark, d):
    """
    Benchmarks __pow__ scaling with the exponent `n` for a fixed dimension `d`.
    """
    transfer_matrix = TransferMatrix.new(
        A=UniformMps.new(d=d, p=2),
        B=UniformMps.new(d=d, p=2)
    )

    E = transfer_matrix.__pow__(FIXED_N)
    benchmark(E.derivative)

@pytest.mark.parametrize("d", [2, 5, 10, 15])
def test_transfer_matrix_derivative(benchmark, d):
    """
    Benchmarks __pow__ scaling with the exponent `n` for a fixed dimension `d`.
    """
    E = TransferMatrix.new(
        A=UniformMps.new(d=d, p=2),
        B=UniformMps.new(d=d, p=2)
    )
    benchmark(E.derivative)
