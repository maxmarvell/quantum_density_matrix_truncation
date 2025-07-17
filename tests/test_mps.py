import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.utils.mps import trotter_step
from qdmt.model import TransverseFieldIsing
import numpy as np
from ncon import ncon

@pytest.fixture(params=[
    (5, 2, True),
    (5, 2, False)
])
def A(request: pytest.FixtureRequest) -> UniformMps:
    b, p, left_canonical = request.param
    return UniformMps.new(b, p, left_canonical)

@pytest.fixture(params=[5, 3])
def Al(request: pytest.FixtureRequest) -> UniformMps:
    d = request.param
    return UniformMps.new(d, 2)

@pytest.fixture(params=[5, 3])
def Bl(request: pytest.FixtureRequest) -> UniformMps:
    d = request.param
    return UniformMps.new(d, 2)


def test_normalized(A: UniformMps):
    assert A.is_normalized()


def test_mps_canonical_forms(A: UniformMps):

    # assert np.allclose(ncon((A.tensor, A.conj), ((1, 2, -2), (1, 2, -1))), np.eye(A.d))

    Al, Ac, Ar, C = A.mixed_canonical()
    assert np.allclose(ncon((Al, Al.conj()), ((1, 2, -2), (1, 2, -1))), np.eye(A.d))
    assert np.allclose(ncon((Ar, Ar.conj()), ((-1, 2, 1), (-2, 2, 1))), np.eye(A.d))


def test_conjugate_chain(Al: UniformMps):

    chain = Al.mps_chain(5)

    Al_conj = UniformMps(Al.tensor.conj())
    chain_conj = Al_conj.mps_chain(5)

    assert np.allclose(chain_conj, chain.conj())


def test_mps_chain(Al: UniformMps):


    model = TransverseFieldIsing(0.1, 0.1)
    U1, U2 = model.trotter_second_order()

    mps = Al.mps_chain(6)
    mps = trotter_step(mps, U1)
    mps = trotter_step(mps, U2, start=1)
    mps = trotter_step(mps, U1, start=2, stop=3)