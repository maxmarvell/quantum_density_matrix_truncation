from ncon import ncon
import numpy as np
from uniform_MPS import *

def transfer_blocks(A: UniformMPS, B: UniformMPS):
    '''
        Compute two transfer matrices

        _AB : 1--A--3
                 |
              2--B*-4

        A_B : 1--B--3
                 |
              2--A*-4

    '''

    ABdag = ncon((A.value, np.conj(B.value)), ((-1, 1, -3), (-2, 1, -4)))
    BAdag = ncon((B.value, np.conj(A.value)), ((-1, 1, -3), (-2, 1, -4)))

    return TransferMatrix(ABdag), TransferMatrix(BAdag)

def compute_contraction(ABdag, BAdag, rA, rB, L):
    return ncon(((ABdag ** L).value , (BAdag ** L).value , rB, rA), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6)))

def cost(A: UniformMPS, B: UniformMPS, L: int):

    ABdag, BAdag = transfer_blocks(A, B)
    AdagA, _ = transfer_blocks(A, A)
    BdagB, _ = transfer_blocks(B, B)

    costAB = compute_contraction(ABdag, BAdag, A.r, B.r, L)
    costAA = compute_contraction(AdagA, AdagA, A.r, A.r, L)
    costBB = compute_contraction(BdagB, BdagB, B.r, B.r, L)

    return costAA + costBB - 2*costAB

if __name__ == "__main__":
    
    Da = 5
    Db = 3
    p = 2

    A = UniformMPS.random(Da, p)
    A.r_dominant()

    B = UniformMPS.random(Db, p)
    B.r_dominant()

    # assert cost 0
    assert np.allclose(cost(A, A, 3), 0j)
    assert np.allclose(cost(B, B, 10), 0j)

    pA = A.reduced_density_mat(3)
    pB = B.reduced_density_mat(3)

    assert np.allclose(ncon((pA, pA), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))) + ncon((pB, pB), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))) - 2*ncon((pA, pB), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))), cost(A, B, 3))

    ABdag, BAdag = transfer_blocks(A, B)

    _A = A.gauge()

    assert not np.allclose(A.value, _A.value)
    assert np.allclose(_A.r, ncon((_A.transfer_matrix.value, _A.r), ((-1, -2, 1, 2), (1, 2))))

    # check under unitary gauge transform the cost is zero
    assert np.allclose(cost(A, _A, 3), 0j)
    assert np.allclose(cost(A, _A, 10), 0j)

    # test fidelity
    L = 50
    r = A.fidelity(B)

    print(r**(2*L))
    print(cost(A, B, L))