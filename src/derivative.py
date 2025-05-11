import numpy as np
from ncon import ncon
from transfer_matrix import FirstOrderTrotterizedTransferMatrix, TransferMatrix
from cost import transfer_blocks
from fixed_point import RightFixedPoint
from uniform_MPS import UniformMPS


def derivative(A: UniformMPS, B: UniformMPS, U1: np.ndarray, U2: np.ndarray, L: int, rA: RightFixedPoint = None, rB: RightFixedPoint = None):

    if L % 2 != 0:
        return NotImplemented

    Da, Db = A.d, B.d
    ABdag = FirstOrderTrotterizedTransferMatrix.new(A, B, U1, U2)
    BAdag = FirstOrderTrotterizedTransferMatrix.new(B, A, np.conj(U2).transpose(2, 3, 0, 1), np.conj(U1).transpose(2, 3, 0, 1))
    BBdag, _ = transfer_blocks(B, B)

    if rA is None:
        E = TransferMatrix.new(A, A)
        rA = RightFixedPoint(E).tensor

    if rB is None:
        rB = RightFixedPoint(BBdag)

    T_BB = BBdag ** L
    T_BA = BAdag ** (L / 2)
    T_AB = ABdag ** (L / 2)

    res = np.zeros_like(B.tensor)

    # rho(B)^2 contribution
    D = T_BB.derivative()
    tensors = (D, T_BB.tensor, rB.tensor, rB.tensor)
    indices = ((1, 2, 3, 4, -1, -2, -3), (2, 1, 5, 6), (3, 6), (4, 5))
    res += 2 * ncon(tensors, indices) / (np.sqrt(Db) * np.sqrt(Db))

    # right fixed point
    tensors = (T_BB.tensor, T_BB.tensor, rB.tensor)
    indices = ((1, 2, 3, -1), (2, 1, -2, 4), (3, 4))
    v = ncon(tensors, indices)
    res += 2 * rB.derivative(v) / (np.sqrt(Db) * np.sqrt(Db))

    # rho(A(t+dt))rho(B) contribution
    D = T_AB.derivative()
    tensors = (A.tensor, A.tensor, U1, D, T_BA.tensor, np.conj(U1), A.conj, A.conj, rA, rB.tensor)
    indices = ((1, 2, 3), (3, 4, 5), (2, 4, 6, 7), (5, 7, 8, 9, 10, 11, -1, -2, -3), (8, 6, 1, 12, 13, 14), (15, 16, 13, 10), (14, 15, 17), (17, 16, 18), (9, 18), (12, 11))
    res -= 2 * ncon(tensors, indices) / (np.sqrt(Da) * np.sqrt(Db))

    # right fixed point
    tensors = (A.tensor, A.tensor, U1, T_AB.tensor, T_BA.tensor, np.conj(U1), A.conj, A.conj, rA)
    indices = ((1, 2, 3), (3, 4, 5), (2, 4, 6, 7), (5, 7, 8, 9, 10, -1), (8, 6, 1, -2, 11, 12), (13, 14, 11, 10), (12, 13, 15), (15, 14, 16), (9, 16))
    v = ncon(tensors, indices)
    res -= 2 * rB.derivative(v) / (np.sqrt(Da) * np.sqrt(Db))

    return res

if __name__ == "__main__":

    Da = 5
    Db = 3
    p = 2

    A = UniformMPS.from_random(Da, p)
    B = UniformMPS.from_random(Db, p)

    # Pauli gates
    Sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
    Sy = np.array([[0, 1j],
                [-1j, 0]], dtype=complex)
    Sz = np.array([[1, 0],
                [0, -1]], dtype=complex)

    S = {'I': np.eye(2, dtype=complex), 'X': Sx, 'Y': Sy, 'Z': Sz}

    # generate unitary
    from scipy.linalg import expm

    g = 0.1
    delta = 0.1
    ZZ = np.kron(S['Z'], S['Z'])
    XI = np.kron(S['X'], S['I'])
    U = expm(-1j*delta*(ZZ+g*XI)).reshape(2, 2, 2, 2)

    assert np.allclose(ncon((U, np.conj(U)), ((-1, -2, 1, 2), (-3, -4, 1, 2))).reshape(4, 4), np.eye(4))

    res = derivative(A, B, U, U, 2)
    print(res.shape)