import numpy as np
from scipy.linalg import expm

class Pauli():
    Sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
    Sy = np.array([[0, 1j],
                [-1j, 0]], dtype=complex)
    Sz = np.array([[1, 0],
                [0, -1]], dtype=complex)
    
    I = np.eye(2, dtype=complex)

    # S = {'I': np.eye(2, dtype=complex), 'X': Sx, 'Y': Sy, 'Z': Sz}

def transverse_ising(delta:float, g:float = 0.1):
    ZZ = np.kron(Pauli.Sz, Pauli.Sz)    
    XI = np.kron(Pauli.Sx, Pauli.I)
    return expm(-1j*delta*(ZZ+g*XI)).reshape(2, 2, 2, 2)


if __name__ == "__main__":

    from ncon import ncon
    U = transverse_ising(0.1)

    assert np.allclose(ncon((U, np.conj(U)), ((-1, -2, 1, 2), (-3, -4, 1, 2))).reshape(4, 4), np.eye(4))
    assert np.allclose(U.reshape(4, 4) @ np.conj(U).transpose(2, 3, 0, 1).reshape(4, 4), np.eye(4))