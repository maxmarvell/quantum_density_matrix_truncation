from ncon import ncon
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.stats import unitary_group
from typing import Self

class UniformMPS():

    def __init__(self, A: np.ndarray, normalise:bool = True):
        self.bond_dim, self.phys_dim, _ = A.shape
        if normalise:
            self.value = self.normalise(A)
        else:
            self.value = A

    def __add__(self, other: Self):
        return UniformMPS(self.value + other.value, normalise=False)
    
    def __sub__(self, other: Self):
        return UniformMPS(self.value - other.value, normalise=False)
    
    def __mul__(self, other):
        return UniformMPS(self.value * other)
    
    __rmul__ = __mul__

    @staticmethod
    def random(bond_dim: int, phys_dim: int):
        """
            Initialise a random normalised uniform MPS states in left canonical form.

            args:
                bond_dim: Bond dimension of the uniform matrix product state
                phys_dim: Physical dimension of the uniform matrix product state

            returns:
                Random normalised uniform matrix product states, shape (bond_dim, phys_dim, bond_dim)
        """
        U = unitary_group.rvs(bond_dim*phys_dim)
        A = U.reshape(bond_dim, phys_dim, bond_dim, phys_dim)[:, :, :, 0]
        return UniformMPS(A)
    
    @staticmethod
    def normalise(A: np.ndarray):
        D = A.shape[0]
        E = TransferMatrix.construct_transfer_matrix(A)
        norm = eigs(np.reshape(E.value, (D ** 2, D ** 2)), k=1, which='LM', return_eigenvectors=False)
        A /= np.sqrt(norm)
        return A
    
    @property
    def transfer_matrix(self, A: np.ndarray = None):

        if A is None:
            A = self.value

        E = ncon((A, np.conj(A)), ([-1, 1, -3], [-2, 1, -4]))

        return TransferMatrix(E)
    
    def r_dominant(self):
        """
            Find the dominant right eigenvector 
        """
        _, r = eigs(self.transfer_matrix.value.reshape((self.bond_dim**2, self.bond_dim**2)), k=1, which='LM')
        r = r.reshape((self.bond_dim, self.bond_dim))
        self.r = r / np.trace(r)
    
    def gauge(self, U: np.ndarray = None):

        if U is None:
            U = random_unitary(self.bond_dim)

        # check unitarity
        assert np.allclose(ncon((np.conj(U).T, U), ((-1, 1), (1, -2))), np.eye(self.bond_dim)), 'U should satisfy unitary constraint U^dagger*U=1'

        A = ncon((U, self.value, np.conj(U).T), ((-1, 1), (1, -2, 3), (3, -3)))
        r = ncon((U, self.r, np.conj(U).T), ((-1, 1), (1, 2), (2, -2)))

        gauged = UniformMPS(A)
        gauged.r = r

        return gauged
    
    def mps_chain(self, L: int):
        tensors = [self.value for i in range(L)]
        indices = [[-i, -(i+1), i] if i == 1 else [i-1, -(i+1), -(i+2)] if i == L else [i-1, -(i+1), i] for i in range(1, L+1)]
        return ncon(tensors, indices)
    
    def reduced_density_mat(self, L:int):
        """
            Returns the reduced density matrix in the form:

            e.g    1   3   5   7
                   |   |   |   |
                 - A - A - A - A - 
                |                 R
                 - A*- A*- A*- A*-
                   |   |   |   |
                   2   4   6   8
        """

        mps = self.mps_chain(L)
        return ncon((mps, self.r, np.conj(mps)), ([1, *[i for i in range(-1, -(2*L), -2)], 2], [2, 3], [1, *[i for i in range(-2, -(2*L+1), -2)], 3]))

    def fidelity(self, other:Self):
        E = TransferMatrix(ncon((self.value, other.value), ([-1, 1, -3], [-4, 1, -2])))
        return E.leading_r_eigen()
    

class TransferMatrix():
    
    def __init__(self, value: np.ndarray):
        self.value = value
        self.dims = (value.shape[0], value.shape[1])

    def __matmul__(self, other):
        res = ncon((self.value, other.value), ((-1, -2, 1, 2), (1, 2, -3, -4)))
        return TransferMatrix(res)
    
    def __pow__(self, n: int):

        d1, d2 = self.dims
        if n == 0:
            return TransferMatrix(np.identity(d1*d2, self.value.dtype).reshape(d1, d2, d1, d2))

        order = int(np.log2(n))
        E = TransferMatrix(self.value)
        for _ in range(order):
            E = E @ E

        rmd = n - 2**order
        if rmd > 0:
            E = E @ self ** rmd

        return E
    
    @staticmethod
    def construct_transfer_matrix(*args: np.ndarray):

        if len(args) == 1:
            A, B = args[0], args[0]
        elif len(args) == 2:
            A, B = args[0], args[1]
        else:
            raise TypeError("TransferMatrix.__init__() must provide 1 or 2 positional arguments")

        E = ncon((A, np.conj(B)), ([-1, 1, -3], [-2, 1, -4]))

        return TransferMatrix(E)
    
    def leading_r_eigen(self):
        d1, d2 = self.dims
        r, _ = eigs(self.value.reshape((d1*d2, d1*d2)), k=1, which='LM')
        return r[0]
        

def random_unitary(D: int):
    U = unitary_group.rvs(D)
    assert np.allclose(ncon((np.conj(U).T, U), ((-1, 1), (1, -2))), np.eye(D))
    return U
            
if __name__ == "__main__":

    Da = 5
    p = 2   

    uMPS = UniformMPS.random(Da, p)
    uMPS.r_dominant()
    r = uMPS.r
    E = uMPS.transfer_matrix

    # check normalisation
    assert np.abs(ncon((r, ), ((1, 1))) - 1) < 1e-12

    # check left subspace
    assert np.allclose(ncon((E.value,), ((1, 1, -1, -2))), np.eye(Da))

    # check right subspace
    assert np.allclose(r, np.conj(r).T)

    assert np.allclose(r, ncon((E.value, r), ((-1, -2, 1, 2), (1, 2))))

    # check power function is working
    assert np.allclose((E ** 4).value, ncon((E.value, E.value, E.value, E.value), ((-1, -2, 3, 4), (3, 4, 5, 6), (5, 6, 7, 8), (7, 8, -3, -4))))

    # check gauge transform
    p = uMPS.reduced_density_mat(3)
    Up = uMPS.gauge().reduced_density_mat(3)

    assert np.allclose(p, Up)

    assert np.allclose(
        ncon((p, Up), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))),
        ncon((p, p), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))),
        ncon((Up, Up), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5)))
    )

    assert np.allclose(E.value, ncon(((E ** 0).value, E.value), ((-1, -2, 1, 2), (1, 2, -3, -4))))

    print("All assertions passed!")