from ncon import ncon
import numpy as np
from scipy.linalg import polar
from scipy.sparse.linalg import eigs
from typing import Self

class UniformMps():

    def __init__(self, A: np.ndarray):

        if A.ndim == 2:
            self.d = A.shape[1]

            if A.shape[0] % A.shape[1] != 0:
                raise ValueError("Error: To define a uniform matrix product state from an isometry isometry must have shape like `(d*D, D)`")
            
            self.p = A.shape[0] // A.shape[1]
            self.tensor = A.reshape(self.d, self.p, self.d)

        elif A.ndim == 3:
            self.d, self.p, _ = A.shape
            self.tensor = A

    @classmethod
    def new(cls, d: int, p: int, left_canonical: bool = True):
        """
            Initialise a random normalised uniform MPS states in left canonical form.

            args:
                d: Bond dimension of the uniform matrix product state
                p: Physical dimension of the uniform matrix product state

            returns:
                Random normalised uniform matrix product states, shape (d, p, d)
        """

        # generate random A not necessarily left canonical
        A = np.random.normal(size=(d*p, d)) + 1j * np.random.normal(size=(d*p, d))

        if left_canonical:
            A, _ = polar(A)
            A = A.reshape(d, p, d)
            return cls(A)
        
        A = A.reshape(d, p, d)

        # random A is not necessarily normalized
        if not left_canonical:
            E = ncon((A, A.conj()), ((-1, 1, -3), (-2, 1, -4)))
            norm = eigs(np.reshape(E, (d ** 2, d ** 2)), k=1, which='LM', return_eigenvectors=False)
            A = A / np.sqrt(norm)

        return cls(A)

    @property
    def matrix(self):
        return self.tensor.reshape(self.d * self.p, self.d)
    
    @property
    def isometry(self):
        return self.tensor.reshape(self.d * self.p, self.d)
    
    @property 
    def conj(self):
        return self.tensor.conj()
    
    def mps_chain(self, L: int) -> np.ndarray:
        if L == 0:
            return np.eye(self.d)
        if L == 1:
            return self.tensor
        tensors = [self.tensor for _ in range(L)]
        indices = [[-i, -(i+1), i] if i == 1 else [i-1, -(i+1), -(i+2)] if i == L else [i-1, -(i+1), i] for i in range(1, L+1)]
        return ncon(tensors, indices)

    def fidelity(self, other: Self):
        d1, d2 = self.d, other.d
        E = ncon((self.tensor, other.conj), ((-1, 1, -3), (-2, 1, -4)))
        r = eigs(E.reshape(d1*d2, d1*d2), k=1, which='LM', return_eigenvectors=False)
        return np.abs(r[0])
    
    def correlation_length(self):
        d = self.d
        E = ncon((self.tensor, self.conj), ((-1, 1, -3), (-2, 1, -4)))
        r = eigs(E.reshape(d*d, d*d), k=2, which='LM', return_eigenvectors=False)
        r_sorted = np.sort(np.abs(r))[::-1]
        return -1/np.log(np.abs(r_sorted[1])/np.abs(r_sorted[0]))
    
    def normalization(self):
        d = self.d
        E = ncon((self.tensor, self.conj), ((-1, 1, -3), (-2, 1, -4)))
        r = eigs(E.reshape(d*d, d*d), k=1, which='LM', return_eigenvectors=False)
        return np.abs(r[0])
    
    def check_left_orthonormal(self):
        I = ncon((self.tensor, self.conj), ((2, 1, -2), (2, 1, -1)))
        return np.allclose(I, np.eye(self.d))

    def left_orthorganlize(self) -> None:
        A, _ = polar(self.matrix)
        self.tensor = A.reshape(self.d, self.p, self.d)
            
if __name__ == "__main__":
    theta = phi = np.pi / 2

    D = 6
    psi = np.array([np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)])

    A = np.zeros((D, 2, D), dtype=np.complex128)
    A[0, 0, 0] = np.cos(theta / 2)
    A[0, 1, 0] = np.exp(phi * 1j) * np.sin(theta / 2)

    print(psi)

    mps = UniformMps(A)

    print(mps.check_left_orthonormal())