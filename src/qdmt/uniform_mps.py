from ncon import ncon
import numpy as np
from scipy.linalg import polar
from scipy.sparse.linalg import eigs
from scipy.stats import unitary_group
from typing import Self

class UniformMps():

    def __init__(self, A: np.ndarray):
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
    
    def normalization(self):
        d = self.d
        E = ncon((self.tensor, self.conj), ((-1, 1, -3), (-2, 1, -4)))
        r = eigs(E.reshape(d*d, d*d), k=1, which='LM', return_eigenvectors=False)
        return np.abs(r[0])
            
if __name__ == "__main__":
    pass