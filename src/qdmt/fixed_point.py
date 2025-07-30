from ncon import ncon
from scipy.sparse.linalg import eigs, LinearOperator, gmres
from functools import partial
from typing import Self
import numpy as np

from qdmt.transfer_matrix import TransferMatrix
from qdmt.uniform_mps import UniformMps

def EtildeLeft(E: TransferMatrix, r, v):

    d = E.d1
    v = v.reshape(d, d)

    # transfermatrix contribution
    transfer = ncon((v, E.tensor), ((1, 2), (2, 1, -2, -1)))

    # fixed point contribution
    fixed = np.trace(v @ r) * np.eye(d)

    # sum these with the contribution of the identity
    return v - transfer + fixed

class RightFixedPoint():

    E: TransferMatrix
    tensor: np.ndarray

    def __init__(self, E: TransferMatrix):
        self.E = E
        _, r = eigs(E.to_matrix(), k=1, which='LM')
        r = r.reshape((E.d1, E.d2))
        r /= (np.trace(r) / np.abs(np.trace(r)))
        r = (r + np.conj(r).T) / 2
        r *= np.sign(np.trace(r))

        ## assumes left canonical
        self.tensor = r / np.trace(r)

    @classmethod
    def from_mps(cls, u_mps: UniformMps):
        E = TransferMatrix.new(u_mps, u_mps)
        return cls(E)

    def derivative(self, v) -> np.ndarray:
        _, nb = self.tensor.shape
        L = LinearOperator((nb ** 2, nb ** 2), matvec=partial(self._pseudoinverse_operator))
        Rh = gmres(L, v.reshape(nb ** 2))[0]
        Rh = Rh.reshape(nb, nb)
        return ncon((Rh, self.E.B.tensor, self.tensor), ((-1, 1), (1, -2, 2), (2, -3)))
    
    def _pseudoinverse_operator(self: Self, v: np.ndarray):
        
        d, _ = self.tensor.shape
        v = v.reshape(d, d)

        # transfermatrix contribution
        transfer = ncon((v, self.E.tensor), ((1, 2), (2, 1, -2, -1)))

        # fixed point contribution
        fixed = np.trace(v @ self.tensor) * np.eye(d)

        # sum these with the contribution of the identity
        return v - transfer + fixed

class LeftFixedPoint():

    def __init__(self, E: TransferMatrix):
        self.shape = (E.d1, E.d2)
        self.E = E

        _, l = eigs(E.to_matrix().T, k=1, which='LM')
        l = l.reshape((E.d1, E.d2))
        l /= (np.trace(l) / np.abs(np.trace(l)))
        l = (l + l.conj().T) / 2
        l *= np.sign(np.trace(l))

        ## assume left canonical form so should canonicalize to 
        self.tensor = l / lA.tensor[0, 0]

    def derivative(self, v) -> np.ndarray:
        _, nb = self.shape
        L = LinearOperator((nb ** 2, nb ** 2), matvec=partial(EtildeLeft, self.E, self.tensor))
        Rh = gmres(L, v.reshape(nb ** 2))[0]
        Rh = Rh.reshape(nb, nb)
        return ncon((Rh, self.E.B.tensor, self.tensor), ((-1, 1), (1, -2, 2), (2, -3)))

        
if __name__ == "__main__":
    pass