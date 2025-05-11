from ncon import ncon
from scipy.sparse.linalg import eigs, LinearOperator, gmres
from functools import partial
from transfer_matrix import *

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

    def __init__(self, E: TransferMatrix):
        self.shape = (E.d1, E.d2)
        self.E = E

        _, r = eigs(E.to_matrix(), k=1, which='LM')
        r = r.reshape((E.d1, E.d2))
        r /= (np.trace(r) / np.abs(np.trace(r)))
        r = (r + np.conj(r).T) / 2
        r *= np.sign(np.trace(r))

        self.tensor = r / np.trace(r*(1 / np.sqrt(E.d1)))

    def derivative(self, v):
        _, nb = self.shape
        L = LinearOperator((nb ** 2, nb ** 2), matvec=partial(EtildeLeft, self.E, self.tensor))
        Rh = gmres(L, v.reshape(nb ** 2))[0]
        Rh = Rh.reshape(nb, nb)
        return ncon((Rh, self.E.B.tensor, self.tensor), ((-1, 1), (1, -2, 2), (2, -3)))

        
if __name__ == "__main__":
    pass