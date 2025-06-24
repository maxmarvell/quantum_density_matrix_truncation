from uniform_MPS import UniformMPS
from ncon import ncon
import numpy as np
from scipy.linalg import expm
from fixed_point import RightFixedPoint
from abc import ABC, abstractmethod

def preconditioning(G: np.ndarray, r: RightFixedPoint):
    d, _ = r.shape
    delta = np.linalg.norm(G.reshape((-1, d)))**2
    Rinv = np.linalg.inv(r.tensor + np.eye(d)*delta)
    return G @ Rinv

class Projector(ABC):
    @abstractmethod
    def project(self, A: UniformMPS, D: np.ndarray):
        pass

class IdentityProjector(Projector):
    def project(self, A, D):
        return D

class GrassmanProjector(Projector):
    def project(self, A, D):
        return D - ncon((A.tensor, A.conj, D), ((-1, -2, 3), (1, 2, 3), (1, 2, -3)))

class SteifelProjector(Projector):
    def project(self, A, D):
        d, p = A.d, A.p
        A = A.matrix
        D = D.reshape((d*p, d))
        res = D - 0.5 * A @ (np.conj(A).T @ D + np.conj(D).T @ A)
        return res.reshape((d, p, d))

class Retraction(ABC):
    def __init__(self, projector: Projector, preconditioning: bool = True):
        self.projector = projector
        self.preconditioning = preconditioning

    @abstractmethod
    def retract(self):
        pass

class RiemannianGradientDescent(Retraction):
    def retract(self, A: UniformMPS, D: np.ndarray, alpha: float, r: RightFixedPoint = None):
        G = self.projector.project(A, D)
        if self.preconditioning:
            G = preconditioning(G, r)
        return A.tensor - G * alpha 


# def grassman_retraction(W, G, alpha):
#     '''
#         Peform a geodesic retraction on the Grassmann manifold based on the Euclidean metric.
#     '''
#     n, m = W.shape
#     Q, R = np.linalg.qr((np.eye(n) - W @ W.conj().T) @ G, mode='reduced')
#     Zero = np.zeros((m, m))

#     a = np.block([W, Q])
#     b = np.block([
#         [Zero, -alpha*R.conj().T],
#         [alpha*R, Zero]
#     ])
#     b = expm(b)[..., :m]
#     return a @ b