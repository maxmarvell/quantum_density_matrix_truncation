from ncon import ncon
import numpy as np
from scipy.linalg import expm, polar
from abc import ABC, abstractmethod

def preconditioning(G: np.ndarray, r: np.ndarray) -> np.ndarray:
    d, _ = r.shape
    delta = np.linalg.norm(G.reshape((-1, d)))**2
    # print(delta)
    Rinv = np.linalg.inv(r + np.eye(d)*delta)
    return G @ Rinv

class AbstractProjector(ABC):
    @abstractmethod
    def project(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        pass

class IdentityProjector(AbstractProjector):
    def project(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        return D

class GrassmanProjector(AbstractProjector):
    def project(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        return D - ncon((A, np.conj(A), D), ((-1, -2, 3), (1, 2, 3), (1, 2, -3)))

class SteifelProjector(AbstractProjector):
    def project(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        d, p, _ = A.shape
        A = A.reshape((d*p, d))
        D = D.reshape((d*p, d))
        res = D - 0.5 * A @ (np.conj(A).T @ D + np.conj(D).T @ A)
        return res.reshape((d, p, d))
    
class AbstractUpdate(ABC):
    def __init__(self, projector: AbstractProjector, preconditioning: bool = True):
        self.projector = projector
        self.preconditioning = preconditioning

    @abstractmethod
    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> np.ndarray:
        pass

class GradientDescent(AbstractUpdate):
    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> np.ndarray:
        G = self.projector.project(A, D)
        if self.preconditioning:
            G = preconditioning(G, r)
        A_new = A - G * alpha
        d, p, _ = A_new.shape
        _A, _ = polar(A_new.reshape(d*p, d))
        return _A.reshape(d, p, d)
    
class Retraction(AbstractUpdate):
    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> np.ndarray:

        d, p, _ = A.shape
        Y = A.reshape(d*p, d)
        # F_Y = D.reshape(d*p, d)
        
        G = self.projector.project(A, D)

        if self.preconditioning:
            G = preconditioning(G, r)

        U, s, Vh = np.linalg.svd(-G.reshape(d*p, d), full_matrices=False)
        V = Vh.conj().T

        t = alpha # Step size
        
        # Geodesic formula from paper (2.65) and (3.1)
        Y_new_mat = (Y @ V) @ np.diag(np.cos(s * t)) @ Vh + U @ np.diag(np.sin(s * t)) @ Vh
        
        return Y_new_mat.reshape(d, p, d)

class TestRetraction(AbstractUpdate):
    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> np.ndarray:

        G = self.projector.project(A, D)
        d, _, _ = A.shape

        if self.preconditioning:
            G = -preconditioning(G, r)

        Zero = np.zeros((d, d))
        I = np.eye(d)

        a = np.block([A, alpha*G])
        GhG = ncon((np.conj(G), G), ((1, 2, -1), (1, 2, -2)))
        b = np.block([
            [Zero, -alpha**2 * GhG],
            [I, Zero]
        ])
        b = expm(b)[..., :d]

        return a @ b