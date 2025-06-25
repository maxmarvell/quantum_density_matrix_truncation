from ncon import ncon
import numpy as np
from scipy.linalg import expm
from abc import ABC, abstractmethod

def preconditioning(G: np.ndarray, r: np.ndarray) -> np.ndarray:
    d, _ = r.shape
    delta = np.linalg.norm(G.reshape((-1, d)))**2
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
        return A - G * alpha
    
class Retraction(AbstractUpdate):
    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> np.ndarray:
        
        G = self.projector.project(A, D)
        if self.preconditioning:
            G = preconditioning(G, r)

        d, p, _ = A.shape
        Zero = np.zeros((d*p, d*p))
        I = np.eye(d*p)

        A = A.reshape(d*p, d)
        G = G.reshape(d*p, d)

        a = np.block([A, alpha*G])
        b = np.block([
            [Zero, -alpha**2 * G.conj().T @ G],
            [I, Zero]
        ])
        b = expm(b)[..., :d*p]

        return a @ b