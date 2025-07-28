from abc import ABC, abstractmethod
from ncon import ncon
import numpy as np

class AbstractManifold(ABC):
    @abstractmethod
    def project(A: np.ndarray, D: np.ndarray) -> np.ndarray:
        pass

def euclidian_metric(X: np.ndarray, Y: np.ndarray):
    return np.real(np.trace(X.conj().T @ Y))

class Grassmann(AbstractManifold):
    def project(self, W: np.ndarray, D: np.ndarray) -> np.ndarray:
        if W.ndim == 2 and D.ndim == 2:
            return D - W @ W.conj().T @ D
        elif W.ndim == 3 and D.ndim == 3:
            return D - ncon((W, np.conj(W), D), ((-1, -2, 3), (1, 2, 3), (1, 2, -3)))
        else:
            raise ValueError("Error: Expected W and D to be either rank 3 (uMPS) or rank 2 (isometry) tensors!")
        

class Steifel(AbstractManifold):
    def project(self, W: np.ndarray, D: np.ndarray) -> np.ndarray:
        if W.ndim == 2 and D.ndim == 2:
            return D - 0.5 * W @ (np.conj(W).T @ D + np.conj(D).T @ W)
        elif W.ndim == 3 and D.ndim == 3:
            raise NotImplementedError("Error: Not implemented Steifel projection on uMPS shape `(D, d, D)`")
        else:
            raise ValueError("Error: Expected W and D to be either rank 3 (uMPS) or rank 2 (isometry) tensors!")
