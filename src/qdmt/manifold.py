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
    def project(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:

        if A.ndim == 2 and D.ndim == 2:
            return D - A @ A.conj().T @ D
        
        else:
            return D - ncon((A, np.conj(A), D), ((-1, -2, 3), (1, 2, 3), (1, 2, -3)))
