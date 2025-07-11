from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import expm
from ncon import ncon

class Pauli():
    Sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
    Sy = np.array([[0, 1j],
                [-1j, 0]], dtype=complex)
    Sz = np.array([[1, 0],
                [0, -1]], dtype=complex)
    
    I = np.eye(2, dtype=complex)


class AbstractModel(ABC):

    H: np.ndarray
    U_half_dt: np.ndarray
    U_quarter_dt: np.ndarray

    def __init__(self, delta_t: float | None = None):
        self.H = None
        self.U_half_dt = None
        self.U_quarter_dt = None
        self._delta_t = delta_t

    @property
    def delta_t(self):
        return self._delta_t
    
    @delta_t.setter
    def delta_t(self, value: float):
        self._delta_t = value

        if self.U_half_dt is not None and self._delta_t is not None:
            self._compute_U_half_dt()

        if self.U_quarter_dt is not None and self._delta_t is not None:
            self._compute_U_quarter_dt()


    def trotter_first_order(self) -> tuple[np.ndarray, np.ndarray]:
        if self._delta_t is None:
            raise AttributeError("The step size delta_t has not been set!")
        if self.U_half_dt is None:
            self._compute_U_half_dt()
        return self.U_half_dt, self.U_half_dt
        

    def trotter_second_order(self) -> tuple[np.ndarray, np.ndarray]:
        if self._delta_t is None:
            raise AttributeError("The step size delta_t has not been set!")
        if self.U_half_dt is None:
            self._compute_U_half_dt()
        if self.U_quarter_dt is None:
            self._compute_U_quarter_dt()
        return self.U_quarter_dt, self.U_half_dt

    @abstractmethod
    def _compute_U_half_dt(self) -> None:
        pass

    @abstractmethod
    def _compute_U_quarter_dt(self) -> None:
        pass

class TransverseFieldIsing(AbstractModel):

    ZZ = np.kron(Pauli.Sz, Pauli.Sz) 
    XI = np.kron(Pauli.Sx, Pauli.I)

    def __init__(self, g: float, delta_t: float | None = None) -> None:
        super().__init__(delta_t)
        self.g = g
        self.H = (self.ZZ+self.g*self.XI).reshape(2, 2, 2, 2)

    def _compute_U_half_dt(self) -> None:
        H = self.H.reshape(4, 4)
        self.U_half_dt = expm(-1j*self.delta_t*H).reshape(2, 2, 2, 2)
    
    def _compute_U_quarter_dt(self) -> None:
        H = self.H.reshape(4, 4)
        self.U_quarter_dt = expm(-.5j*self.delta_t*H).reshape(2, 2, 2, 2)
        
    