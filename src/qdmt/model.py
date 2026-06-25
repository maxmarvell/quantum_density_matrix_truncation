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
    


    def trotter_zeroth_order(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Zeroth-order (single-layer) with built-in rescaling:
        return (I, U(2*dt)) where U is the same two-site gate used for U_half_dt.

        This keeps your existing evolution code unchanged: it will apply layer-1 with U1 (identity)
        and layer-2 shifted with U2 (the rescaled gate).
        """
        if self._delta_t is None:
            raise AttributeError("The step size delta_t has not been set!")

        # Make sure we have a reference gate to infer shape/dtype.
        if self.U_half_dt is None:
            self._compute_U_half_dt()
        U_ref = self.U_half_dt

        # Build identity in the same representation as U_ref.
        if U_ref.ndim == 2:
            I = np.eye(U_ref.shape[0], dtype=U_ref.dtype)
        elif U_ref.ndim == 4:
            d = U_ref.shape[0]
            I = np.eye(d * d, dtype=U_ref.dtype).reshape(d, d, d, d)
        else:
            raise ValueError(f"Unexpected U_half_dt ndim={U_ref.ndim}; expected 2 or 4.")

        # Rescaled gate at 2*dt, using the existing delta_t-driven machinery.
        dt0 = self._delta_t
        self.delta_t = 2.0 * dt0          # uses setter; recomputes cached gates if present
        U_rescaled = self._compute_U_half_dt()
        self.delta_t = dt0                # restore; setter recomputes back if caches exist

        return I, U_rescaled



    def trotter_zeroth_order_average(self) -> tuple[np.ndarray, np.ndarray]:
        """
       this is for the average
        """
        if self._delta_t is None:
            raise AttributeError("The step size delta_t has not been set!")

        # Make sure we have a reference gate to infer shape/dtype.
        if self.U_half_dt is None:
            self._compute_U_half_dt()
        U_ref = self.U_half_dt

        # Build identity in the same representation as U_ref.
        if U_ref.ndim == 2:
            I = np.eye(U_ref.shape[0], dtype=U_ref.dtype)
        elif U_ref.ndim == 4:
            d = U_ref.shape[0]
            I = np.eye(d * d, dtype=U_ref.dtype).reshape(d, d, d, d)
        else:
            raise ValueError(f"Unexpected U_half_dt ndim={U_ref.ndim}; expected 2 or 4.")

        # Rescaled gate at 2*dt, using the existing delta_t-driven machinery.
        dt0 = self._delta_t
        self.delta_t = 2.0 * dt0          # uses setter; recomputes cached gates if present
        U_rescaled = self._compute_U_half_dt()
        self.delta_t = dt0                # restore; setter recomputes back if caches exist

        return U_rescaled, U_rescaled

    @abstractmethod
    def _compute_U_half_dt(self) -> None:
        pass

    @abstractmethod
    def _compute_U_quarter_dt(self) -> None:
        pass

class TransverseFieldIsing(AbstractModel):

    ZZ = np.kron(Pauli.Sz, Pauli.Sz) 
    XI = np.kron(Pauli.Sx, Pauli.I)
    ZI = np.kron(Pauli.Sz, Pauli.I)

    def __init__(self, g: float, delta_t: float | None = None, h: float = 0, J: float = 1.) -> None:
        super().__init__(delta_t)
        self.g = g
        self.J = J
        self.h = h
        self.H = (self.J*self.ZZ+self.g*self.XI+self.h*self.ZI).reshape(2, 2, 2, 2)

    def _compute_U_half_dt(self) -> np.ndarray:
        H = self.H.reshape(4, 4)
        self.U_half_dt = expm(-1j*self.delta_t*H).reshape(2, 2, 2, 2)
        return self.U_half_dt
    
    def _compute_U_quarter_dt(self) -> np.ndarray:
        H = self.H.reshape(4, 4)
        self.U_quarter_dt = expm(-.5j*self.delta_t*H).reshape(2, 2, 2, 2)
        return self.U_quarter_dt

class TransverseFieldIsingSym(AbstractModel):

    ZZ = np.kron(Pauli.Sz, Pauli.Sz) 
    XI = np.kron(Pauli.Sx, Pauli.I)
    IX = np.kron(Pauli.I, Pauli.Sx)
    ZI = np.kron(Pauli.Sz, Pauli.I)

    def __init__(self, g: float, delta_t: float | None = None, h: float = 0, J: float = 1.) -> None:
        super().__init__(delta_t)
        self.g = g
        self.J = J
        self.h = h
        self.H = (self.J*self.ZZ+0.5*self.g*self.XI+0.5*self.g*self.IX+self.h*self.ZI).reshape(2, 2, 2, 2)


    
    def _compute_U_quarter_dt(self) -> np.ndarray:
        H = self.H.reshape(4, 4)
        self.U_quarter_dt = expm(-.5j*self.delta_t*H).reshape(2, 2, 2, 2)
        return self.U_quarter_dt
    

    def _compute_U_half_dt(self) -> np.ndarray:
        """
        Match your QASM gate order:

            CX-RZ(2*dt*J)-CX   (implements exp(-i dt * J * ZZ))
            RX(dt*g) on q0     (implements exp(-i dt * (g/2) * XI))
            RX(dt*g) on q1     (implements exp(-i dt * (g/2) * IX))

        QASM applies ZZ first, then RX(q0), then RX(q1),
        so the matrix product is U = Uix @ Uxi @ Uzz.
        """
        dt = float(self.delta_t)

        Uzz = expm(-1j * dt * (self.J * self.ZZ))
        Uxi = expm(-1j * dt * (0.5 * self.g * self.XI))
        Uix = expm(-1j * dt * (0.5 * self.g * self.IX))

        # if self.h != 0.0:
        #     # If you *also* implement an on-site Z field in QASM, include it here similarly.
        #     # NOTE: as written, your QASM snippet did not include Z-field rotations.
        #     Uzi = expm(-1j * dt * (self.h * self.ZI))
        #     # Decide where it sits in your circuit order; put it last by default:
        #     return (Uzi @ Uix @ Uxi @ Uzz).reshape(2, 2, 2, 2)
        self.U_half_dt =(Uix @ Uxi @ Uzz).reshape(2, 2, 2, 2)
        return self.U_half_dt 
        


class HeisenbergXXZ(AbstractModel):

    ZZ = np.kron(Pauli.Sz, Pauli.Sz) 
    XX = np.kron(Pauli.Sx, Pauli.Sx) 
    YY = np.kron(Pauli.Sy, Pauli.Sy) 
    ZI = np.kron(Pauli.Sz, Pauli.I)

    def __init__(self, h:float, Delta: float, delta_t: float):
        super().__init__(delta_t)
        self.h = h
        self.Delta = Delta
        self.H = (self.Delta*self.ZZ+self.XX+self.YY).reshape(2, 2, 2, 2)

    def _compute_U_half_dt(self) -> None:
        H = self.H.reshape(4, 4)
        self.U_half_dt = expm(-1j*self.delta_t*H).reshape(2, 2, 2, 2)
    
    def _compute_U_quarter_dt(self) -> None:
        H = self.H.reshape(4, 4)
        self.U_quarter_dt = expm(-.5j*self.delta_t*H).reshape(2, 2, 2, 2)