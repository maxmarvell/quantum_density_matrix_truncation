from dataclasses import dataclass
import numpy as np

@dataclass
class Isometry:
    V: np.ndarray
    tol: float = 1e-10

    def __post_init__(self):
        if self.V.ndim != 2:
            raise ValueError("V must be a 2D array.")
        
    @property
    def m(self) -> int: return self.V.shape[0]
    @property
    def n(self) -> int: return self.V.shape[1]
    @property
    def dtype(self): return self.V.dtype

    @classmethod
    def random(cls, m: int, n: int, *, seed: int | None = None, tol: float = 1e-10) -> "Isometry":
        rng = np.random.default_rng(seed)
        M = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))
        Q, _ = np.linalg.qr(M)
        return cls(Q.astype(M.dtype), tol=tol)
    
    @classmethod
    def from_matrix(cls, M: np.ndarray, *, project: bool = False, tol: float = 1e-10) -> "Isometry":
        iso = cls(M, tol=tol)
        if project and not iso.is_isometry(tol=tol):
            iso = iso.project_qr()
        return iso
    
    def is_isometry(self, tol: float | None = None) -> bool:
        t = self.tol if tol is None else tol
        I = self.V.conj().T @ self.V
        diff = I - np.eye(self.n, dtype=self.V.dtype)
        dist = np.linalg.norm(diff)   # Frobenius norm by default
        # print(f"‖V†V − I‖ = {dist:.3e}")
        return np.allclose(I, np.eye(self.n, dtype=self.V.dtype), atol=t, rtol=0)
    
    def adjoint(self) -> "Isometry":
        return Isometry(self.V.conj().T, tol=self.tol)
    
    def conj(self) -> "Isometry":
        return Isometry(self.V.conj(), tol=self.tol)
    
    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        return self.V @ other
    
    def project_qr(self) -> "Isometry":
        Q, _ = np.linalg.qr(self.V)
        return Isometry(Q.astype(self.V.dtype), tol=self.tol)
    
    def orthonormalize(self, method: str = "qr") -> "Isometry":
        if method == "qr":
            return self.project_qr()
        else:
            raise ValueError("method must be 'qr'")