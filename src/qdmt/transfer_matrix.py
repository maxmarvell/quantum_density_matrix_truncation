from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import LinAlgError
from ncon import ncon
from typing import Optional, Sequence
from scipy.sparse.linalg import LinearOperator, eigs, gmres
from functools import partial
import warnings

from qdmt.uniform_mps import UniformMps

class LeftFixedPoint():
    def __init__(self, tensor: np.ndarray, A: np.ndarray):
        self.tensor, self.A = tensor, A
        self.D = tensor.shape[0]


class RightFixedPoint():
    def __init__(self, tensor: np.ndarray, A: np.ndarray):
        self.tensor, self.A = tensor, A
        self.D = tensor.shape[0]

    def derivative(self, v: np.ndarray) -> np.ndarray:
        L = LinearOperator((self.D ** 2, self.D ** 2), matvec=partial(self._pseudoinverse_operator))
        Rh: np.ndarray = gmres(L, v.reshape(self.D ** 2))[0]
        Rh = Rh.reshape(self.D, self.D)
        return ncon((Rh, self.A, self.tensor), ((-1, 1), (1, -2, 2), (2, -3)))
    
    def _pseudoinverse_operator(self, v: np.ndarray):
        # reshape as a matrix
        v = v.reshape(self.D, self.D)

        # get the transfermatrix contribution
        transfer = ncon([v, self.A, self.A.conj()], [[1, 2], [2, 3, -2], [1, 3, -1]])

        # fixed point contribution
        fixed = np.trace(v @ self.tensor) * np.eye(self.D)

        # sum these with the contribution of the identity
        return v - transfer + fixed

class AbstractTransferMatrix(ABC):

    _derivative: Optional[np.ndarray]

    def __init__(self, tensor: np.ndarray, *, tape: Sequence["AbstractTransferMatrix"] = None):
        if not isinstance(tensor, np.ndarray):
            raise TypeError("E must be a numpy array.")
        if len(tensor.shape) % 2 != 0:
            raise ValueError("Balanced number of input and output legs required.")
        
        n = len(tensor.shape) // 2
        if tensor.shape[:n] != tensor.shape[-n:]:
            raise ValueError("Dimensions of input and output legs should be the same.")
        
        self._tensor = tensor
        self._n = n
        self._derivative = None
        self.tape = list(tape) if tape is not None else []

    @property
    def tensor(self) -> np.ndarray:
        return self._tensor
    
    @property
    def n(self) -> int:
        return self._n
    
    @abstractmethod
    def identity_like(self) -> "AbstractTransferMatrix":
        """blah"""

    @abstractmethod
    def derivative(self) -> np.ndarray:
        """Return d/dθ tensor (same shape as `tensor`)."""

    @abstractmethod
    def __matmul__(self, other) -> "AbstractTransferMatrix":
        pass

    @abstractmethod
    def _matvec(self, other) -> "AbstractTransferMatrix":
        pass

    @abstractmethod
    def _rmatvec(self, other) -> "AbstractTransferMatrix":
        pass

    def to_matrix(self) -> np.ndarray:
        N = int(np.prod(self.tensor.shape[:self.n]))
        return self.tensor.reshape((N, N))

    def linop(self) -> LinearOperator:
        return LinearOperator((self._n, self._n), matvec=self._matvec, dtype=self.tensor.dtype)
    
    def __pow__(self, n: int) -> "AbstractTransferMatrix":

        if n == 0:
            return self.identity_like()

        result = None
        power = self
        while n > 0:
            if n % 2 == 1:
                if result is None:
                    result = power
                else:
                    result = result @ power
            if n == 1:
                return result
            power = power @ power
            n //= 2
        return result


class TransferMatrix(AbstractTransferMatrix):

    def __init__(self, tensor: np.ndarray, *, tape: Sequence["TransferMatrix"] = None):
        super().__init__(tensor, tape=tape)

        # dimensions
        self.Da, self.Db = tensor.shape[:self.n]

        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

    @property
    def A(self) -> Optional[np.ndarray]:
        return self._A

    @property
    def B(self) -> Optional[np.ndarray]:
        return self._B
    
    @classmethod
    def new(cls, A: UniformMps, B: UniformMps):

        E = ncon((A.tensor, B.tensor.conj()), ((-1, 1, -3), (-2, 1, -4)))
        obj = cls(E)
        obj._A, obj._B = A.tensor, B.tensor
        return obj
    
    def _compute_derivative(self) -> np.ndarray:
        if self.A is None or self.B is None:
            raise ValueError("Can only call _compute_derivative for non-composite transfer matrix object.")
        Ib = np.eye(self.Db)
        return ncon([self.A, Ib, Ib], ((-1, -6, -3), (-2, -5), (-7, -4)))
    
    def _matvec(self, v: np.ndarray):
        if self.A != None and self.B != None:
            return ncon([self.A, self.B.conj(), v], [[-1, 3, 1], [-2, 3, 2], [1, 2]])
        return ncon([self.tensor, v], [[-1, -2, 1, 2], [1, 2]])
    
    def _rmatvec(self, v):
        if self.A != None and self.B != None:
            return ncon([v, self.A, self.B.conj()], [[1, 2], [-1, 3, 2], [-2, 3, 1]])
        return ncon([self.tensor, v], [[-1, -2, 1, 2], [1, 2]])

    def __matmul__(self, other):

        if isinstance(other, TransferMatrix):
            tensor = ncon((self.tensor, other.tensor), ((-1, -2, 1, 2), (1, 2, -3, -4)))
            return TransferMatrix(tensor, tape=[self, other])
        
        if isinstance(other, np.ndarray):
            if other.ndim == 2:
                return self._matvec(other)
            else:
                raise ValueError("Unsupported operand shape for multiplication.")
            
        raise NotImplementedError()

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray) and other.ndim == 2:
            return self._rmatvec(other)
        raise NotImplementedError()
    
    def derivative(self):

        if self._derivative is not None:
            return self._derivative
        
        # if no parent tensors simply compute derivative and return
        elif len(self.tape) == 0:
            self._derivative = self._compute_derivative()
            return self._derivative
        
        elif len(self.tape) == 2:
            L = self.tape[0]
            R = self.tape[1]

            indices = [[-1, -2, 1, 2, -5, -6, -7], [1, 2, -3, -4]]
            LRdL = ncon([L.derivative(), R.tensor], indices)

            indices = [[-1, -2, 1, 2], [1, 2, -3, -4, -5, -6, -7]]
            LRdR = ncon([L.tensor, R.derivative()], indices)

            self._derivative = LRdL + LRdR
            return self._derivative
        
        raise NotImplementedError
    
    def identity_like(self):
        N = int(np.prod(self.tensor.shape[:self.n]))
        I = np.eye(N).reshape(*self.tensor.shape)
        return TransferMatrix(I)
    
    def right_fixed_point(self):
        if self.A is None or self.B is None:
            raise ValueError("Need A and B to define fixed point.")
        
        if self.A is not self.B:
            warnings.warn("A and B differ: right_fixed_point is not guaranteed.")

        M = self.to_matrix()
        eig, r = eigs(M, k=1, which='LM')

        if np.allclose(eig, [1.]):
            r = r.reshape((self.Da, self.Db))
            r /= (np.trace(r) / np.abs(np.trace(r)))
            r = (r + np.conj(r).T) / 2
            r *= np.sign(np.trace(r))
                    # --- NEW: normalize trace to 1 ---
            tr = np.trace(r).real
            r /= tr
            return RightFixedPoint(r, self.A)
        

    def left_fixed_point(self, atol=1e-10, rtol=1e-10):
        if self.A is None or self.B is None:
            raise ValueError("Need A and B to define fixed point.")
        if self.A is not self.B:
            warnings.warn("A and B differ: left_fixed_point is not guaranteed.")

        M = self.to_matrix()
        eig, l = eigs(M.T, k=1, which="LM")   # left eigenvector of M

        if np.isclose(eig[0], 1.0, atol=atol, rtol=rtol):
            l = l.reshape((self.Da, self.Db))
            return LeftFixedPoint(l, self.A)

        raise LinAlgError("Transfer matrix has no eigenvalue 1: left fixed point did not converge.")
    
        


        raise LinAlgError("Transfer matrix has no eigenvalue 1: right fixed point did not converge.")


class FirstOrderTrotterizedTransferMatrix(AbstractTransferMatrix):

    def __init__(self, array: np.ndarray, *, tape: Sequence["FirstOrderTrotterizedTransferMatrix"] = None):
        super().__init__(array, tape=tape)
        self.Da, self.d, self.Db = array.shape[:self.n]
        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._U1: Optional[np.ndarray] = None
        self._U2: Optional[np.ndarray] = None

    @property
    def A(self) -> Optional[np.ndarray]:
        return self._A

    @property
    def B(self) -> Optional[np.ndarray]:
        return self._B
    
    @property
    def U1(self) -> Optional[np.ndarray]:
        return self._U1

    @property
    def U2(self) -> Optional[np.ndarray]:
        return self._U2

    @classmethod
    def new(cls, A: UniformMps, B: UniformMps, U1: np.ndarray, U2: np.ndarray):

        tensors = [B.tensor.conj(), B.tensor.conj(), U2, U1, A.tensor, A.tensor]
        indices = [(-3, 1, 2), (2, 3, -6), (-2, 4, 1, 3), (5, 6, 4, -5), (-1, 5, 7), (7, 6, -4)]
        E = ncon(tensors, indices)

        # create object
        obj = FirstOrderTrotterizedTransferMatrix(E)
        obj._A, obj._B, obj._U1, obj._U2 = A.tensor, B.tensor, U1, U2
        return obj
    
    def __matmul__(self, other):
        if isinstance(other, FirstOrderTrotterizedTransferMatrix):
            indices = [[-1, -2, -3, 1, 2, 3], [1, 2, 3, -4, -5, -6]]
            tensor = ncon([self.array, other.array], indices, order=[1, 3, 2])
            return FirstOrderTrotterizedTransferMatrix(tensor, [self, other])

        raise NotImplementedError()
    
    def identity_like(self) -> "FirstOrderTrotterizedTransferMatrix":
        N = int(np.prod(self.tensor.shape[:self.n]))
        I = np.eye(N).reshape(*self.tensor.shape)
        return FirstOrderTrotterizedTransferMatrix(I)
    
    def derivative(self):

        if self._derivative is not None:
            return self._derivative

        if len(self.tape) == 0:

            self._derivative = self._compute_derivative()

            return self._derivative
        
        if len(self.tape) == 2:

            L = self.tape[0]
            R = self.tape[1]

            tensors = [L.derivative(), R.array]
            indices = [[-1, -2, -3, 1, 2, 3, -7, -8, -9], [1, 2, 3, -4, -5, -6]]
            LRdL = ncon(tensors, indices)

            tensors = [L.array, R.derivative()]
            indices = [[-1, -2, -3, 1, 2, 3], [1, 2, 3, -4, -5, -6, -7, -8, -9]]
            LRdR = ncon(tensors, indices)

            self.D = LRdL + LRdR

            return self.D
        
        raise NotImplementedError()

    def _compute_derivative(self) -> np.ndarray:

        if self.A == None or self.B == None or self.U1 == None or self.U2 == None:
            raise ValueError("Can only call _compute_derivative for non-composite transfer matrix object.")
        
        Ib = np.eye(self.Db)
        tensors = [Ib, self.B.conj(), self.U2, self.U1, self.A, self.A]
        indices = [[-3, -7], [-9, 1, -6], [-2, 2, -8, 1], [3, 4, 2, -5], [-1, 3, 5], [5, 4, -4]]
        D1 = ncon(tensors, indices)

        tensors = [self.B.conj(), Ib, self.U2, self.U1, self.A, self.A]
        indices = [[-3, 1, -7], [-9, -6], [-2, 2, 1, -8], [3, 4, 2, -5], [-1, 3, 5], [5, 4, -4]]
        D2 = ncon(tensors, indices)

        self.D = D1 + D2

        return self.D
            
class SecondOrderTrotterizedTransferMatrix(AbstractTransferMatrix):

    _A: Optional[UniformMps]
    _B: Optional[UniformMps]
    _U1: Optional[np.ndarray]
    _U2: Optional[np.ndarray]

    def __init__(self, array, *, tape: Sequence["SecondOrderTrotterizedTransferMatrix"] = None):
        super().__init__(array, tape=tape)
        self.Da, self.d, _, self.Db = array.shape[:self.n]
        self._A, self._B, self._U1, self._U2 = None, None, None, None

    @property
    def A(self) -> Optional[UniformMps]:
        return self._A

    @property
    def B(self) -> Optional[UniformMps]:
        return self._B
    
    @property
    def U1(self) -> Optional[np.ndarray]:
        return self._U1

    @property
    def U2(self) -> Optional[np.ndarray]:
        return self._U2

    @classmethod
    def new(cls, A: UniformMps, B: UniformMps, U1: np.ndarray, U2: np.ndarray):

        tensors = [B.tensor.conj(), B.tensor.conj(), U1, U2, U1, A.tensor, A.tensor]
        indices = [[-4, 1, 2], [2, 3, -8], [-3, 4, 1, 3], [-2, 5, 4, -7], [6, 7, 5, -6], [-1, 6, 8], [8, 7, -5]]
        E = ncon(tensors, indices)

        obj = SecondOrderTrotterizedTransferMatrix(E)
        obj._A, obj._B, obj._U1, obj._U2 = A, B, U1, U2
        return obj
    
    def identity_like(self) -> "SecondOrderTrotterizedTransferMatrix":
        N = int(np.prod(self.tensor.shape[:self.n]))
        I = np.eye(N).reshape(*self.tensor.shape)
        return SecondOrderTrotterizedTransferMatrix(I)

    def __matmul__(self, other):
        if isinstance(other, SecondOrderTrotterizedTransferMatrix):
            indices = [[-1, -2, -3, -4, 1, 2, 3, 4], [1, 2, 3, 4, -5, -6, -7, -8]]
            tensor = ncon([self._tensor, other._tensor], indices, order=[1, 4, 2, 3])
            return SecondOrderTrotterizedTransferMatrix(tensor, tape=[self, other])
            
        raise NotImplementedError()
    
    def _matvec(self, v: np.ndarray) -> np.ndarray:
        AA = np.einsum('fgh,aef->ghae', self.A.tensor, self.A.tensor)
        BB = np.einsum('onp,dmo->npdm', self.B.tensor.conj(), self.B.tensor.conj())
        AAU1 = np.einsum('ghae,egij->haij', AA, self.U1)
        BBU1 = np.einsum('npdm,ckmn->pdck', BB, self.U1)

        return 
    
    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        AA = np.einsum('fgh,aef->ghae', self.A.tensor, self.A.tensor)
        AAU1 = np.einsum('ghae,egij->haij', AA, self.U1)
        BB = np.einsum('onp,dmo->npdm', self.B.tensor.conj(), self.B.tensor.conj())
        BBU1 = np.einsum('npdm,ckmn->pdck', BB, self.U1)
        vAAU1 = np.einsum('pdck,abcd->pkab', BBU1, v)
        vAAU1U2 = np.einsum('pkab,bikl->pail', vAAU1, self.U2)
        return np.einsum('pail,haij->hjlp', vAAU1U2, AAU1)
    


    def _compute_derivative(self) -> None:

        if self.A is None or self.B is None or self.U1 is None or self.U2 is None:
            raise ValueError("Can only call _compute_derivative for non-composite transfer matrix object.")
        
        A, B, U1, U2 = self.A, self.B, self.U1, self.U2

        tensors = [A.tensor, A.tensor, U1, U2, U1]
        indices = [[-1, 1, 2], [2, 3, -5], [1, 3, 4, -6], [-2, 4, 5, -7], [-3, 5, -4, -8]]
        order = [2, 1, 3, 4, 5]
        res = ncon(tensors, indices, order=order)

        tensors = [np.eye(self.Db), B.tensor.conj(), res]
        indices = [[-4, -9], [-11, 1, -8], [-1, -2, -3, -10, -5, -6, -7, 1]]

        # for i, X in enumerate(tensors):
        #     print(f"t[{i}] shape={X.shape}  indices={indices[i]}  (rank={X.ndim})")


        deriv_1 = ncon(tensors, indices)

        tensors = [B.tensor.conj(), res, np.eye(self.Db)]
        indices = [[-4, 1, -9], [-1, -2, -3, 1, -5, -6, -7, -10], [-11, -8]]
        deriv_2 = ncon(tensors, indices)

        self._derivative = deriv_1 + deriv_2
        

    def derivative(self):
        
        if self._derivative is not None:
            return self._derivative
        
        # if no parent tensors simply compute derivative and return
        if len(self.tape) == 0:
            self._compute_derivative()
            return self._derivative
        
        if len(self.tape) == 2:
            L = self.tape[0]
            R = self.tape[1]

            tensors = [L.derivative(), R.tensor]
            indices = [
                [-1, -2, -3, -4, 1, 2, 3, 4, -9, -10, -11], 
                [1, 2, 3, 4, -5, -6, -7, -8]
            ]
            D1 = ncon(tensors, indices, order=[1, 4, 2, 3])
            
            tensors = [L.tensor, R.derivative()]
            indices = [
                [-1, -2, -3, -4, 1, 2, 3, 4], 
                [1, 2, 3, 4, -5, -6, -7, -8, -9, -10, -11]
            ]
            D2 = ncon(tensors, indices, order=[1, 4, 2, 3])

            self._derivative = D1 + D2
            return self._derivative
        
        raise NotImplementedError()

if __name__ == "__main__":
    
   pass