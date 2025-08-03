from abc import ABC, abstractmethod
import numpy as np
from ncon import ncon
from typing import Self
from scipy.sparse.linalg import eigs

from qdmt.uniform_mps import UniformMps

class AbstractTransferMatrix(ABC):

    @abstractmethod
    def __matmul__(self, other):
        pass
    
    def __pow__(self, n: int):
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

    @abstractmethod
    def derivative(self) -> np.ndarray:
        pass

class TransferMatrix(AbstractTransferMatrix):
    
    A: UniformMps
    B: UniformMps

    def __init__(self, tensor: np.ndarray, tape: list = []):
        self.tensor = tensor
        self.d1 = tensor.shape[0]
        self.d2 = tensor.shape[1]
        self.tape = tape
        self.D = None

    @classmethod
    def new(cls, A: UniformMps, B: UniformMps):
        tensor = ncon((A.tensor, B.conj), ((-1, 1, -3), (-2, 1, -4)))
        E = cls(tensor)
        E.D = ncon((A.tensor, np.eye(B.d), np.eye(B.d)), ((-1, -6, -3), (-2, -5), (-7, -4)))
        E.A = A
        E.B = B
        return E

    def _matvec(self, v):
        return ncon((self.tensor, v), ((-1, -2, 1, 2), (1, 2)))
    
    def _rmatvec(self, v):
        return ncon((v, self.tensor), ((1, 2), (2, 1, -2, -1)))

    def __matmul__(self, other):

        if isinstance(other, TransferMatrix):
            tensor = ncon((self.tensor, other.tensor), ((-1, -2, 1, 2), (1, 2, -3, -4)))
            return TransferMatrix(tensor, [self, other])
        
        if isinstance(other, np.ndarray):
            if other.ndim == 2:
                return self._matvec(other)
            else:
                raise ValueError("Unsupported operand shape for multiplication.")
            
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray) and other.ndim == 2:
            return self._rmatvec(other)
        return NotImplemented
    
    def derivative(self):

        if self.D is not None:
            return self.D

        E1 = self.tape[0]
        E2 = self.tape[1]

        self.D = ncon(
            (E1.derivative(), E2.tensor), 
            ((-1, -2, 1, 2, -5, -6, -7), (1, 2, -3, -4))
            ) + ncon(
                (E1.tensor, E2.derivative()), 
                ((-1, -2, 1, 2), (1, 2, -3, -4, -5, -6, -7))
                )

        return self.D
    
    def __pow__(self, n: int):

        if n == 0:
            d1, d2 = self.d1, self.d2
            return TransferMatrix(np.identity(d1*d2).reshape(d1, d2, d1, d2))

        return super().__pow__(n)
    
    def to_matrix(self):
        Da, Db = self.d1, self.d2
        return self.tensor.reshape(Da*Db, Da*Db)
    
    def fidelity(self) -> np.complex128:
        M = self.to_matrix()
        r = eigs(M, k=1, which='LM', return_eigenvectors=False)
        return r[0]

class FirstOrderTrotterizedTransferMatrix(AbstractTransferMatrix):

    A: UniformMps
    B: UniformMps
    U1: np.ndarray
    U2: np.ndarray

    def __init__(self, tensor: np.ndarray, tape: list = []):
        self.tensor = tensor
        self.d1, self.p, self.d2 = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        self.tape = tape
        self.D = None

    @classmethod
    def new(cls, A: UniformMps, B: UniformMps, U1, U2):

        tensors = (B.conj, B.conj, U2, U1, A.tensor, A.tensor)
        indices = ((-3, 1, 2), (2, 3, -6), (-2, 4, 1, 3), (5, 6, 4, -5), (-1, 5, 7), (7, 6, -4))
        res = ncon(tensors, indices)
        E = FirstOrderTrotterizedTransferMatrix(res)

        E.A = A
        E.B = B
        E.U1 = U1
        E.U2 = U2

        return E
    
    def __matmul__(self, other):
        if isinstance(other, FirstOrderTrotterizedTransferMatrix):
            indices = [[-1, -2, -3, 1, 2, 3], [1, 2, 3, -4, -5, -6]]
            tensor = ncon([self.tensor, other.tensor], indices, order=[1, 3, 2])
            return FirstOrderTrotterizedTransferMatrix(tensor, [self, other])
            
        return NotImplemented
    
    def derivative(self):

        if self.D is not None:
            return self.D

        if len(self.tape) == 0:

            Ib = np.eye(self.d2, dtype=complex)

            tensors = (Ib, self.B.conj, self.U2, self.U1, self.A.tensor, self.A.tensor)
            indices = ((-3, -7), (-9, 1, -6), (-2, 2, -8, 1), (3, 4, 2, -5), (-1, 3, 5), (5, 4, -4))
            D1 = ncon(tensors, indices)

            tensors = (self.B.conj, Ib, self.U2, self.U1, self.A.tensor, self.A.tensor)
            indices = ((-3, 1, -7), (-9, -6), (-2, 2, 1, -8), (3, 4, 2, -5), (-1, 3, 5), (5, 4, -4))
            D2 = ncon(tensors, indices)

            self.D = D1 + D2

            return self.D
        
        if len(self.tape) == 2:

            E1 = self.tape[0]
            E2 = self.tape[1]

            tensors = (E1.derivative(), E2.tensor)
            indices = ((-1, -2, -3, 1, 2, 3, -7, -8, -9), (1, 2, 3, -4, -5, -6))
            D1 = ncon(tensors, indices)

            tensors = (E1.tensor, E2.derivative())
            indices = ((-1, -2, -3, 1, 2, 3), (1, 2, 3, -4, -5, -6, -7, -8, -9))
            D2 = ncon(tensors, indices)

            self.D = D1 + D2

            return self.D
        
        return NotImplemented

    def __pow__(self, n: int):

        if n == 0:
            d1, d2, p = self.d1, self.d2, self.p
            return FirstOrderTrotterizedTransferMatrix(np.identity(d1*d2*p).reshape(d1, p, d2, d1, p, d2))

        return super().__pow__(n)
            
class SecondOrderTrotterizedTransferMatrix(AbstractTransferMatrix):

    A: UniformMps
    B: UniformMps
    U1: np.ndarray
    U2: np.ndarray
    _derivative: np.ndarray | None

    def __init__(self, tensor: np.ndarray, tape: list[Self] = [], L: int = 1):
        self.tensor = tensor
        self.d1, self.d2, self.p = tensor.shape[0], tensor.shape[3], tensor.shape[1]
        self.tape = tape
        self._derivative = None
        self.L = L

    @classmethod
    def new(cls, A: UniformMps, B: UniformMps, U1: np.ndarray, U2: np.ndarray):
        tensors = [B.conj, B.conj, U1, U2, U1, A.tensor, A.tensor]
        indices = [[-4, 1, 2], [2, 3, -8], [-3, 4, 1, 3], [-2, 5, 4, -7], [6, 7, 5, -6], [-1, 6, 8], [8, 7, -5]]
        res = ncon(tensors, indices)
        E = SecondOrderTrotterizedTransferMatrix(res)
        E.A = A
        E.B = B
        E.U1 = U1
        E.U2 = U2
        return E

    def __matmul__(self, other):
        if isinstance(other, SecondOrderTrotterizedTransferMatrix):
            indices = [[-1, -2, -3, -4, 1, 2, 3, 4], [1, 2, 3, 4, -5, -6, -7, -8]]
            tensor = ncon([self.tensor, other.tensor], indices, order=[1, 4, 2, 3])
            return SecondOrderTrotterizedTransferMatrix(tensor, [self, other], L=self.L+other.L)
            
        return NotImplemented
    
    def __pow__(self, n: int):

        if n == 0:
            d1, d2, p = self.d1, self.d2, self.p
            return FirstOrderTrotterizedTransferMatrix(np.identity(d1*d2*p).reshape(d1, p, d2, d1, p, d2))

        return super().__pow__(n)

    def _compute_derivative(self) -> None:
        
        A, B, U1, U2 = self.A.tensor, self.B.tensor, self.U1, self.U2
        d = self.d2

        tensors = [A, A, U1, U2, U1]
        indices = [[-1, 1, 2], [2, 3, -5], [1, 3, 4, -6], [-2, 4, 5, -7], [-3, 5, -4, -8]]
        order = [2, 1, 3, 4, 5]
        res = ncon(tensors, indices, order=order)

        tensors = [np.eye(d), B.conj(), res]
        indices = [[-4, -9], [-11, 1, -8], [-1, -2, -3, -10, -5, -6, -7, 1]]
        deriv_1 = ncon(tensors, indices)

        tensors = [B.conj(), res, np.eye(d)]
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
            E1 = self.tape[0]
            E2 = self.tape[1]

            tensors = [E1.derivative(), E2.tensor]
            indices = [
                [-1, -2, -3, -4, 1, 2, 3, 4, -9, -10, -11], 
                [1, 2, 3, 4, -5, -6, -7, -8]
            ]
            D1 = ncon(tensors, indices, order=[1, 4, 2, 3])
            
            tensors = [E1.tensor, E2.derivative()]
            indices = [
                [-1, -2, -3, -4, 1, 2, 3, 4], 
                [1, 2, 3, 4, -5, -6, -7, -8, -9, -10, -11]]
            D2 = ncon(tensors, indices, order=[1, 4, 2, 3])

            self._derivative = D1 + D2
            return self._derivative
        
        return NotImplemented
        

if __name__ == "__main__":

    from qdmt.model import TransverseFieldIsing
    TFIM = TransverseFieldIsing(.2, .1)
    U1, U2 = TFIM.trotter_second_order()
    A = UniformMps.new(2, 2)
    E = SecondOrderTrotterizedTransferMatrix.new(A, A, U1, U2)

    T = E.__pow__(10)
    T.derivative()