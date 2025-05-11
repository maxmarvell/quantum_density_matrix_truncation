from abc import ABC, abstractmethod
import numpy as np
from ncon import ncon
from uniform_MPS import UniformMPS

class AbstractTransferMatrix(ABC):

    # @abstractmethod
    # def _matvec(self, v):
    #     pass

    # @abstractmethod
    # def _rmatvec(self, v):
    #     pass

    @abstractmethod
    def __matmul__(self, other):
        pass

    # @abstractmethod
    # def __rmatmul__(self, other):
    #     pass

    @abstractmethod
    def __pow__(self, other):
        pass

    @abstractmethod
    def derivative(self):
        pass

class TransferMatrix(AbstractTransferMatrix):
    
    def __init__(self, tensor: np.ndarray, tape: list = []):
        self.tensor = tensor
        self.d1 = tensor.shape[0]
        self.d2 = tensor.shape[1]
        self.tape = tape
        self.D = None

    @classmethod
    def new(cls, A: UniformMPS, B: UniformMPS):
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

        result = None
        power = self
        while n > 0:
            if n % 2 == 1:
                if result is None:
                    result = power
                else:
                    result = result @ power
            power = power @ power
            n //= 2
        return result
    
    def to_matrix(self):
        Da, Db = self.d1, self.d2
        return self.tensor.reshape(Da*Db, Da*Db)

class FirstOrderTrotterizedTransferMatrix(AbstractTransferMatrix):

    def __init__(self, tensor: np.ndarray, tape: list = []):
        self.tensor = tensor
        self.d1 = tensor.shape[0]
        self.d2 = tensor.shape[2]
        self.tape = tape
        self.D = None

    @classmethod
    def new(cls, A: UniformMPS, B: UniformMPS, U1, U2):

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
            tensor = ncon((self.tensor, other.tensor), ((-1, -2, -3, 1, 2, 3), (1, 2, 3, -4, -5, -6)))
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
            d1, d2 = self.d1, self.d2
            return TransferMatrix(np.identity(d1*d2).reshape(d1, d2, d1, d2))

        result = None
        power = self
        while n > 0:
            if n % 2 == 1:
                if result is None:
                    result = power
                else:
                    result = result @ power
            power = power @ power
            n //= 2
        return result
            

if __name__ == "__main__":

    d, p = 5, 2

    A = UniformMPS.from_random(d, p)
    E = TransferMatrix.new(A, A)

    assert np.allclose((E ** 3).tensor, ncon((E.tensor, E.tensor, E.tensor), ((-1, -2, 1, 2), (1, 2, 3, 4), (3, 4, -3, -4))))
    assert np.allclose(
        (E ** 7).tensor, 
        ncon((E.tensor, E.tensor, E.tensor, E.tensor, E.tensor, E.tensor, E.tensor), ((-1, -2, 1, 2), (1, 2, 3, 4), (3, 4, 5, 6), (5, 6, 7, 8), (7, 8, 9, 10), (9, 10, 11, 12), (11, 12, -3, -4)))
    )

    D2 = (E ** 2).derivative()
    assert np.allclose(
        D2, 
        ncon((A.tensor, np.eye(A.d), E.tensor), ((-1, -6, 1), (-2, -5), (1, -7, -3, -4))) + 
        ncon((E.tensor, np.eye(A.d), A.tensor), ((-1, -2, 1, -5), (-7, -4), (1, -6, -3)))
    )

    D3 = (E ** 3).derivative()
    assert np.allclose(
        D3, 
        ncon((E.tensor, D2), ((-1, -2, 1, 2), (1, 2, -3, -4, -5, -6, -7))) + 
        ncon((E.D, (E**2).tensor), ((-1, -2, 1, 2, -5, -6, -7), (1, 2, -3, -4))),
    )

    exact = np.zeros_like(D3)
    for i in range(3):
        exact += ncon(((E ** i).tensor, (E ** (3-i-1)).tensor, A.tensor), ((-1, -2, 1, -5), (2, -7, -3, -4), (1, -6, 2)))
    assert np.allclose(D3, exact)


    from utils.unitary import transverse_ising
    U = transverse_ising(0.1)
    assert np.allclose(ncon((U, np.conj(U)), ((-1, -2, 1, 2), (-3, -4, 1, 2))).reshape(4, 4), np.eye(4))

    res = ncon((A.tensor, A.tensor, U, np.conj(U), A.conj, A.conj), ((-1, 1, 2), (2, 3, -3), (1, 3, 4, 5), (6, 7, 4, 5), (-2, 6, 8), (8, 7, -4)))
    np.allclose(res, (E**2).tensor)

    # first order trotterized
    E = FirstOrderTrotterizedTransferMatrix.new(A, A, U, U)

    # E5 = E ** 20
    # print(E5.derivative().shape) 