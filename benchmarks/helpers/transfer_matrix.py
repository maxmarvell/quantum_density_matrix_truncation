import numpy as np
from ncon import ncon

from qdmt.transfer_matrix import AbstractTransferMatrix 
from qdmt.uniform_mps import UniformMps

            
class SecondOrderTrotterizedTransferMatrix(AbstractTransferMatrix):

    A: UniformMps
    B: UniformMps
    U1: np.ndarray
    U2: np.ndarray
    _derivative: np.ndarray | None

    def __init__(self, tensor: np.ndarray, tape: list = []):
        self.tensor = tensor
        self.d1, self.d2, self.p = tensor.shape[0], tensor.shape[3], tensor.shape[1]
        self.tape = tape
        self._derivative = None

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
            d1, d2, p = self.d1, self.d2, self.p
            E = self.tensor.reshape(p**2*d1*d2,-1)
            res = E @ E
            res = res.reshape(d1,p,p,d2,d1,p,p,d2)
            return SecondOrderTrotterizedTransferMatrix(res, [self, other])
            
        return NotImplemented
    
    def __pow__(self, n: int):

        if n == 0:
            d1, d2, p = self.d1, self.d2, self.p
            raise NotImplementedError

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
    A = UniformMps.new(4, 2)

    E = SecondOrderTrotterizedTransferMatrix.new(A, A, U1, U2)
    E.__pow__(5)
    
