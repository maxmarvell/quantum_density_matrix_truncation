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

    def __init__(self, tensor: np.ndarray):
        self.tensor = tensor
        self.d1, self.d2, self.p = tensor.shape[0], tensor.shape[3], tensor.shape[1]
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
            indices = [[-1, -2, -3, -4, 1, 2, 3, 4], [1, 2, 3, 4, -5, -6, -7, -8]]
            tensor = ncon([self.tensor, other.tensor], indices, order=[1, 4, 2, 3])
            return SecondOrderTrotterizedTransferMatrix(tensor)
            
        return NotImplemented
    
    def __pow__(self, n: int):
        if n == 0:
            d1, d2, p = self.d1, self.d2, self.p
            return SecondOrderTrotterizedTransferMatrix(np.identity(d1*d2*p*p).reshape(d1, p, p, d2, d1, p, p, d2))

    def derivative_of_power(E, n):
        """Computes the derivative of E**n more efficiently."""
        if n == 0:
            # Derivative of identity is zero (or a zero tensor of correct shape)
            return np.zeros_like(E.derivative()) 

        dE = E.derivative()
        
        # Pre-compute powers of E needed
        powers = [None] * n
        powers[0] = E
        for i in range(1, n):
            powers[i] = powers[i-1] @ E
            
        # Identity matrix (as a TransferMatrix)
        identity_E = E.__pow__(0) 

        total_derivative = None
        for i in range(n):
            # E^i
            left_term = powers[i-1].tensor if i > 0 else identity_E.tensor
            # E^(n-1-i)
            right_term = powers[n-2-i].tensor if (n-1-i) >= 0 else identity_E.tensor

            # contract the three terms
            tensors = [left_term, dE, right_term]
            indices = [[-1, -2, -3, -4, 1, 3, 4, 2], [1, 3, 4, 2, 5, 7, 8, 6, -9, -10, -11], [5, 7, 8, 6, -5, -6, -7, -8]]
            current_term = ncon(tensors, indices)

            if total_derivative is None:
                total_derivative = current_term
            else:
                total_derivative += current_term # Assumes you implement __add__

        return total_derivative
        

    def derivative(self):
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

        return deriv_1 + deriv_2


if __name__ == "__main__":

    from qdmt.model import TransverseFieldIsing

    TFIM = TransverseFieldIsing(.2, .1)
    U1, U2 = TFIM.trotter_second_order()
    A = UniformMps.new(4, 2)

    E = SecondOrderTrotterizedTransferMatrix.new(A, A, U1, U2)
    E.derivative_of_power(5)
    
