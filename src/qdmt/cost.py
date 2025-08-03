from abc import ABC, abstractmethod
from ncon import ncon
import numpy as np
import numpy.typing as npt
from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrix import (
    FirstOrderTrotterizedTransferMatrix as FirstOrder,
    SecondOrderTrotterizedTransferMatrix as SecondOrder,
    TransferMatrix
)
from typing import Self
from qdmt.model import AbstractModel
from qdmt.utils.mps import trotter_step
from qdmt.fixed_point import RightFixedPoint
import copy

import opt_einsum as oe

class AbstractCostFunction(ABC):

    def __init__(self, A: UniformMps, L: int):
        self.L = L
        self.A = A
        AAdag = TransferMatrix.new(self.A, self.A)
        self.rA = RightFixedPoint(AAdag)

    @abstractmethod
    def cost(self, B: UniformMps, rB: RightFixedPoint) -> np.float64:
        pass

    @abstractmethod
    def derivative(self, B: UniformMps, rB: RightFixedPoint) -> np.ndarray:
        pass

    def copy(self) -> 'AbstractCostFunction':
        new_f = copy.copy(self)
        return new_f
    
    def fg(self, B: UniformMps):
        n, p = B.matrix.shape
        C = self.cost(B)
        D = self.derivative(B)
        return np.abs(C), D.reshape(n, p)

class HilbertSchmidt(AbstractCostFunction):

    def __init__(self, A, L):
        super().__init__(A, L)

        # compute and store the cost of rho(A)^2
        AAdag = TransferMatrix.new(self.A, self.A)
        T_AA = AAdag.__pow__(L).tensor
        self.costAA  = ncon((T_AA, T_AA, self.rA.tensor, self.rA.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6)))

        self.normA = ncon((T_AA, self.rA.tensor), ((1, 1, 3, 2), (3, 2)))


    def cost(self:Self, B: UniformMps, rB: RightFixedPoint) -> np.float64:

        BBdag = TransferMatrix.new(B, B)

        self.A.correlation_length()

        L = self.L
        res = 0

        # rho(B)^2 contribution
        T_BB = BBdag.__pow__(L).tensor
        res += ncon((T_BB, T_BB, rB.tensor, rB.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6)))

        # rho(A)^2 contribution
        res += self.costAA

        # 2*rho(A)*rho(B) contribution
        res -= 2 * self._compute_rho_A_rho_B(B, rB)

        return np.abs(res)

    def _compute_rho_A_rho_B(self, B: UniformMps, rB: RightFixedPoint):

        ABdag = TransferMatrix.new(self.A, B)
        BAdag = TransferMatrix.new(B, self.A)

        T_AB = ABdag.__pow__(self.L).tensor
        T_BA = BAdag.__pow__(self.L).tensor

        return abs(ncon((T_AB, T_BA, rB.tensor, self.rA.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6))))
    
    def derivative(self, B: UniformMps, rB: RightFixedPoint):

        ABdag = TransferMatrix.new(self.A, B)
        BAdag = TransferMatrix.new(B, self.A)
        BBdag = TransferMatrix.new(B, B)

        L = self.L
        res = np.zeros_like(B.tensor, dtype=np.complex128)

        # rho(B)^2 contribution
        T_BB = BBdag.__pow__(L)
        D = T_BB.derivative()
        res += 2 * ncon((D, T_BB.tensor, rB.tensor, rB.tensor), ((1, 2, 3, 4, -1, -2, -3), (2, 1, 5, 6), (5, 4), (3, 6)))

        # right fixed point
        tensors = (T_BB.tensor, T_BB.tensor, rB.tensor)
        indices = ((1, 2, 3, -1), (2, 1, -2, 4), (3, 4))
        v = ncon(tensors, indices)
        res += 2 * rB.derivative(v)

        # 2rho(A)rho(B) contribution
        T_AB = ABdag.__pow__(L)
        T_BA = BAdag.__pow__(L)
        D = T_AB.derivative()
        res -= 2 * ncon((D, T_BA.tensor, rB.tensor, self.rA.tensor), ((1, 2, 3, 4, -1, -2, -3), (2, 1, 5, 6), (5, 4), (3, 6)))

        # right fixed point
        tensors = (T_AB.tensor, T_BA.tensor, self.rA.tensor)
        indices = ((1, 2, 3, -1), (2, 1, -2, 4), (3, 4))
        v = ncon(tensors, indices)
        res -= 2 * rB.derivative(v)

        return res

class EvolvedHilbertSchmidt(AbstractCostFunction):

    def __init__(self, A: UniformMps, model: AbstractModel, L: int, trotterization_order: int = 2):

        super().__init__(A, L)

        self.trotterization_order = trotterization_order

        if trotterization_order == 1:
            self.U1, self.U2 = model.trotter_first_order()

            # compute all invarients of the cost function
            self.purity_A = self._compute_first_totterized_purity()
            self.left_auxillary = self._compute_left_auxillary_first_trotterized_overlap()
            self.right_auxillary = self._compute_right_auxillary_first_trotterized_overlap()

        elif trotterization_order == 2:

            if L % 2 == 1:
                raise NotImplementedError("Second-order Trotterization is not implemented for odd L.")


            self.U1, self.U2 = model.trotter_second_order()

            # compute all invarients of the cost function
            self.purity_A = self._compute_second_trotterized_purity()
            self._compute_auxillary_second_trotterized()

            # assume A and B have same shape
            self._compile_contractions()

        else:
            raise ValueError(f"Trotterization order must be 1 or 2, but got {trotterization_order}")

        self.rhoA = self._compute_rhoA()

    def cost(self, B: UniformMps, rB: RightFixedPoint):

        hilbert_schmidt_distance = 0 + 0j

        # rho(B)^2 contribution
        hilbert_schmidt_distance += self._compute_trace_product_rhoB_rhoB(B, rB)

        # rho(A(t+dt))^2 contribution
        hilbert_schmidt_distance += self.purity_A

        # rho(A(t+dt))rho(B) contribution        
        hilbert_schmidt_distance -= 2 * self._compute_trace_product_rhoA_rhoB(B, rB)

        return np.abs(hilbert_schmidt_distance)

    def derivative(self, B: UniformMps, rB: RightFixedPoint):

        derivative = np.zeros_like(B.tensor, dtype=np.complex128)

        # rho(B)^2 contribution
        derivative += 2 * self._compute_derivative_rho_B_rho_B(B, rB)

        # rho(A(t+dt))rho(B) contribution
        derivative -= 2 * self._compute_derivative_rho_A_rho_B(B, rB)

        return derivative

    def _compute_first_totterized_purity(self) -> np.complex128:

        A, U1, U2, L = self.A, self.U1, self.U2, self.L

        # calculate left dangling tensor
        tensors = [A.tensor, A.tensor, U1, U1.conj(), A.conj, A.conj]
        indices = [[1, 2, 3], [3, 4, -1], [2, 4, 5, -2], [6, 7, 5, -3], [1, 6, 8], [8, 7, -4]]
        tmp = ncon(tensors, indices)

        tensors = [tmp, tmp]
        indices = [[-1, 1, 2, -4], [-3, 2, 1, -2]]
        tmp_left = ncon(tensors, indices)

        AAdag = TransferMatrix.new(A, A)

        # different behaviour for an odd and even number of sites
        if L % 2 == 0:
            tensors = [A.tensor, A.tensor, U1, U1.conj(), A.conj, A.conj, self.rA.tensor]
            indices = [[-1, 1, 2], [2, 3, 4], [1, 3, -2, 5], [6, 7, -3, 5], [-4, 6, 8], [8, 7, 9], [4, 9]]
            tmp = ncon(tensors, indices)

            tensors = [tmp, tmp]
            indices = [[-1, 1, 2, -4], [-3, 2, 1, -2]]
            tmp_right = ncon(tensors, indices)

            T_AA = AAdag.__pow__(L-2).tensor

        else:
            tmp = A.mps_chain(4)
            tensors = [tmp, U1, U1, U2]
            indices = [[-1, 1, 2, 3, 4, -6], [1, 2, -2, 5], [3, 4, 6, -5], [5, 6, -3, -4]]
            tmp = ncon(tensors, indices)

            tensors = [tmp, tmp.conj(), self.rA.tensor]
            indices = [[-1, -2, -3, 1, 2, 3], [-4, -5, -6, 1, 2, 4], [3, 4]]
            tmp = ncon(tensors, indices)

            tensors = [tmp, tmp]
            indices = [[-1, 1, 2, -4, 3, 4], [-3, 3, 4, -2, 1, 2]]
            tmp_right = ncon(tensors, indices)

            T_AA = AAdag.__pow__(L-3).tensor

        
        tensors = (tmp_left, T_AA, T_AA, tmp_right)
        indices = ((1, 2, 3, 4), (1, 2, 5, 6), (3, 4, 7, 8), (5, 6, 7, 8))
        return ncon(tensors, indices)

    def _compute_left_auxillary_first_trotterized_overlap(self) -> np.complex128:

        A, U1 = self.A, self.U1

        tensors = [A.tensor, A.tensor, U1]
        indices = [[-1, 1, 2], [2, 3, -3], [1, 3, -2, -4]]
        return ncon(tensors, indices)

    def _compute_right_auxillary_first_trotterized_overlap(self) -> np.complex128:

        L, A, U1, U2 = self.L, self.A, self.U1, self.U2

        # different behaviour for an odd and even number of sites
        if L % 2 == 0:
            tensors = [U1.conj(), A.conj, A.conj]
            indices = [[1, 2, -1, -3], [-2, 1, 3], [3, 2, -4]]
            return ncon(tensors, indices)

        tmp1 = A.mps_chain(2)
        tmp2 = A.mps_chain(4)

        tensors = [tmp1, U1]
        indices = [[-1, 1, 2, -4], [1, 2, -2, -3]]
        tmp1 = ncon(tensors, indices)

        tensors = [tmp2, U1, U1, U2]
        indices = [[-1, 1, 2, 3, 4, -6], [1, 2, -2, 5], [3, 4, 6, -5], [5, 6, -3, -4]]
        tmp2 = ncon(tensors, indices)

        tensors = [tmp1, tmp2.conj(), U2, self.rA.tensor]
        indices = [[-1, 1, 2, 3], [-6, -5, -4, 4, 2, 5], [-2, 1, -3, 4], [3, 5]]
        return ncon(tensors, indices)

    def _compute_second_trotterized_purity(self) -> np.complex128:

        A, U1, U2, L, rA = self.A, self.U1, self.U2, self.L, self.rA

        if L <= 4:

            N = L + 4

            mps = A.mps_chain(N)

            # trotterized evolve
            mps = trotter_step(mps, U1)
            mps = trotter_step(mps, U2, start=1, stop=2)
            mps = trotter_step(mps, U2, start=N-3, stop=N-2)

            # compute reduced density matrix over patch size L
            tensors = [mps, mps.conj(), rA.tensor]
            indices = [
                [1, 2, 3] + [-(i+1) for i in range(L)] + [5, 6, 7],
                [1, 2, 3] + [-(i+1+L) for i in range(L)] + [5, 6, 8],
                [7, 8]
            ]
            rho_A = ncon(tensors, indices)

            # compute trace product
            tensors = [rho_A, rho_A]
            indices = [
                [i for i in range(1, 2*L+1)],
                [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)]
            ]
            return ncon(tensors, indices)
        
        else: 

            # create a base mps to calculate the left and right mixing
            mps = A.mps_chain(4)
            mps = trotter_step(mps, U1)
            mps = trotter_step(mps, U2, start=1)

            tensors = [mps, mps.conj()]
            indices = [[1, 2, 3, -4, -5, -6], [1, 2, 3, -3, -2, -1]]
            l = ncon(tensors, indices)

            tensors = [mps, mps.conj(), rA.tensor]
            indices = [[-1, -2, -3, 1, 2, 3], [-6, -5, -4, 1, 2, 4], [3, 4]]
            r = ncon(tensors, indices)

            AA = TransferMatrix.new(A, A).__pow__(L-4).tensor

            # compute trace product
            tensors = [l, l, AA, AA, r, r]
            indices = [
                [1, 2, 3, 4, 5, 6], 
                [7, 5, 4, 3, 2, 8], 
                [6, 7, 9, 16], 
                [8, 1, 15, 14], 
                [9, 10, 11, 12, 13, 14], 
                [15, 13, 12, 11, 10, 16]
            ]
            return ncon(tensors, indices)

    def _compute_auxillary_second_trotterized(self) -> tuple[npt.NDArray[np.complex128]]:

        A, U1, U2, rA = self.A, self.U1, self.U2, self.rA

        mps = A.mps_chain(4)
        mps = trotter_step(mps, U1)
        mps = trotter_step(mps, U2, start=1)

        self.left_auxillary = mps
        self.right_auxillary = ncon([mps.conj(), rA.tensor], [[-1, -2, -3, -4, -5, 1], [-6, 1]])

        return self.left_auxillary, self.right_auxillary
    
    def _compute_trace_product_rhoA_rhoB(self, B: UniformMps, rB: RightFixedPoint) -> np.complex128:

        # heurisistic to determine best contraction pathway
        DA = self.A.d
        DB = B.d
        d = B.p
        L = self.L

        if (DA**3*DB**3*d**6*np.log(L) <= d**(2*L)):
                
            A, U1, U2, L = self.A, self.U1, self.U2, self.L

            if self.trotterization_order == 1:

                ABdag = FirstOrder.new(A, B, U1, U2)
                BAdag = FirstOrder.new(B, A, U2.conj().transpose(2, 3, 0, 1), U1.conj().transpose(2, 3, 0, 1))

                T_AB = ABdag.__pow__(L//2).tensor
                T_BA = BAdag.__pow__(L//2).tensor

                if L % 2 == 0:
                    tensors = (self.left_auxillary, T_AB, T_BA, self.right_auxillary, self.rA.tensor, rB.tensor)
                    indices = ((1, 2, 3, 4), (3, 4, 5, 6, 7, 8), (5, 2, 1, 9, 10, 11), (10, 11, 7, 12), (6, 12), (9, 8))
                else:
                    tensors = [self.left_auxillary, T_AB, T_BA, B.conj, rB.tensor, B.tensor, self.right_auxillary]
                    indices  = [[1, 2, 3, 4], [3, 4, 5, 6, 7, 8], [5, 2, 1, 9, 10, 11], [8, 12, 13], [14, 13], [9, 15, 14], [6, 7, 12, 15, 10, 11]]

            elif self.trotterization_order == 2:

                ABdag = SecondOrder.new(A, B, U1, U2)
                BAdag = SecondOrder.new(B, A, U1.conj().transpose(2, 3, 0, 1), U2.conj().transpose(2, 3, 0, 1))

                T_AB = ABdag.__pow__(L//2).tensor
                T_BA = BAdag.__pow__(L//2).tensor

                tensors = [self.left_auxillary, T_AB, T_BA, rB.tensor, self.right_auxillary]
                indices = [
                    [1, 2, 3, 4, 5, 6],
                    [6, 5, 4, 7, 15, 14, 13, 9],
                    [7, 3, 2, 1, 8, 12, 11, 10],
                    [8, 9],
                    [10, 11, 12, 13, 14, 15]
                ]

            return ncon(tensors, indices)

        else:
            rhoB = self._compute_rhoB(B, rB)

            L = self.L
            tensors = (self.rhoA, rhoB)
            indices = [
                [i for i in range(1, 2*L+1)],
                [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)]
            ]
            rhoArhoB = ncon(tensors, indices)

            return rhoArhoB

    def _compute_trace_product_rhoB_rhoB(self, B: UniformMps, rB: RightFixedPoint) -> np.complex128:
        BBdag = TransferMatrix.new(B, B)
        T_BB = BBdag.__pow__(self.L).tensor
        return ncon((T_BB, T_BB, rB.tensor, rB.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6)))

    def _compute_derivative_rho_A_rho_B(self, B: UniformMps, rB: RightFixedPoint) -> npt.NDArray[np.complex128]:

        # heurisistic to determine best contraction pathway
        DA = self.A.d
        DB = B.d
        d = B.p
        L = self.L

        if (DA**3*DB**5*d**7*np.log(L) <= d**(2*L)*L*DB**2):

            A, L, U1, U2 = self.A, self.L, self.U1, self.U2
            res = np.zeros_like(B.tensor, dtype=np.complex128)

            if self.trotterization_order == 1:
                ABdag = FirstOrder.new(A, B, U1, U2)
                BAdag = FirstOrder.new(B, A, U2.conj().transpose(2, 3, 0, 1), U1.conj().transpose(2, 3, 0, 1))
            else:
                ABdag = SecondOrder.new(A, B, U1, U2)
                BAdag = SecondOrder.new(B, A, U1.conj().transpose(2, 3, 0, 1), U2.conj().transpose(2, 3, 0, 1))

            T_BA = BAdag.__pow__(L // 2)
            T_AB = ABdag.__pow__(L // 2)
            
            # rho(A(t+dt))rho(B) contribution
            D = T_AB.derivative()

            # compute central contribution
            if self.trotterization_order == 1:
                res += self._contract_rhoA_rhoB_central_derivative(D, T_AB, T_BA, B, rB)
            else:
                res += self._contract_rhoA_rhoB_central_derivative(D, T_BA.tensor, rB.tensor)

            res += self._contract_rhoA_rhoB_fixed_point_derivative(T_AB, T_BA, B, rB)
            return res

        else:
            L, rhoA = self.L, self.rhoA
            D = np.zeros_like(B.tensor)

            # central
            tmp = np.zeros_like(B.tensor)
            for i in range(L):
                drhoB = self.compute_drhoB(i, B, rB)

                tensors = [rhoA, drhoB]
                indices = [
                    [i for i in range(1, 2*L+1)],
                    [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)] + [-1, -2, -3]
                ]
                tmp += ncon(tensors, indices)

            drhoBrhoAdrB = self.compute_drhoBrhoAdrB(B)
            tmp += rB.derivative(drhoBrhoAdrB.T)

            return tmp


    def _compute_derivative_rho_B_rho_B(self, B: UniformMps, rB: RightFixedPoint) -> npt.NDArray[np.complex128]:
        
        res = np.zeros_like(B.tensor, dtype=np.complex128)

        BBdag = TransferMatrix.new(B, B)

        T_BB = BBdag ** self.L

        # rho(B)^2 contribution
        D = T_BB.derivative()
        tensors = (D, T_BB.tensor, rB.tensor, rB.tensor)
        indices = ((1, 2, 3, 4, -1, -2, -3), (2, 1, 5, 6), (3, 6), (5, 4))
        res += ncon(tensors, indices)

        # right fixed point
        tensors = (T_BB.tensor, T_BB.tensor, rB.tensor)
        indices = ((1, 2, 3, -1), (2, 1, -2, 4), (3, 4))
        v = ncon(tensors, indices)
        res += rB.derivative(v)

        return res
        
    def _contract_rhoA_rhoB_central_derivative(self, D: np.ndarray, T_AB: TransferMatrix, T_BA: TransferMatrix, B: UniformMps, rB: RightFixedPoint):

        L = self.L

        if self.trotterization_order == 1:

            if L % 2 == 0:
                tensors = (self.left_auxillary, D, T_BA.tensor, self.right_auxillary, self.rA.tensor, rB.tensor)
                indices = ((1, 2, 3, 4), (3, 4, 5, 6, 7, 8, -1, -2, -3), (5, 2, 1, 9, 10, 11), (10, 11, 7, 12), (6, 12), (9, 8))
                return ncon(tensors, indices)

            tensors = [self.left_auxillary, T_AB.tensor, T_BA.tensor, rB.tensor, B.tensor, self.right_auxillary]
            indices = [[1, 2, 3, 4], [3, 4, 5, 6, 7, -1], [5, 2, 1, 8, 9, 10], [11, -3], [8, 12, 11], [6, 7, -2, 12, 9, 10]]
            tmp = ncon(tensors, indices)

            tensors = [self.left_auxillary, D, T_BA.tensor, B.conj, rB.tensor, B.tensor, self.right_auxillary]
            indices = [[1, 2, 3, 4], [3, 4, 5, 6, 7, 8, -1, -2, -3], [5, 2, 1, 9, 10, 11], [8, 12, 13], [14, 13], [9, 15, 14], [6, 7, 12, 15, 10, 11]]
            return ncon(tensors, indices) + tmp

        elif self.trotterization_order == 2:

            tensors = [self.left_auxillary, D, T_BA.tensor, rB.tensor, self.right_auxillary]
            indices = [
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 7, 15, 14, 13, 9, -1, -2, -3],
                [7, 3, 2, 1, 8, 12, 11, 10],
                [8, 9],
                [10, 11, 12, 13, 14, 15]
            ]
            order = [8, 1, 2, 3, 10, 11, 12, 6, 7, 9, 15, 4, 5, 13, 14]
            return ncon(tensors, indices, order=order, )

    def _contract_rhoA_rhoB_fixed_point_derivative(self, T_AB, T_BA, B: UniformMps, rB: RightFixedPoint):

        L = self.L

        if self.trotterization_order == 1:
            if L % 2 == 0:        
                tensors = (self.left_auxillary, T_AB.tensor, T_BA.tensor, self.right_auxillary, self.rA.tensor)
                indices = ((1, 2, 3, 4), (3, 4, 5, 6, 7, -1), (5, 2, 1, -2, 8, 9), (8, 9, 7, 10), (6, 10))
            else:
                tensors = [self.left_auxillary, T_AB.tensor, T_BA.tensor, B.conj, B.tensor, self.right_auxillary]
                indices = [[1, 2, 3, 4], [3, 4, 5, 6, 7, 8], [5, 2, 1, 9, 10, 11], [8, 12, -1], [9, 13, -2], [6, 7, 12, 13, 10, 11]]

        elif self.trotterization_order == 2:
            tensors = [self.left_auxillary, T_AB.tensor, T_BA.tensor, self.right_auxillary]
            indices = [
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 7, 13, 12, 11, -1],
                [7, 3, 2, 1, -2, 10, 8, 9],
                [9, 8, 10, 11, 12, 13]
            ]
        
        v = ncon(tensors, indices)
        return rB.derivative(v)
    
    def _compile_contractions(self) -> None:

        D, d, _ = self.A.tensor.shape
        eq = 'abcdef,fedgonmipqr,gcbahlkj,hi,jklmno->pqr'
        constants = [0, 4]
        ops = [
            self.left_auxillary,
            (D, d, d, D, D, d, d, D, D, d, D), 
            (D, d, d, D, D, d, d, D),
            (D, D),
            self.right_auxillary
        ]

        expr = oe.contract_expression(eq, *ops, constants=constants)
        self._contract_rhoA_rhoB_central_derivative = expr

    def _compute_rhoA(self) -> npt.NDArray:
        
        A, U1, U2, rA, L = self.A, self.U1, self.U2, self.rA, self.L

        N = 0
        if L % 2 == 0 and self.trotterization_order == 1:
            N = L + 2
        elif self.trotterization_order == 1:
            N = L + 3
        elif L % 2 == 0 and self.trotterization_order == 2:
            N = L + 4

        mps = A.mps_chain(N)

        # first trotter step
        mps = trotter_step(mps, U1)

        # second trotter step
        mps = trotter_step(mps, U2, start=1)

        # conditionally apply final trotter step
        if self.trotterization_order == 2:
            mps = trotter_step(mps, U1)

        tensors = [mps, mps.conj(), rA.tensor]

        if self.trotterization_order == 1: 
            indices = [
                [1, 2] + [-(i+1) for i in range(L)] + ([3, 4] if (L%2 == 0) else [3, 4, 5]),
                [1, 2] + [-(i+L+1) for i in range(L)] + ([3, 5] if (L%2 == 0) else [3, 4, 6]),
                [4, 5] if (L%2 == 0) else [5, 6]
            ]
        else:
            indices = [
                [1, 2, 3] + [-(i+1) for i in range(L)] + [4, 5, 6],
                [1, 2, 3] + [-(i+L+1) for i in range(L)] + [4, 5, 7],
                [6, 7]
            ]

        rhoA = ncon(tensors, indices)

        return rhoA
    
    def _compute_rhoB(self, B: UniformMps, rB: RightFixedPoint) -> npt.NDArray[np.complex128]:

        L = self.L
        chain = B.mps_chain(self.L)

        tensors = (chain, chain.conj(), rB.tensor)
        indices = [
            [1] + [-i for i in range(1, L+1)] + [2],
            [1] + [-i for i in range(L+1, 2*L+1)] + [3],
            [2, 3]
        ]

        rhoB = ncon(tensors, indices)
        return rhoB
    
    def compute_drhoB(self, i: int, B: UniformMps, rB: RightFixedPoint) -> npt.NDArray[np.complex128]:

        L = self.L
        chain = B.mps_chain(self.L)
        sub_chains = [B.mps_chain(i).conj(), B.mps_chain(L-i-1).conj()]

        tensors = [chain, *sub_chains, np.eye(B.p), rB.tensor]
        indices = [
            [1] + [-i for i in range(1, L+1)] + [2],
            [1] + [-i for i in range(L+1, L+1+i)] + [-2*L-1],
            [-2*L-3] + [-i for i in range(L+2+i, 2*L+1)] + [3],
            [-L-1-i, -2*L-2],
            [2, 3]
        ]
        res = ncon(tensors, indices)

        return res

    def compute_drhoBrhoAdrB(self, B: UniformMps) -> npt.NDArray[np.complex128]:

        rhoA, L = self.rhoA, self.L
        chain = B.mps_chain(self.L)

        tensors = (chain, chain.conj())
        indices = [
            [1] + [-i for i in range(1, L+1)] + [-(2*L+1)],
            [1] + [-i for i in range(L+1, 2*L+1)] + [-(2*L+2)]
        ]
        drhoBdrB = ncon(tensors, indices)

        tensors = [drhoBdrB, rhoA]
        indices = [
            [i for i in range(1, 2*L+1)] + [-1, -2],
            [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)]
        ]
        drhoBrhoAdrB = ncon(tensors, indices)

        return drhoBrhoAdrB

if __name__ == "__main__":
    pass