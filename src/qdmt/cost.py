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

class AbstractCostFunction(ABC):

    def __init__(self, A: UniformMps, B: UniformMps, L: int):
        self.L = L
        self.A = A
        self.B = B

        AAdag = TransferMatrix.new(self.A, self.A)
        self.rA = RightFixedPoint(AAdag)

    @property
    def B(self) -> UniformMps:
        return self._B
    
    @B.setter
    def B(self, value):
        self._B = value
        BBdag = TransferMatrix.new(value, value)
        self.rB = RightFixedPoint(BBdag)

    
    @abstractmethod
    def cost(self) -> np.complex128:
        pass

    @abstractmethod
    def derivative(self) -> np.ndarray:
        pass

    def copy(self) -> 'AbstractCostFunction':
        new_f = copy.copy(self)
        return new_f

class HilbertSchmidt(AbstractCostFunction):

    def __init__(self, A, B, L):
        super().__init__(A, B, L)

        # compute and store the cost of rho(A)^2
        AAdag = TransferMatrix.new(self.A, self.A)
        T_AA = AAdag.__pow__(L).tensor
        self.costAA  = ncon((T_AA, T_AA, self.rA.tensor, self.rA.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6)))


    def cost(self:Self) -> np.float64:

        ABdag = TransferMatrix.new(self.A, self.B)
        BAdag = TransferMatrix.new(self.B, self.A)
        BBdag = TransferMatrix.new(self.B, self.B)

        L = self.L
        res = 0

        # rho(B)^2 contribution
        T_BB = BBdag.__pow__(L).tensor
        res += ncon((T_BB, T_BB, self.rB.tensor, self.rB.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6)))

        # rho(A)^2 contribution
        res += self.costAA

        # 2rho(A)rho(B) contribution
        T_AB = ABdag.__pow__(L).tensor
        T_BA = BAdag.__pow__(L).tensor
        res -= 2 * ncon((T_AB, T_BA, self.rB.tensor, self.rA.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6)))

        return np.abs(res)
    
    def derivative(self):

        ABdag = TransferMatrix.new(self.A, self.B)
        BAdag = TransferMatrix.new(self.B, self.A)
        BBdag = TransferMatrix.new(self.B, self.B)

        L = self.L
        res = np.zeros_like(self.B.tensor, dtype=np.complex128)

        # rho(B)^2 contribution
        T_BB = BBdag.__pow__(L)
        D = T_BB.derivative()
        res += 2 * ncon((D, T_BB.tensor, self.rB.tensor, self.rB.tensor), ((1, 2, 3, 4, -1, -2, -3), (2, 1, 5, 6), (5, 4), (3, 6)))

        # right fixed point
        tensors = (T_BB.tensor, T_BB.tensor, self.rB.tensor)
        indices = ((1, 2, 3, -1), (2, 1, -2, 4), (3, 4))
        v = ncon(tensors, indices)
        res += 2 * self.rB.derivative(v)

        # 2rho(A)rho(B) contribution
        T_AB = ABdag.__pow__(L)
        T_BA = BAdag.__pow__(L)
        D = T_AB.derivative()
        res -= 2 * ncon((D, T_BA.tensor, self.rB.tensor, self.rA.tensor), ((1, 2, 3, 4, -1, -2, -3), (2, 1, 5, 6), (5, 4), (3, 6)))

        # right fixed point
        tensors = (T_AB.tensor, T_BA.tensor, self.rA.tensor)
        indices = ((1, 2, 3, -1), (2, 1, -2, 4), (3, 4))
        v = ncon(tensors, indices)
        res -= 2 * self.rB.derivative(v)

        return res

class EvolvedHilbertSchmidt(AbstractCostFunction):

    def __init__(self, A: UniformMps, B: UniformMps, model: AbstractModel, L: int, trotterization_order: int = 2):

        super().__init__(A, B, L)

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

        else:
            raise ValueError(f"Trotterization order must be 1 or 2, but got {trotterization_order}")

    def cost(self):

        hilbert_schmidt_distance = 0 + 0j

        # rho(B)^2 contribution
        hilbert_schmidt_distance += self._compute_trace_product_rhoB_rhoB()

        # rho(A(t+dt))^2 contribution
        hilbert_schmidt_distance += self.purity_A

        # rho(A(t+dt))rho(B) contribution        
        hilbert_schmidt_distance -= 2 * self._compute_trace_product_rhoA_rhoB()

        return np.abs(hilbert_schmidt_distance)

    def derivative(self):

        derivative = np.zeros_like(self.B.tensor, dtype=np.complex128)

        # rho(B)^2 contribution
        derivative += 2 * self._compute_derivative_rho_B_rho_B()

        # rho(A(t+dt))rho(B) contribution
        derivative -= 2 * self._compute_derivative_rho_A_rho_B()

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

        tensors = [A.tensor, A.tensor, self.U1]
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
    
    def _compute_trace_product_rhoA_rhoB(self) -> np.complex128:

        A, B, U1, U2, rB, L = self.A, self.B, self.U1, self.U2, self.rB, self.L

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

    def _compute_trace_product_rhoB_rhoB(self) -> np.complex128:
        BBdag = TransferMatrix.new(self.B, self.B)
        T_BB = BBdag.__pow__(self.L).tensor
        return ncon((T_BB, T_BB, self.rB.tensor, self.rB.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6)))

    def _compute_derivative_rho_A_rho_B(self) -> npt.NDArray[np.complex128]:

        A, B, L, U1, U2 = self.A, self.B, self.L, self.U1, self.U2
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
        res += self._contract_rhoA_rhoB_central_derivative(D, T_AB, T_BA)
        res += self._contract_rhoA_rhoB_fixed_point_derivative(T_AB, T_BA)

        return res

    def _compute_derivative_rho_B_rho_B(self) -> npt.NDArray[np.complex128]:
        
        res = np.zeros_like(self.B.tensor, dtype=np.complex128)

        BBdag = TransferMatrix.new(self.B, self.B)

        rB = self.rB

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
        
    def _contract_rhoA_rhoB_central_derivative(self, D, T_AB, T_BA):

        L = self.L

        if self.trotterization_order == 1:

            if L % 2 == 0:
                tensors = (self.left_auxillary, D, T_BA.tensor, self.right_auxillary, self.rA.tensor, self.rB.tensor)
                indices = ((1, 2, 3, 4), (3, 4, 5, 6, 7, 8, -1, -2, -3), (5, 2, 1, 9, 10, 11), (10, 11, 7, 12), (6, 12), (9, 8))
                return ncon(tensors, indices)

            tensors = [self.left_auxillary, T_AB.tensor, T_BA.tensor, self.rB.tensor, B.tensor, self.right_auxillary]
            indices = [[1, 2, 3, 4], [3, 4, 5, 6, 7, -1], [5, 2, 1, 8, 9, 10], [11, -3], [8, 12, 11], [6, 7, -2, 12, 9, 10]]
            tmp = ncon(tensors, indices)

            tensors = [self.left_auxillary, D, T_BA.tensor, self.B.conj, self.rB.tensor, B.tensor, self.right_auxillary]
            indices = [[1, 2, 3, 4], [3, 4, 5, 6, 7, 8, -1, -2, -3], [5, 2, 1, 9, 10, 11], [8, 12, 13], [14, 13], [9, 15, 14], [6, 7, 12, 15, 10, 11]]
            return ncon(tensors, indices) + tmp

        elif self.trotterization_order == 2:
            
            tensors = [self.left_auxillary, D, T_BA.tensor, self.rB.tensor, self.right_auxillary]
            indices = [
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 7, 15, 14, 13, 9, -1, -2, -3],
                [7, 3, 2, 1, 8, 12, 11, 10],
                [8, 9],
                [10, 11, 12, 13, 14, 15]
            ]
            order = [8, 1, 2, 3, 10, 11, 12, 6, 7, 9, 15, 4, 5, 13, 14]
            return ncon(tensors, indices, order=order, )

    def _contract_rhoA_rhoB_fixed_point_derivative(self, T_AB, T_BA):

        L = self.L

        if self.trotterization_order == 1:
            if L % 2 == 0:        
                tensors = (self.left_auxillary, T_AB.tensor, T_BA.tensor, self.right_auxillary, self.rA.tensor)
                indices = ((1, 2, 3, 4), (3, 4, 5, 6, 7, -1), (5, 2, 1, -2, 8, 9), (8, 9, 7, 10), (6, 10))
            else: 
                B = self.B
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
        return self.rB.derivative(v)
    
if __name__ == "__main__":
    
    Da = 5
    Db = 5
    p = 2

    A = UniformMps.new(Da, p)
    B = UniformMps.new(Db, p)

    from qdmt.model import TransverseFieldIsing

    tfim = TransverseFieldIsing(0.1, 0.1)

    f = EvolvedHilbertSchmidt(A, B, tfim, L=10)
    f.cost()
    f.derivative()

    # assert cost 0
    # assert np.allclose(cost(A, A, U, U, 4), 0j)
    # assert np.allclose(cost(B, B, U, U, 10), 0j)

    # pA = A.reduced_density_mat(3)
    # pB = B.reduced_density_mat(3)

    # assert np.allclose(ncon((pA, pA), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))) + ncon((pB, pB), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))) - 2*ncon((pA, pB), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))), cost(A, B, 3))

    # ABdag, BAdag = transfer_blocks(A, B)

    # _A = A.gauge()

    # assert not np.allclose(A.value, _A.value)
    # assert np.allclose(_A.r, ncon((_A.transfer_matrix.value, _A.r), ((-1, -2, 1, 2), (1, 2))))

    # check under unitary gauge transform the cost is zero
    # assert np.allclose(cost(A, _A, 3), 0j)
    # assert np.allclose(cost(A, _A, 10), 0j)

    # # test fidelity
    # L = 50
    # r = A.fidelity(B)

    # print(r**(2*L))
    # print(cost(A, B, L))

    # D, p = 5, 2
    # SEED = 42
    # np.random.seed(SEED)
    # A = UniformMps.new(D, p)
    # B = UniformMps.new(D, p)
    # f = HilbertSchmidt(A, B, 4)
    # print(f.cost())
    # D = f.derivative()


    # print(D)
