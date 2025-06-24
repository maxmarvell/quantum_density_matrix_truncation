from abc import ABC, abstractmethod

from ncon import ncon
import numpy as np
from uniform_MPS import UniformMPS
from transfer_matrix import *
from fixed_point import RightFixedPoint

class AbstractCostFunction(ABC):

    def __init__(self, A: UniformMPS, B: UniformMPS, L: int):
        self.A = A
        self.B = B
        self.L = L

        AAdag = TransferMatrix.new(self.A, self.A)
        self.rA = RightFixedPoint(AAdag)

        self.rB = None

    
    @abstractmethod
    def cost(self):
        pass

    @abstractmethod
    def derivative(self):
        pass

class HilbertSchmidt(AbstractCostFunction):

    def cost(self):

        ABdag, BAdag = transfer_blocks(self.A, self.B)
        AAdag = TransferMatrix.new(self.A, self.A)
        BBdag = TransferMatrix.new(self.B, self.B)

        Da, Db = self.A.d, self.B.d

        if self.rB is None:
            self.rB = RightFixedPoint(BBdag)

        res = 0

        # rho(B)^2 contribution
        T_BB = (BBdag ** self.L).tensor
        res += ncon((T_BB, T_BB, self.rB.tensor, self.rB.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6))) / (np.sqrt(Db) * np.sqrt(Db))

        # rho(A)^2 contribution
        T_AA = (AAdag ** self.L).tensor
        res += ncon((T_AA, T_AA, self.rA.tensor, self.rA.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6))) / (np.sqrt(Da) * np.sqrt(Da))

        # 2rho(A)rho(B) contribution
        T_AB = (ABdag ** self.L).tensor
        T_BA = (BAdag ** self.L).tensor
        res -= 2 * ncon((T_AB, T_BA, self.rB.tensor, self.rA.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6))) / (np.sqrt(Da) * np.sqrt(Db))

        return res
    
    def derivative(self):

        ABdag, BAdag = transfer_blocks(self.A, self.B)
        BBdag = TransferMatrix.new(self.B, self.B)

        Da, Db = self.A.d, self.B.d

        if self.rB is None:
            self.rB = RightFixedPoint(BBdag)

        res = np.zeros_like(self.B.tensor, complex)

        # rho(B)^2 contribution
        T_BB = (BBdag ** self.L)
        D = T_BB.derivative()
        res += 2 * ncon((D, T_BB.tensor, self.rB.tensor, self.rB.tensor), ((1, 2, 3, 4, -1, -2, -3), (2, 1, 5, 6), (5, 4), (3, 6))) / (np.sqrt(Db) * np.sqrt(Db))

        # right fixed point
        tensors = (T_BB.tensor, T_BB.tensor, self.rB.tensor)
        indices = ((1, 2, 3, -1), (2, 1, -2, 4), (3, 4))
        v = ncon(tensors, indices)
        res += 2 * self.rB.derivative(v) / (np.sqrt(Db) * np.sqrt(Db))

        # 2rho(A)rho(B) contribution
        T_AB = (ABdag ** self.L)
        T_BA = (BAdag ** self.L)
        D = T_AB.derivative()
        res -= 2 * ncon((D, T_BA.tensor, self.rB.tensor, self.rA.tensor), ((1, 2, 3, 4, -1, -2, -3), (2, 1, 5, 6), (5, 4), (3, 6))) / (np.sqrt(Da) * np.sqrt(Db))

        # right fixed point
        tensors = (T_AB.tensor, T_BA.tensor, self.rA.tensor)
        indices = ((1, 2, 3, -1), (2, 1, -2, 4), (3, 4))
        v = ncon(tensors, indices)
        res -= 2 * self.rB.derivative(v) / (np.sqrt(Da) * np.sqrt(Db))

        return res

class EvolvedHilbertSchmidt(AbstractCostFunction):

    def __init__(self, A: UniformMPS, B: UniformMPS, U1: np.ndarray, U2: np.ndarray, L: int):

        if L % 2 != 0:
            return NotImplemented
        
        super().__init__(A, B, L)
        self.U1 = U1
        self.U2 = U2

        tensors = (A.tensor, A.tensor, self.U1)
        indices = ((-1, 1, 2), (2, 3, -3), (1, 3, -2, -4))
        self.left_auxillary = ncon(tensors, indices)

        tensors = (np.conj(self.U1), A.conj, A.conj)
        indices = ((1, 2, -1, -3), (-2, 1, 3), (3, 2, -4))
        self.right_auxillary = ncon(tensors, indices)

        # calculate left dangling tensor
        tensors = (A.tensor, A.tensor, U1, np.conj(U1), A.conj, A.conj)
        indices = ((1, 2, 3), (3, 4, -1), (2, 4, 5, -2), (6, 7, 5, -3), (1, 6, 8), (8, 7, -4))
        tmp = ncon(tensors, indices)

        tensors = (tmp, tmp)
        indices = ((-1, 1, 2, -4), (-3, 2, 1, -2))
        tmp_left = ncon(tensors, indices)

        # calculate right dangling tensor
        tensors = (A.tensor, A.tensor, U1, np.conj(U1), A.conj, A.conj, self.rA.tensor)
        indices = ((-1, 1, 2), (2, 3, 4), (1, 3, -2, 5), (6, 7, -3, 5), (-4, 6, 8), (8, 7, 9), (4, 9))
        tmp = ncon(tensors, indices)

        tensors = (tmp, tmp)
        indices = ((-1, 1, 2, -4), (-3, 2, 1, -2))
        tmp_right = ncon(tensors, indices)

        AAdag = TransferMatrix.new(A, A)
        T_AA = (AAdag ** (L-2)).tensor
        tensors = (tmp_left, T_AA, T_AA, tmp_right)
        indices = ((1, 2, 3, 4), (1, 2, 5, 6), (3, 4, 7, 8), (5, 6, 7, 8))
        self.costAA = ncon(tensors, indices) / (np.sqrt(A.d) * np.sqrt(A.d))

    def cost(self):
        A, B = self.A, self.B
        Da, Db = A.d, B.d

        ABdag = FirstOrderTrotterizedTransferMatrix.new(A, B, self.U1, self.U2)
        BAdag = FirstOrderTrotterizedTransferMatrix.new(B, A, np.conj(self.U2).transpose(2, 3, 0, 1), np.conj(self.U1).transpose(2, 3, 0, 1))
        BBdag = TransferMatrix.new(self.B, self.B)

        if self.rB is None:
            self.rB = RightFixedPoint(BBdag)

        res = 0

        # rho(B)^2 contribution
        T_BB = (BBdag ** self.L).tensor
        res += ncon((T_BB, T_BB, self.rB.tensor, self.rB.tensor), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6))) / (np.sqrt(Db) * np.sqrt(Db))

        # rho(A(t+dt))^2 contribution
        res += self.costAA

        # rho(A(t+dt))rho(B) contribution
        T_AB = (ABdag ** (self.L/2)).tensor
        T_BA = (BAdag ** (self.L/2)).tensor
        tensors = (self.left_auxillary, T_AB, T_BA, self.right_auxillary, self.rA.tensor, self.rB.tensor)
        indices = ((1, 2, 3, 4), (3, 4, 5, 6, 7, 8), (5, 2, 1, 9, 10, 11), (10, 11, 7, 12), (6, 12), (9, 8))
        res -= 2 * ncon(tensors, indices) / (np.sqrt(Da) * np.sqrt(Db))

        return res

    def derivative(self):
        A, B = self.A, self.B
        Da, Db = A.d, B.d

        ABdag = FirstOrderTrotterizedTransferMatrix.new(A, B, self.U1, self.U2)
        BAdag = FirstOrderTrotterizedTransferMatrix.new(B, A, np.conj(self.U2).transpose(2, 3, 0, 1), np.conj(self.U1).transpose(2, 3, 0, 1))
        BBdag = TransferMatrix.new(self.B, self.B)

        if self.rB is None:
            self.rB = RightFixedPoint(BBdag)

        rB = self.rB

        T_BB = BBdag ** self.L
        T_BA = BAdag ** (self.L / 2)
        T_AB = ABdag ** (self.L / 2)

        res = np.zeros_like(B.tensor)

        # rho(B)^2 contribution
        D = T_BB.derivative()
        tensors = (D, T_BB.tensor, rB.tensor, rB.tensor)
        indices = ((1, 2, 3, 4, -1, -2, -3), (2, 1, 5, 6), (3, 6), (4, 5))
        res += 2 * ncon(tensors, indices) / (np.sqrt(Db) * np.sqrt(Db))

        # right fixed point
        tensors = (T_BB.tensor, T_BB.tensor, rB.tensor)
        indices = ((1, 2, 3, -1), (2, 1, -2, 4), (3, 4))
        v = ncon(tensors, indices)
        res += 2 * rB.derivative(v) / (np.sqrt(Db) * np.sqrt(Db))

        # rho(A(t+dt))rho(B) contribution
        D = T_AB.derivative()
        tensors = (self.left_auxillary, D, T_BA.tensor, self.right_auxillary, self.rA.tensor, self.rB.tensor)
        indices = ((1, 2, 3, 4), (3, 4, 5, 6, 7, 8, -1, -2, -3), (5, 2, 1, 9, 10, 11), (10, 11, 7, 12), (6, 12), (9, 8))
        res -= 2 * ncon(tensors, indices) / (np.sqrt(Da) * np.sqrt(Db))

        # right fixed point
        tensors = (self.left_auxillary, T_AB.tensor, T_BA.tensor, self.right_auxillary, self.rA.tensor)
        indices = ((1, 2, 3, 4), (3, 4, 5, 6, 7, -1), (5, 2, 1, -2, 8, 9), (8, 9, 7, 10), (6, 10))
        v = ncon(tensors, indices)
        res -= 2 * rB.derivative(v) / (np.sqrt(Da) * np.sqrt(Db))

        return res

def transfer_blocks(A: UniformMPS, B: UniformMPS):
    '''
        Compute two transfer matrices

        _AB : 1--A--3
                 |
              2--B*-4

        A_B : 1--B--3
                 |
              2--A*-4

    '''
    return TransferMatrix.new(A, B), TransferMatrix.new(B, A)

def compute_contraction(ABdag, BAdag, rA, rB, L):
    return ncon(((ABdag ** L).value , (BAdag ** L).value , rB, rA), ((1, 2, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6)))

if __name__ == "__main__":
    
    Da = 5
    Db = 3
    p = 2

    A = UniformMPS.from_random(Da, p)
    B = UniformMPS.from_random(Db, p)

    from utils.unitary import transverse_ising
    U = transverse_ising(0.1)

    assert np.allclose(ncon((U, np.conj(U)), ((-1, -2, 1, 2), (-3, -4, 1, 2))).reshape(4, 4), np.eye(4))

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

    D, p = 5, 2
    SEED = 42
    np.random.seed(SEED)
    A = UniformMPS.from_random(D, p)
    B = UniformMPS.from_random(D, p)
    f = HilbertSchmidt(A, B, 4)
    print(f.cost())
    D = f.derivative()


    print(D)
