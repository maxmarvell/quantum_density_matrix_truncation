from qdmt.cost import AbstractCostFunction
from qdmt.model import AbstractModel
from qdmt.uniform_mps import UniformMps
from qdmt.fixed_point import RightFixedPoint
from qdmt.transfer_matrix import TransferMatrix
from qdmt.utils.mps import trotter_step
import numpy as np
import numpy.typing as npt
from ncon import ncon

class EvolvedHilbertSchmidt(AbstractCostFunction):
    def __init__(self, A: UniformMps, B: UniformMps, model: AbstractModel, L: int, order: int = 1):
        super().__init__(A, B, L)

        self.order = order

        if order == 1:
            self.U1, self.U2 = model.trotter_first_order()
        elif order == 2:
            self.U1, self.U2 = model.trotter_second_order()

        # construct rho(A(0); dt)
        self.rhoA = self.compute_rhoA()

        # construct rho(B)
        self.rhoB = self.compute_rhoB()
    
    @AbstractCostFunction.B.setter
    def B(self, value):
        self._B = value
        BBdag = TransferMatrix.new(value, value)
        self.rB = RightFixedPoint(BBdag)
        self.rhoB = self.compute_rhoB()

    def compute_rhoA(self) -> npt.NDArray[np.complex128]:

        A, U1, U2, rA, L = self.A, self.U1, self.U2, self.rA, self.L

        N = 0
        if L % 2 == 0 and self.order == 1:
            N = L + 2
        elif self.order == 1:
            N = L + 3
        elif L % 2 == 0 and self.order == 2:
            N = L + 4

        mps = A.mps_chain(N)

        # first trotter step
        mps = trotter_step(mps, U1)

        # second trotter step
        mps = trotter_step(mps, U2, start=1)

        # conditionally apply final trotter step
        if self.order == 2:
            mps = trotter_step(mps, U1)

        tensors = [mps, mps.conj(), rA.tensor]

        if self.order == 1: 
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
    
    def compute_rhoB(self) -> npt.NDArray[np.complex128]:

        B, rB, L = self.B, self.rB, self.L

        chain = B.mps_chain(self.L)

        tensors = (chain, chain.conj(), rB.tensor)
        indices = [
            [1] + [-i for i in range(1, L+1)] + [2],
            [1] + [-i for i in range(L+1, 2*L+1)] + [3],
            [2, 3]
        ]

        rhoB = ncon(tensors, indices)
        return rhoB
        
    def compute_rhoArhoB(self) -> np.complex128:

        rhoA, rhoB, L = self.rhoA, self.rhoB, self.L

        tensors = (rhoA, rhoB)
        indices = [
            [i for i in range(1, 2*L+1)],
            [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)]
        ]
        rhoArhoB = ncon(tensors, indices)

        return rhoArhoB
    
    def compute_rhoArhoA(self) -> np.complex128:

        rhoA, L = self.rhoA, self.L

        tensors = (rhoA, rhoA)
        indices = [
            [i for i in range(1, 2*L+1)],
            [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)]
        ]
        rhoArhoA = ncon(tensors, indices)

        return rhoArhoA
    
    def compute_rhoBrhoB(self) -> np.complex128:

        rhoB, L = self.rhoB, self.L

        tensors = (rhoB, rhoB)
        indices = [
            [i for i in range(1, 2*L+1)],
            [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)]
        ]
        res = ncon(tensors, indices)

        return res
    
    def cost(self) -> np.complex128:
        res = 0. + 0.j
        res += self.compute_rhoArhoA()
        res += self.compute_rhoBrhoB()
        res -= 2*self.compute_rhoArhoB()
        return res
    
    def compute_drhoB(self, i: int) -> npt.NDArray[np.complex128]:

        B, rB, L = self.B, self.rB, self.L

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
    
    def compute_drhoBrhoBdrB(self) -> npt.NDArray[np.complex128]:
        B, rhoB, L = self.B, self.rhoB, self.L

        chain = B.mps_chain(self.L)

        tensors = (chain, chain.conj())
        indices = [
            [1] + [-i for i in range(1, L+1)] + [-(2*L+1)],
            [1] + [-i for i in range(L+1, 2*L+1)] + [-(2*L+2)]
        ]
        drhoBdrB = ncon(tensors, indices)

        tensors = [drhoBdrB, rhoB]
        indices = [
            [i for i in range(1, 2*L+1)] + [-1, -2],
            [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)]
        ]
        drhoBrhoBdrB = ncon(tensors, indices)

        return drhoBrhoBdrB
    
    def compute_drhoBrhoAdrB(self) -> npt.NDArray[np.complex128]:
        B, rhoA, L = self.B, self.rhoA, self.L

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

    def derivative(self) -> npt.NDArray[np.complex128]:
        B, rhoA, rhoB, L, rB = self.B, self.rhoA, self.rhoB, self.L, self.rB

        D = np.zeros_like(B.tensor)

        # rho(A(t+dt))rho(B) contribution
        tmp = np.zeros_like(B.tensor)
        for i in range(L):
            drhoB = self.compute_drhoB(i)

            tensors = [rhoA, drhoB]
            indices = [
                [i for i in range(1, 2*L+1)],
                [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)] + [-1, -2, -3]
            ]
            tmp += ncon(tensors, indices)

        D -= 2*tmp

        # rho(B)^2 contribution
        tmp = np.zeros_like(B.tensor)
        for i in range(L):
            drhoB = self.compute_drhoB(i)

            tensors = [rhoB, drhoB]
            indices = [
                [i for i in range(1, 2*L+1)],
                [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)] + [-1, -2, -3]
            ]
            tmp += ncon(tensors, indices)
 
        D += 2*tmp

        drhoBrhoBdrB = self.compute_drhoBrhoBdrB()
        D += 2*rB.derivative(drhoBrhoBdrB.T)

        drhoBrhoAdrB = self.compute_drhoBrhoAdrB()
        D -= 2*rB.derivative(drhoBrhoAdrB.T)

        return D
