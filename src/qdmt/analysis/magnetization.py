import numpy as np
from ncon import ncon
from scipy.integrate import quad

from qdmt.uniform_mps import UniformMps
from qdmt.model import Pauli
from qdmt.transfer_matrix import TransferMatrix

def transverse_magnetization(A: UniformMps) -> np.float64:
    r = TransferMatrix.new(A, A).right_fixed_point()
    return np.real(ncon([A.tensor, Pauli.Sx,  A.tensor.conj(), r.tensor], [[1, 2, 3], [2, 4], [1, 4, 5], [3, 5]]))

def longitudinal_magnetization(A: UniformMps) -> np.float64:
    r = TransferMatrix.new(A, A).right_fixed_point()
    return np.real(ncon([A.tensor, Pauli.Sz,  A.tensor.conj(), r.tensor], [[1, 2, 3], [2, 4], [1, 4, 5], [3, 5]]))

def analytic(t, g0, g1):
    def theta(k, g):
        return 0.5 * np.arctan2(np.sin(k), g - np.cos(k))

    def delta(k):
        return theta(k, g1) - theta(k, g0)

    def epsilon(k):
        return 2 * np.sqrt((g1 - np.cos(k))**2 + np.sin(k)**2)

    def integrand(k):
        th1 = theta(k, g1)
        d = delta(k)
        ek = epsilon(k)
        return (
            np.cos(2 * th1) * np.cos(2 * d) +
            np.sin(2 * th1) * np.sin(2 * d) * np.cos(2 * ek * t)
        ) / np.pi

    return quad(integrand, 0, np.pi)[0]