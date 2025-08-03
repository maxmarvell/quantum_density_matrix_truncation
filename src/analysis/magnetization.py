import numpy as np
from ncon import ncon

from qdmt.uniform_mps import UniformMps
from qdmt.model import Pauli
from qdmt.fixed_point import RightFixedPoint

def transverse_magnetization(A: UniformMps) -> np.float64:
    r = RightFixedPoint.from_mps(A)
    return np.real(ncon([A.tensor, Pauli.Sx,  A.conj, r.tensor], [[1, 2, 3], [2, 4], [1, 4, 5], [3, 5]]))

def longitudinal_magnetization(A: UniformMps) -> np.float64:
    r = RightFixedPoint.from_mps(A)
    return np.real(ncon([A.tensor, Pauli.Sz,  A.conj, r.tensor], [[1, 2, 3], [2, 4], [1, 4, 5], [3, 5]]))