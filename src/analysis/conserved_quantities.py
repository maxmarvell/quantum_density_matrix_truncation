import numpy as np
from ncon import ncon

from qdmt.uniform_mps import UniformMps
from qdmt.fixed_point import RightFixedPoint
from qdmt.model import AbstractModel

def compute_energy(A: UniformMps, model: AbstractModel) -> np.float64:
    r = RightFixedPoint.from_mps(A)
    tensors = [A.tensor, A.tensor, model.H,  A.conj, A.conj, r.tensor]
    indices = [[1, 2, 3], [3, 4, 5], [2, 4, 6, 7], [1, 6, 8], [8, 7, 9], [5, 9]]
    return np.real(ncon(tensors, indices))

def compute_normalization(A: UniformMps):
    return A.normalization()

