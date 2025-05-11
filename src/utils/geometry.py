from uniform_MPS import UniformMPS
from ncon import ncon
import numpy as np
from scipy.linalg import expm
from cost import AbstractCostFunction
from fixed_point import RightFixedPoint

def project_tangent_grassman(B: UniformMPS, D: np.ndarray):
    return D - ncon((B.tensor, B.conj, D), ((-1, -2, 3), (1, 2, 3), (1, 2, -3)))

def project_tangent_stiefel(B: UniformMPS, D: np.ndarray):
    d, p = B.d, B.p
    A = B.matrix
    D = D.reshape((d*p, d))
    res = D - 0.5 * A @ (np.conj(A).T @ D + np.conj(D).T @ A)
    return res.reshape((d, p, d))

def preconditioning(G: np.ndarray, r: RightFixedPoint):
    d, _ = r.shape
    delta = np.linalg.norm(G.reshape((-1, d)))**2
    Rinv = np.linalg.inv(r.tensor + np.eye(d)*delta)
    return G @ Rinv

def grassman_retraction(B: UniformMPS, D: np.ndarray, alpha: float, r: RightFixedPoint):
    G = project_tangent_grassman(B, D)
    G = preconditioning(G, r)
    return B.tensor - alpha * G

def steifel_retraction(B: UniformMPS, D: np.ndarray, alpha: float, r: RightFixedPoint):
    G = project_tangent_stiefel(B, D)
    G = preconditioning(G, r)
    return B.tensor - alpha * G

def simple_retraction(W, G, alpha):
    """
        Perform a simple retraction on isometry W
    """
    return W - G * alpha

# def grassman_retraction(W, G, alpha):
#     '''
#         Peform a geodesic retraction on the Grassmann manifold based on the Euclidean metric.
#     '''
#     n, m = W.shape
#     Q, R = np.linalg.qr((np.eye(n) - W @ W.conj().T) @ G, mode='reduced')
#     Zero = np.zeros((m, m))

#     a = np.block([W, Q])
#     b = np.block([
#         [Zero, -alpha*R.conj().T],
#         [alpha*R, Zero]
#     ])
#     b = expm(b)[..., :m]
#     return a @ b
