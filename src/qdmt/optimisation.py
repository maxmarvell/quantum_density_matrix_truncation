from abc import ABC, abstractmethod
import numpy as np
from ncon import ncon

from qdmt.manifold import AbstractManifold
from qdmt.cost import AbstractCostFunction
from qdmt.uniform_mps import UniformMps
from qdmt.linesearch import linesearch

def preconditioning(G: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Apply MPS preconditioning to a gradient tensor G, as in M. Hauru arXiv:2007.03638v4  [quant-ph]  13 Jan 2021.

    Args:
        G: Gradient tensor to Grassmann / Steifel manifold, lives in tangent space with shape `(n, p)`
        r: Right fixed point of variational uMPS parametrized by B has shape `(p, p)`

    Returns:
        Precondtioned gradient tensor with shape `(n, p)`
    """
    d, _ = r.shape
    delta = np.linalg.norm(G)**2
    Rinv = np.linalg.inv(r + np.eye(d)*delta)
    return G @ Rinv

class AbstractOptimizer(ABC):

    def __init__(self, f: AbstractCostFunction, M: AbstractManifold):
        self.f = f
        self.M = M

    @abstractmethod
    def update(self, precondition: bool = True) -> tuple[UniformMps, float, float]:
        pass

    def optimize(self, max_iters: int, tol: float, verbose: bool = False, rtol: float = 1e-3) -> tuple[UniformMps, np.complex128]:
        print(f"Initial cost: {abs(self.f.cost())}")

        norm = tol
        cost = []
        gradient_norm = []

        for i in range(max_iters):

            new_B, tmp_cost, norm = self.update()

            # guarantee left canonical is preserved
            if not new_B.check_left_orthonormal():
                print("Warning: The uniform matrix product state is no longer left canonical, orthoganlizing...")
                new_B.left_orthorganlize()

            self.f.B = new_B

            cost.append(tmp_cost)
            gradient_norm.append(norm)

            if verbose and i % 100 == 0:
                print(f"Iteration {i}:")
                print(f"\tCost - {abs(tmp_cost)}")
                print(f"\tGradient Norm - {norm}")

            if i > 1:
                if abs(cost[i-1] - cost[i-2]) / abs(cost[i-2]) < rtol:
                    print("\nConverged, cost function not sufficiently decreasing!")
                    print(f"Iteration {i}:")
                    print(f"\tCost - {abs(tmp_cost)}")
                    print(f"\tGradient Norm - {norm}")
                    break

            if norm < tol:
                print("\nConverged!")
                print(f"Iteration {i}:")
                print(f"\tCost - {abs(tmp_cost)}")
                print(f"\tGradient Norm - {norm}")
                break

        return self.f.B, np.abs(self.f.cost()), np.abs(norm)

class GradientDescent(AbstractOptimizer):

    def __init__(self, f: AbstractCostFunction, M: AbstractManifold, alpha0: float = 1.0, c1: float = 1e-4, c2: float = 0.9):
        super().__init__(f, M)
        self.alpha0 = alpha0
        self.tmp_f = f.copy()
        self.c1 = c1
        self.c2 = c2

    def _retract(self, W: np.ndarray, X: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:

        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        V = Vh.conj().T

        # Geodesic formula from paper (2.65) and (3.1)
        W_prime = (W @ V) @ np.diag(np.cos(s * alpha)) @ Vh + U @ np.diag(np.sin(s * alpha)) @ Vh

        Q_X = X @ W.conj().T - W @ X.conj().T
        X_prime = Q_X @ W_prime

        return W_prime, X_prime
    
    def update(self, precondition: bool = True) -> tuple[UniformMps, float, float]:

        # Gradient and cost at current point
        d, p = self.f.B.d, self.f.B.p
        W = self.f.B.isometry
        C, D = self.f.cost(), self.f.derivative().reshape(d*p, d)

        # Project the gradient onto tangent space and precondition if specified
        G = self.M.project(W, D)
        if precondition:
            G = preconditioning(G, self.f.rB.tensor)

        # Descent direction is steepest gradient
        X = -G

        # --- Try a linesearch if fails then fallback to alpha = 0.1 ---
        alpha0, W_prime, C_prime, _, converged = linesearch(self.tmp_f, self.M, W, X, C, G, self._retract, alpha0=self.alpha0)

        if converged:
            self.alpha0 = alpha0
            return UniformMps(W_prime), C_prime, np.linalg.norm(G)
        
        else:
            print("\nLinesearch failed to converge - defaulting to step size of 0.1")
            W_prime = self._retract(W, X, 0.1)
            return UniformMps(W_prime), C, np.linalg.norm(G)

if __name__ == "__main__":

    d = 4
    p = 2

    A = UniformMps.new(d, p)
    
    from qdmt.model import TransverseFieldIsing
    from qdmt.cost import EvolvedHilbertSchmidt
    from qdmt.manifold import Grassmann

    tfim = TransverseFieldIsing(0.1, 0.1)
    M = Grassmann()
    f = EvolvedHilbertSchmidt(A, A, tfim, 4, trotterization_order=2)
    opt = GradientDescent(f, M)
    opt.optimize(1000, 1e-8, verbose=True)
    print(np.allclose(ncon((opt.f.B.conj, opt.f.B.tensor), ((1, 2, -1), (1, 2, -2))), np.eye(d, dtype=np.complex128), rtol=1e-12))

