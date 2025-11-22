from abc import ABC, abstractmethod
import numpy as np
from ncon import ncon
import sys

from qdmt.manifold import AbstractManifold, euclidian_metric as inner
from qdmt.cost import AbstractCostFunction
from qdmt.uniform_mps import UniformMps
from qdmt.linesearch import linesearch
from qdmt.transfer_matrix import RightFixedPoint, TransferMatrix

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

    def fg(self, B: UniformMps, rB: RightFixedPoint):
        C = self.f.cost(B, rB)
        D = self.f.derivative(B, rB).reshape(B.m, B.n)
        return C, self.M.project(B.V, D)

    @abstractmethod
    def optimize(self) -> tuple[UniformMps, np.float64, np.float64]:
        pass

class GradientDescent(AbstractOptimizer):

    alpha0 = 1.0
    c1 = 1.0
    c2 = 1e-1

    def __init__(self, f: AbstractCostFunction, M: AbstractManifold, B0: UniformMps, max_iter: int, tol: float = 1e-8, precondition: bool = True, verbose: bool = True):
        super().__init__(f, M)
        self.tmp_f = f.copy()
        self.B0 = B0
        self.rB0 = RightFixedPoint.from_mps(B0)
        self.precondtion = precondition
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def optimize(self):
        W = self.B0.matrix
        C, G = self.fg(self.B0)
        cost_history = [C]

        norm_grad = np.sqrt(inner(G, G))
        grad_history = [norm_grad]

        if self.precondtion:
            G_tilde = preconditioning(G, self.rB0.tensor)
        else:
            G_tilde = G

        alpha = 1 / (np.sqrt(inner(G_tilde, G_tilde)))

        if self.verbose:
            print(f"GD: initializing with f = {C:.8e}, ‖∇f‖ = {norm_grad:.4e}")

        # --- Main optimization loop ---
        iter = 0
        while True:
            if self.precondtion:
                u_mps = UniformMps(W)
                rB = RightFixedPoint.from_mps(u_mps)
                G_tilde = preconditioning(G, rB.tensor)
            else:
                G_tilde = G

            # initial search direction is negative gradient
            X = -G_tilde

            alpha, W, C, G, success = linesearch(self.tmp_f, self.M, W, X, C, G, alpha0=alpha)

            if not success:
                alpha = min(alpha, 0.1)
                W, _ = self.M.retract(W, X, alpha)
                B = UniformMps(W)
                rB = TransferMatrix.new(B, B).right_fixed_point()
                C, G = self.fg(B, rB)
            else:
                alpha = min(alpha*1.1, 1)
                B = UniformMps(W)
                rB = TransferMatrix.new(B, B).right_fixed_point()

            if not B.is_isometry():
                print("Warning: Not left orthonormal left orthonormalizing!")
                B = B.orthonormalize()
                W = B.V

            iter += 1

            cost_history.append(C)
            norm_grad = np.sqrt(inner(G, G))
            grad_history.append(norm_grad)

            if norm_grad <= self.tol or iter >= self.max_iter:
                break

            if self.verbose == True:
                print(f"GD: iter {iter:4d}: f = {C:.8e}, ‖∇f‖ = {norm_grad:.4e}, α = {alpha:.2e}")

            alpha *= 2   

        history = [cost_history, grad_history]
        return W, C, G, history

    
    def update(self, precondition: bool = True) -> tuple[UniformMps, float, float]:

        # Gradient and cost at current point
        d, p = self.f.B.d, self.f.B.p
        W = self.f.B.isometry
        C, D = self.f.cost(), self.f.derivative().reshape(d*p, d)

        # Project the gradient onto tangent space and precondition if specified
        G = self.M.project(W, D)
        if precondition:
            G = preconditioning(G, self.f.rB.tensor)

        if self.alpha0 == None:
            self.alpha0 = 1 / np.linalg.norm(G) ** 2

        # Descent direction is steepest gradient
        X = -G

        # --- Try a linesearch if fails then fallback to alpha = 0.1 ---
        alpha0, W_prime, C_prime, _, converged = linesearch(self.tmp_f, self.M, W, X, C, G, self._retract, alpha0=2*self.alpha0)

        if converged:
            self.alpha0 = alpha0
            print(f"Converged alpha: {alpha0}")
            return UniformMps(W_prime), C_prime, np.linalg.norm(G)
        
        else:
            print("\nLinesearch failed to converge - defaulting to step size of 0.1")
            W_prime, _ = self._retract(W, X, 0.1)
            return UniformMps(W_prime), C, np.linalg.norm(G)
        
class ConjugateGradient(AbstractOptimizer):

    THETA = 1.0
    ETA = 0.4

    def __init__(self, f: AbstractCostFunction, M: AbstractManifold, B0: UniformMps, max_iter: int, tol: float = 1e-8, restart: int = 100,  precondition: bool = True, verbose: bool = False,):
        super().__init__(f, M)
        self.B0 = B0
        self.rB0 = TransferMatrix.new(B0, B0).right_fixed_point()
        self.precondtion = precondition
        self.restart = restart
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.tmp_f = f.copy()
    
    def _Hager_Zhang(self, g, gprev, Pg, Pgprev, dprev):
        dd = inner(dprev, dprev)
        dg = inner(dprev, g)
        dgprev = inner(dprev, gprev)
        gPg = inner(g, Pg)
        gprevPgprev = inner(gprev, Pgprev)
        gPgprev = inner(g, Pgprev)
        gprevPg = inner(gprev, Pg) # should probably be the same as gprevPg

        dy = dg - dgprev
        gPy = gPg - gPgprev
        yPy = gPg + gprevPgprev - gPgprev - gprevPg

        beta = (gPy - self.THETA*(yPy/dy)*dg)/dy

        eta = self.ETA*dgprev/dd
        if beta < eta and self.verbose:
            print("Warning: Resorting to default eta")
        return max(beta, eta)

    def optimize(self):

        W = self.B0.V
        rB = TransferMatrix.new(self.B0, self.B0).right_fixed_point()
        C, G = self.fg(self.B0, rB)

        # print(self.B0)
        # print("C coming")
        # print(C)
        # sys.exit()


        cost_history = [C]

        norm_grad = np.sqrt(inner(G, G))
        grad_history = [norm_grad]

        if self.precondtion:
            G_tilde = preconditioning(G, self.rB0.tensor)
        else:
            G_tilde = G

        alpha = 1 / (np.sqrt(inner(G_tilde, G_tilde)))

        if self.verbose:
            print(f"CG: initializing with f = {C:.8e}, ‖∇f‖ = {norm_grad:.4e}")

        # --- Main optimization loop ---
        iter = 0
        flag = False
        while True:
            # print(iter)
            if self.precondtion:
                G_tilde = preconditioning(G, rB.tensor)
            else:
                G_tilde = G

            # initial search direction is negative gradient
            X = -G_tilde

            # use momentum if not at restart
            if iter % self.restart == 0:
                beta = 0
            else:
                beta = self._Hager_Zhang(G, G_prev, G_tilde, G_tilde_prev, X_prev)
                X = X + X_prev * beta

            # store current quantities as previous quantities
            W_prev = W
            X_prev = X
            G_prev = G
            C_prev = C
            G_tilde_prev = G_tilde


            # print(f"iter={iter} pre line search")
            # UniformMps(W).is_isometry()
            
            alpha, W, C, G, success = linesearch(self.fg, self.M, W, X, C, G, alpha0=alpha)
            # print(f"iter={iter} after line search")
            # UniformMps(W).is_isometry()

            if not success:
                alpha = min(alpha, 0.1)
                W, _ = self.M.retract(W, X, alpha)
                B = UniformMps(W)
                rB = TransferMatrix.new(B, B).right_fixed_point()
                C, G = self.fg(B, rB)
            else:
                alpha = min(alpha*1.1, 1)
                B = UniformMps(W)
                rB = TransferMatrix.new(B, B).right_fixed_point()

            if not B.is_isometry():
                print("Warning: Not left orthonormal left orthonormalizing!")
                B = B.orthonormalize()
                W = B.V
         
            # if not B.is_isometry():
            #     C, G = self.fg(B, rB)
            #     print("Warning: Not left orthonormal left orthonormalizing!")
            #     print(f" BEFORE CG: iter {iter:4d}: f = {C:.8e}, ‖∇f‖ = {norm_grad:.4e}, α = {alpha:.2e}, β = {beta:.2e}")
            #     flag = True
            #     # sys.exit()
            #     B = B.orthonormalize()
            #     W = B.V
            #     B = UniformMps(W)
            #     # added these two lines for suspected bug
            #     rB = TransferMatrix.new(B, B).right_fixed_point()
            #     C, G = self.fg(B, rB)
            #     print(f" after CG: iter {iter:4d}: f = {C:.8e}, ‖∇f‖ = {norm_grad:.4e}, α = {alpha:.2e}, β = {beta:.2e}")


            # if flag and iter > 600:
            #     sys.exit()
            iter += 1

            cost_history.append(C)
            norm_grad = np.sqrt(inner(G, G))
            grad_history.append(norm_grad)

            if norm_grad <= self.tol or iter >= self.max_iter or np.abs(C)<=2*1e-16:
                break

            if self.verbose == True and iter % 10 == 0:
                print(f"CG: iter {iter:4d}: f = {C:.8e}, ‖∇f‖ = {norm_grad:.4e}, α = {alpha:.2e}, β = {beta:.2e}")

            # print("some transport")
            G_prev = self.M.transport(G_prev, W_prev, X_prev, alpha, W)
            if self.precondtion:
                G_tilde_prev = self.M.transport(G_tilde_prev, W_prev, X_prev, alpha, W)
            else:
                G_tilde_prev = G_prev
            X_prev = self.M.transport(X_prev, W_prev, X_prev, alpha, W)  
            # print("loop done")    

        print(f"\nCG: Converged! iter {iter:4d}: f = {C:.8e}, ‖∇f‖ = {norm_grad:.4e}, α = {alpha:.2e}, β = {beta:.2e}")

        history = [cost_history, grad_history]
        return UniformMps(W), C, norm_grad, history

if __name__ == "__main__":
    
    from qdmt.model import TransverseFieldIsing
    from qdmt.cost import EvolvedHilbertSchmidt
    from qdmt.manifold import Grassmann

    theta = phi = np.pi / 2

    psi = np.array([np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)])

    A = UniformMps(psi.reshape(1, 2, 1))
    B = UniformMps.new(4, 2)

    model = TransverseFieldIsing(g=1.05, delta_t=0.1, h=-0.5, J=-1)
    M = Grassmann()
    f = EvolvedHilbertSchmidt(A, model, 4, trotterization_order=2)
    opt = ConjugateGradient(f, M, B, max_iter=1000, verbose=True, tol=1e-6)
    A_opt, _, _, _ = opt.optimize()
    D, d, _ = A_opt.tensor.shape
    print(np.allclose(ncon((A_opt.tensor, A_opt.conj), ((1, 2, -1), (1, 2, -2))), np.eye(D, dtype=np.complex128), rtol=1e-12))

