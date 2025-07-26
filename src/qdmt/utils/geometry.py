from ncon import ncon
import numpy as np
from scipy.linalg import expm, polar
from abc import ABC, abstractmethod

def preconditioning(G: np.ndarray, r: np.ndarray) -> np.ndarray:
    d, _ = r.shape
    delta = np.linalg.norm(G.reshape((-1, d)))**2
    # print(delta)
    Rinv = np.linalg.inv(r + np.eye(d)*delta)
    return G @ Rinv

class AbstractProjector(ABC):
    @abstractmethod
    def project(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        pass

class IdentityProjector(AbstractProjector):
    def project(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        return D

class GrassmanProjector(AbstractProjector):
    def project(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        return D - ncon((A, np.conj(A), D), ((-1, -2, 3), (1, 2, 3), (1, 2, -3)))

class SteifelProjector(AbstractProjector):
    def project(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        d, p, _ = A.shape
        A = A.reshape((d*p, d))
        D = D.reshape((d*p, d))
        res = D - 0.5 * A @ (np.conj(A).T @ D + np.conj(D).T @ A)
        return res.reshape((d, p, d))
    
class AbstractUpdate(ABC):
    def __init__(self, projector: AbstractProjector, preconditioning: bool = True):
        self.projector = projector
        self.preconditioning = preconditioning

    @abstractmethod
    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> tuple[np.ndarray, np.float64]:
        pass

class GradientDescent(AbstractUpdate):
    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> tuple[np.ndarray, np.float64]:
        G = self.projector.project(A, D)
        if self.preconditioning:
            G = preconditioning(G, r)
        A_new = A - G * alpha
        d, p, _ = A_new.shape
        _A, _ = polar(A_new.reshape(d*p, d))
        return _A.reshape(d, p, d), np.linalg.norm(G)
    
class Retraction(AbstractUpdate):
    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> tuple[np.ndarray, np.float64]:

        d, p, _ = A.shape
        Y = A.reshape(d*p, d)
        # F_Y = D.reshape(d*p, d)
        
        G = self.projector.project(A, D)

        if self.preconditioning:
            G = preconditioning(G, r)

        U, s, Vh = np.linalg.svd(-G.reshape(d*p, d), full_matrices=False)
        V = Vh.conj().T

        t = alpha # Step size
        
        # Geodesic formula from paper (2.65) and (3.1)
        Y_new_mat = (Y @ V) @ np.diag(np.cos(s * t)) @ Vh + U @ np.diag(np.sin(s * t)) @ Vh
        
        return Y_new_mat.reshape(d, p, d), np.linalg.norm(G)

class TestRetraction(AbstractUpdate):
    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> np.ndarray:

        G = self.projector.project(A, D)
        d, _, _ = A.shape

        if self.preconditioning:
            G = -preconditioning(G, r)

        Zero = np.zeros((d, d))
        I = np.eye(d)

        a = np.block([A, alpha*G])
        GhG = ncon((np.conj(G), G), ((1, 2, -1), (1, 2, -2)))
        b = np.block([
            [Zero, -alpha**2 * GhG],
            [I, Zero]
        ])
        b = expm(b)[..., :d]

        return a @ b


class ConjugateGradient(AbstractUpdate):
    def __init__(self, projector: AbstractProjector, preconditioning: bool = True, retraction_instance: Retraction = None):
        super().__init__(projector, preconditioning)
        self._previous_gradient: np.ndarray | None = None
        self._previous_direction: np.ndarray | None = None
        self._iteration = 0 # To track iterations for CG restart/initialization
        
        if retraction_instance is None:
            self.retraction_instance = Retraction(projector, preconditioning=False) # Retraction should not re-precondition
        else:
            self.retraction_instance = retraction_instance
        
    def _riemannian_inner_product(self, X: np.ndarray, Y: np.ndarray) -> np.complex128:
        # g_W(X, Y) = Re(Tr[X^dagger Y])
        return np.sum(np.conj(X.reshape(8, 4)).T @ Y.reshape(8, 4)).real

    def _vector_transport(self, Y_old: np.ndarray, A_new_manifold_point: np.ndarray) -> np.ndarray:
        # Simple parallel transport by retraction: project Y_old onto the tangent space at A_new_manifold_point
        # The projector itself performs the necessary operation to get a valid tangent vector
        # by projecting from the ambient space.
        # Here, A_new_manifold_point is the base point for the tangent space.
        # Y_old is the vector from the old tangent space.
        return self.projector.project(A_new_manifold_point, Y_old)

    def update(self, A: np.ndarray, D: np.ndarray, alpha: float, r: np.ndarray = None) -> tuple[np.ndarray, np.float64]:
        
        G_k = self.projector.project(A, D) # Current Riemannian gradient
        
        if self.preconditioning:
            G_k_preconditioned = preconditioning(G_k, r)
        else:
            G_k_preconditioned = G_k

        # First iteration or restart
        if self._previous_gradient is None or self._iteration == 0:
            P_k = -G_k_preconditioned
        else:
            # Calculate beta_k using a Hager-Zhang like formula adapted for Riemannian setting
            # Need to transport old gradient and direction to the current manifold point's tangent space
            # A_candidate is a temporary point to compute the tangent space for transport.
            # This A_candidate needs to be generated consistently with how the actual A_new will be generated.
            # For a fixed alpha, A_new_approx = R_A(P_k_minus_1, alpha) can be used.
            # However, since the final A_new is from R_A(P_k, alpha), a more robust transport point for CG is
            # often just the current point A, or the point from the *previous* retraction step.
            # Given that 'update' only receives 'A' (current point) and 'D' (current Euclidean gradient)
            # and 'alpha' (fixed step size), the simplest is to transport to A's tangent space or
            # approximate transport to the next point's tangent space using A as reference.

            # We will use A as the base point for transporting the previous vectors to its tangent space.
            # Then, we compute P_k, and finally perform retraction from A using P_k to get A_new.
            
            transported_prev_gradient = self._vector_transport(self._previous_gradient, A)
            transported_prev_direction = self._vector_transport(self._previous_direction, A)
            
            # y_k = G_k_preconditioned - transported_prev_gradient
            # Denominator should not be zero; add small epsilon if needed
            
            # Using Polak-Ribiere variant (common in Riemannian CG, often performs well)
            # beta_k_numerator = self._riemannian_inner_product(G_k_preconditioned, G_k_preconditioned - transported_prev_gradient)
            # beta_k_denominator = self._riemannian_inner_product(self._previous_gradient, self._previous_gradient)

            # A common variant for Hager-Zhang that ensures descent, often used in Riemannian context:
            # nu_k = G_k - T_{W_{k-1} -> W_k}(G_{k-1})
            # beta_k = max(0, (Re(Tr(G_k^dagger * nu_k))) / Re(Tr(P_{k-1}^dagger * T_{W_{k-1} -> W_k}(G_{k-1}))))

            # Let's use a simpler version often found, like Fletcher-Reeves or Polak-Ribiere for simplicity,
            # which are also commonly extended to Riemannian manifolds.
            # Polak-Ribiere often modified with max(0, ...) for global convergence.
            
            # Let's follow a commonly cited Riemannian PR-CG update:
            # beta_k = max(0, <G_k, G_k - T(G_{k-1})> / <P_{k-1}, T(G_{k-1})>)
            # where <.,.> is the Riemannian metric inner product.

            # The current G_k is already preconditioned, if preconditioning is on.
            y_k = G_k_preconditioned - transported_prev_gradient
            
            beta_k_numerator = self._riemannian_inner_product(G_k_preconditioned, y_k)
            beta_k_denominator = self._riemannian_inner_product(self._previous_direction, transported_prev_gradient)
            
            # Avoid division by zero
            if np.abs(beta_k_denominator) < 1e-16:
                beta_k = 0.0
            else:
                beta_k = max(0.0, beta_k_numerator / beta_k_denominator)
            
            P_k = -G_k_preconditioned + beta_k * transported_prev_direction

        # Perform retraction using the new conjugate gradient direction
        # The retraction_instance needs to handle its own preconditioning if self.preconditioning is True.
        # However, it's generally better to apply preconditioning ONCE to the search direction (P_k)
        # before passing it to a "raw" retraction.
        # The `retraction_instance` is initialized with `preconditioning=False` to avoid double preconditioning.
        A_new, norm_of_direction = self.retraction_instance.update(A, P_k, alpha, r=None) # Pass r=None to avoid re-preconditioning

        # Store current gradient and direction for the next iteration
        self._previous_gradient = G_k_preconditioned # Store the preconditioned gradient
        self._previous_direction = P_k # Store the preconditioned direction
        self._iteration += 1

        # Return the new manifold point and the norm of the current (preconditioned) gradient
        # The norm reported should ideally be the norm of the gradient, not the search direction.
        return A_new, np.linalg.norm(G_k) # Return norm of true gradient (before preconditioning for reporting)
