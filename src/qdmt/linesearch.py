import numpy as np
from typing import Callable
from dataclasses import dataclass

from qdmt.manifold import AbstractManifold, euclidian_metric
from qdmt.cost import AbstractCostFunction
from qdmt.uniform_mps import UniformMps

@dataclass
class LineSearchPoint:
    alpha: float  # Step size
    C: float      # Cost function value C(A)
    gW: float     # Xal derivative ∇C(A)ᵀd

def linesearch(
    f: AbstractCostFunction,
    M: AbstractManifold,
    W: np.ndarray,
    X: np.ndarray,
    C: float,
    G: np.ndarray,
    retraction: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    alpha0: float = 1.0,
    c1: float = 1e-3,
    c2: float = 0.9,
    epsilon: float = 1e-8,
    rho: float = 5.0,
    gamma: float = 0.66,
    max_iter: int = 20
) -> tuple[float, np.ndarray, float, np.ndarray, bool] | tuple[None, None, None, None, bool] :
    """
    Performs a line search using the Hager-Zhang algorithm (Algorithm 851: CG_DESCENT).

    Finds a step size `alpha` that satisfies the (approximate) Wolfe conditions.

    Args:
        f_df: A function that takes a vector `x` and returns `(f, df)`, where `f` is the function value and `df` is the gradient.
        W: The starting point of the line search.
        X: The search direction.
        C: The cost function value at W.
        G: The gradient at W.
        alpha0: The initial guess for the step size.
        c1, c2: Wolfe condition parameters.
        epsilon: Tolerance for the approximate Wolfe condition.
        rho: Expansion factor for the bracketing phase.
        gamma: Bisection fallback factor.
        max_iter: Maximum number of iterations.

    Returns:
        A tuple (alpha, x_new, f_new, g_new) if successful, otherwise None.
    """

    n, p = W.shape

    # --- Helper Functions ---
    def take_step(alpha: float):
        """Computes function value and derivative at a new point."""

        # update tensor and slope
        W_prime, X_prime = retraction(W, X, alpha)

        # compute cost and gradient at new point
        f.B = UniformMps(W_prime)
        C_prime, D = f.cost(), f.derivative().reshape(n, p)
        G_prime = M.project(W_prime, D)

        # compute slope at new point 
        gW_prime = euclidian_metric(G_prime, X_prime)
        return LineSearchPoint(alpha, C_prime, gW_prime), W_prime, G_prime

    def check_approx_wolfe(p: LineSearchPoint, p0: LineSearchPoint):
        """Checks if a point `p` satisfies the Wolfe conditions relative to `p0`."""
        exact_wolfe = (p.C <= p0.C + c1 * p.alpha * p0.gW) and \
                      (abs(p.gW) <= abs(c2 * p0.gW))
        approx_wolfe = (p.C <= p0.C + epsilon) and \
                       ((2 * c1 - 1) * p0.gW >= p.gW >= c2 * p0.gW)
        return exact_wolfe or approx_wolfe

    gW = euclidian_metric(G, X)
    if gW >= 0:
        print("Warning: Search X is not a descent X.")
        return None

    p0 = LineSearchPoint(0.0, C, gW)
    a, b = p0, p0 # `a` is the lower bound, `b` is the upper bound of the bracket
    
    # --- Main Loop ---
    alpha = alpha0
    for k in range(max_iter):
        # 1. Take a trial step
        p_trial, x_trial, g_trial = take_step(alpha)

        # 2. Check for success
        if check_approx_wolfe(p_trial, p0):
            return alpha, x_trial, p_trial.C, g_trial, True

        # 3. Bracket the interval [a, b]
        if k == 0: # First iteration is for bracketing
            if p_trial.gW >= 0:
                a, b = p0, p_trial
            else: # dphi < 0, so we can expand
                a = p_trial
                alpha *= rho
                continue
        
        # 4. Update the bracket [a, b] using the new point p_trial
        d_alpha_prev = b.alpha - a.alpha
        if p_trial.gW >= 0:
            b = p_trial
        elif p_trial.C <= p0.C + epsilon: # dphi < 0 and phi is low enough
            a = p_trial
        else: # dphi < 0 and phi is too high (overshot a "hump")
            b = p_trial

        # 5. Zoom: Find the next trial alpha within the new bracket [a, b]
        if a.alpha == b.alpha:
             # Should not happen if logic is correct
             break

        # Secant method step to find the next alpha
        # This is a key part of the HZ algorithm for fast convergence
        alpha = (a.alpha * b.gW - b.alpha * a.gW) / (b.gW - a.gW)
        
        # Fallback to bisection if secant step is not effective enough
        if b.alpha - a.alpha > gamma * d_alpha_prev:
            alpha = (a.alpha + b.alpha) / 2.0
            
        # Ensure the next alpha is within the bracket
        if alpha <= min(a.alpha, b.alpha) or alpha >= max(a.alpha, b.alpha):
            alpha = (a.alpha + b.alpha) / 2.0

    print("Warning: Line search did not converge within max_iter.")
    return None, None, None, None, False