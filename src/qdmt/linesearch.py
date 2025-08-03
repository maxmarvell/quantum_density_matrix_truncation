import numpy as np
from typing import Callable
from dataclasses import dataclass

from qdmt.manifold import AbstractManifold, euclidian_metric
from qdmt.cost import AbstractCostFunction
from qdmt.uniform_mps import UniformMps
from qdmt.fixed_point import RightFixedPoint

@dataclass
class LineSearchPoint:
    alpha: float  # Step size
    C: float      # Cost function value C(A)
    gW: float     # Xal derivative ∇C(A)ᵀd

def check_exact_wolfe(p: LineSearchPoint, p0: LineSearchPoint, c1: float, c2: float) -> bool:
    return (p.C <= p0.C + c1 * p.alpha * p0.gW) and (p.gW >= c2 * p0.gW)

def check_approx_wolfe(p: LineSearchPoint, p0: LineSearchPoint, c1: float, c2: float, epsilon: float) -> bool:
    return (p.C <= p0.C + epsilon) and ((2 * c1 - 1) * p0.gW >= p.gW >= c2 * p0.gW)

def satisfies_wolfe(p: LineSearchPoint, p0: LineSearchPoint, c1, c2, epsilon) -> bool:
    return check_exact_wolfe(p, p0, c1, c2) or check_approx_wolfe(p, p0, c1, c2, epsilon)

def bracket(
    take_step: Callable[[float], tuple[LineSearchPoint, np.ndarray, np.ndarray]],
    p0: LineSearchPoint,
    alpha0: float,
    rho: float,
    epsilon: float,
    c1: float,
    c2: float,
    max_iter: int
) -> tuple[LineSearchPoint, LineSearchPoint] | None:
    alpha = alpha0
    a = p0

    for _ in range(max_iter):
        p, _, _ = take_step(alpha)

        if not np.isfinite(p.C) or not np.isfinite(p.gW):
            alpha = (a.alpha + alpha) / 2
            continue

        if p.gW >= 0:
            return a, p  # Bracket found

        if p.C > p0.C + epsilon:
            return a, p  # Bracketed by high cost

        a = p
        alpha *= rho

    return None

def zoom(
    take_step: Callable[[float], tuple[LineSearchPoint, np.ndarray, np.ndarray]],
    p0: LineSearchPoint,
    a: LineSearchPoint,
    b: LineSearchPoint,
    c1: float,
    c2: float,
    epsilon: float,
    gamma: float,
    max_iter: int
) -> tuple[LineSearchPoint, np.ndarray, np.ndarray] | None:

    for _ in range(max_iter):
        d_alpha_prev = b.alpha - a.alpha

        # Secant guess
        alpha = (a.alpha * b.gW - b.alpha * a.gW) / (b.gW - a.gW)
        # Fallback
        if b.alpha - a.alpha > gamma * d_alpha_prev or not (a.alpha < alpha < b.alpha):
            alpha = 0.5 * (a.alpha + b.alpha)

        p, x, g = take_step(alpha)

        if satisfies_wolfe(p, p0, c1, c2, epsilon):
            return p, x, g

        if p.gW >= 0:
            b = p
        elif p.C <= p0.C + epsilon:
            a = p
        else:
            b = p

        if abs(b.alpha - a.alpha) < 1e-10:
            break

    return None

def linesearch(
    fg: AbstractCostFunction,
    M: AbstractManifold,
    W: np.ndarray,
    X: np.ndarray,
    C: float,
    G: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    epsilon: float = 1e-6,
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

    # --- Helper Functions ---
    def take_step(alpha: float):
        """Computes function value and derivative at a new point."""

        # update tensor and slope
        W_prime, X_prime = M.retract(W, X, alpha)

        # compute cost and gradient at new point
        B = UniformMps(W_prime)
        rB = RightFixedPoint.from_mps(B)
        C_prime, G_prime = fg(B, rB)

        # compute slope at new point 
        gW_prime = euclidian_metric(G_prime, X_prime)
        return LineSearchPoint(alpha, C_prime, gW_prime), W_prime, G_prime

    gW = euclidian_metric(G, X)
    if gW >= 0:
        print("Warning: Search X is not a descent X.")
        return None

    p0 = LineSearchPoint(0.0, C, gW)
    a, b = p0, p0 # `a` is the lower bound, `b` is the upper bound of the bracket
    
    def take_step_wrapper(alpha):
        return take_step(alpha)
    
    # Try initial step
    p_init, x_init, g_init = take_step(alpha0)
    if satisfies_wolfe(p_init, p0, c1, c2, epsilon):
        return p_init.alpha, x_init, p_init.C, g_init, True

    # Bracketing phase
    bracket_result = bracket(take_step_wrapper, p0, alpha0, rho, epsilon, c1, c2, max_iter)
    if bracket_result is None:
        print("Warning: Could not bracket.")
        return alpha0, W, C, G, False

    a, b = bracket_result

    # Zoom phase
    zoom_result = zoom(take_step_wrapper, p0, a, b, c1, c2, epsilon, gamma, max_iter)
    if zoom_result is not None:
        p_final, x_final, g_final = zoom_result
        return p_final.alpha, x_final, p_final.C, g_final, True

    print("Warning: Line search did not converge within max_iter.")
    return alpha0, W, C, G, False