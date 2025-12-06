import numpy as np
from ncon import ncon
from qdmt.uniform_mps import UniformMps
from qdmt.cost import HilbertSchmidt
from qdmt.transfer_matrix import TransferMatrix
from qdmt.transfer_matrix import RightFixedPoint
from scipy.linalg import svdvals


def compute_second_Reyni(A: UniformMps, L: int):
    C = HilbertSchmidt(A, L)
    return -np.log(C.costAA)

# def log_negativity(A: UniformMps, L: int):
#     r = RightFixedPoint.from_mps(A)
#     E = TransferMatrix(A, A)
#     E = E.__pow__(L)
#     norm = ncon([E, r], [[1, 2, 3, 4], [4, 3]])



def build_rho(A: UniformMps, L: int):
    """
    Build the 2L-index density matrix ρ for an MPS and right fixed point.
    If no arguments are provided, use the default A and rA.
    """

    rf = TransferMatrix.new(A, A).right_fixed_point()
    rfp = rf.tensor


    # Build the MPS chain
    chain = A.to_mps_chain(L)

    # Index wiring identical for both rhoA and rhoB
    ket_idx = [1] + [-(i+1)     for i in range(L)] + [2]
    bra_idx = [1] + [-(L+i+1)   for i in range(L)] + [3]
    env_idx = [2, 3]

    return ncon([chain, chain.conj(), rfp],
                [ ket_idx, bra_idx,   env_idx ])    



def rho_matrix(A, L):
    rho = build_rho(A, L)
    d = rho.shape[0]       # assumes all physical legs equal
    dim = d**L
    # flatten first L legs into one index, last L legs into one index
    rho_mat = rho.reshape(dim, dim)
    return rho_mat

def compute_von_neumann_entropy(A, L, eps=1e-16):
    rho = rho_matrix(A, L)

    # enforce Hermiticity numerically (just in case of small noise)
    rho = 0.5 * (rho + rho.conj().T)

    # normalize
    tr = np.trace(rho)
    rho /= tr

    # eigenvalues (sorted ascending)
    evals = np.linalg.eigvalsh(rho)

    # project away tiny negative numerical artifacts
    evals = np.real(evals)
    evals[evals < 0] = 0.0

    # avoid log(0): define 0 log 0 = 0 by cutting off small eigenvalues
    evals = evals[evals > eps]

    # compute S = -∑ λ log λ
    S = -np.sum(evals * np.log(evals))
    # print(S)

    return S


def trace_distance(rho, sigma):
    """
    Compute the trace distance between two density matrices:
        D(ρ,σ) = 1/2 || ρ − σ ||_1

    Both inputs should be square matrices of the same dimension.
    """
    # Hermitize to reduce numerical noise
    rho = 0.5 * (rho + rho.conj().T)
    sigma = 0.5 * (sigma + sigma.conj().T)

    # Optional: renormalize if slight trace drift exists
    rho /= np.trace(rho)
    sigma /= np.trace(sigma)

    # singular values of the difference give the trace norm
    diff = rho - sigma
    sval = svdvals(diff)

    # print(np.linalg.norm(diff))
    return 0.5 * np.sum(sval)


def trace_distance_mps(A, B, L):
    """
    Compute the trace distance between the L-site reduced density matrices
    of two uniform MPS tensors A and B.

    Uses: build_rho → rho_matrix → trace_distance
    """

    rhoA = rho_matrix(A, L)
    rhoB = rho_matrix(B, L)

    return trace_distance(rhoA, rhoB)


def compute_trace_distance_successive(states, L):
    """
    Given a list/array of raw MPS tensors (as in data['state']),
    compute the trace distance between the L-site RDMs of
    successive states:

        D[0] = 0
        D[i] = trace_distance_mps(state[i], state[i-1], L)

    Returns
    -------
    np.array of length len(states).
    """

    n = len(states)
    results = np.zeros(n)

    for i in range(1, n):
        A = UniformMps(states[i])
        B = UniformMps(states[i - 1])
        results[i] = trace_distance_mps(A, B, L)

    return results


def compute_trace_distance_to_average(states, dt, t_cut, L):
    """
    Compute trace distance between rho[i] and the average RDM over all
    times t >= t_cut.

    Parameters
    ----------
    states : list/array of MPS tensors
    dt : float
        Time step
    t_cut : float
        Cutoff time for defining the steady/average state
    L : int
        Block size for reduced density matrix extraction
    """

    T = len(states)

    # --- 1. Compute cutoff index ---
    i_cut = int(t_cut / dt)
    if i_cut >= T:
        raise ValueError("t_cut is beyond the simulated time range.")

    # --- 2. Compute all RDMs ---
    rhos = []
    for A in states:
        mps = UniformMps(A)
        rhos.append(rho_matrix(mps, L))

    # --- 3. Average RDM over i >= i_cut ---
    steady_rhos = rhos[i_cut:]
    rho_avg = sum(steady_rhos) / len(steady_rhos)

    # --- 4. Compute trace distance D[i] = dist(rho[i], rho_avg) ---
    D = np.zeros(T)
    for i in range(T):
        D[i] = trace_distance(rhos[i], rho_avg)

    return D
