import numpy as np
from ncon import ncon
from qdmt.uniform_mps import UniformMps
from qdmt.cost import HilbertSchmidt
from qdmt.transfer_matrix import TransferMatrix
from qdmt.transfer_matrix import RightFixedPoint
from scipy.linalg import svdvals
from scipy.sparse.linalg import eigs



def compute_second_Reyni(A: UniformMps, L: int):
    C = HilbertSchmidt(A, L)
    return -np.log(C.costAA)

# def log_negativity(A: UniformMps, L: int):
#     r = RightFixedPoint.from_mps(A)
#     E = TransferMatrix(A, A)
#     E = E.__pow__(L)
#     norm = ncon([E, r], [[1, 2, 3, 4], [4, 3]])

def build_rho(A: UniformMps, L: int, debug=False):
    rf = TransferMatrix.new(A, A).right_fixed_point()
    rfp = rf.tensor

    if debug:
        print("[build_rho] tr(rfp) =", np.trace(rfp))
        print("[build_rho] rfp =\n", rfp)

    chain = A.to_mps_chain(L)

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


def HS_overlap_mps(A, B, L):
    """
    Hilbert–Schmidt overlap between the L-site reduced density matrices of two uMPS
    objects A and B (your UniformMps instances).

    Returns: Tr[rhoA rhoB] (real scalar up to numerical noise)
    """
    rhoA = rho_matrix(A, L)
    rhoB = rho_matrix(B, L)
    val = np.trace(rhoA @ rhoB)
    return float(np.real_if_close(val))

def HS_distance_L_mps(A, B, L):
    """
    Hilbert–Schmidt overlap between the L-site reduced density matrices of two uMPS
    objects A and B (your UniformMps instances).

    Returns: Tr[rhoA rhoB] (real scalar up to numerical noise)
    """
    rhoA = rho_matrix(A, L)
    rhoB = rho_matrix(B, L)
    rho_diff=rhoA-rhoB
    val = np.trace(rho_diff @ rho_diff)
    return float(np.real_if_close(val))


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



def dominant_mixed_transfer_eig(A: np.ndarray, B: np.ndarray) -> complex:
    """
    Dominant eigenvalue mu of the mixed transfer matrix E_AB.

    A, B: uMPS tensors with shape (D, d, D) in order A[left, phys, right].
    Returns: mu (complex), largest magnitude eigenvalue of E_AB (size D^2 x D^2).
    """
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("A and B must have shape (D, d, D).")
    DA, dA, DA2 = A.shape
    DB, dB, DB2 = B.shape
    if (DA != DA2) or (DB != DB2) or (DA != DB) or (dA != dB):
        raise ValueError(f"Shape mismatch: A={A.shape}, B={B.shape} (need same D and d).")

    # E_AB(l,l',r,r') = sum_s A(l,s,r) * conj(B(l',s,r'))
    E = ncon((A, B.conj()), ((-1, 1, -3), (-2, 1, -4)))  # (l, l', r, r')
    M = E.reshape(DA * DA, DA * DA)

    vals = eigs(M, k=1, which="LM", return_eigenvectors=False)
    return vals[0]

def loschmidt_rate_per_site(A_t, A_0) -> float:
    """
    Per-site Loschmidt rate function between uMPS states A_t and A_0:
        lambda = -2 * log |mu|
    where mu is the dominant eigenvalue of the mixed transfer matrix.

    Returns a nonnegative float (up to numerical noise).
    """
    mu = dominant_mixed_transfer_eig(A_t, A_0)
    return float(-2.0 * np.log(np.abs(mu)))


def compute_loschmidt(states):
    n = len(states)
    results = np.zeros(n)
    B = UniformMps(states[0])

    for i in range(0, n):
        A = UniformMps(states[i])
        
        results[i] = loschmidt_rate_per_site(A.tensor,B.tensor) 
    return results 


import numpy as np

def evolve_4qubit_density_with_two_copy_U(
    rho: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """
    Given:
      rho : 8x8 density matrix (3 qubits = 2^3).   [If you truly mean 4 qubits, rho should be 16x16.]
      U   : 4x4 two-qubit unitary

    Builds U2 = U ⊗ U and returns rho' = U2 rho U2†.
    """
    rho = np.asarray(rho, dtype=complex)
    U = np.asarray(U, dtype=complex)

    if U.shape != (4, 4):
        raise ValueError(f"U must be 4x4 (two-qubit unitary), got {U.shape}")

    U2 = np.kron(U, U)

    if rho.shape != U2.shape:
        raise ValueError(
            f"rho shape {rho.shape} must match U⊗U shape {U2.shape}. "
            f"(Note: U⊗U is 16x16; if rho is 8x8 you're on 3 qubits, not 4.)"
        )

    return U2 @ rho @ U2.conj().T


from scipy.linalg import expm

# Pauli matrices
_I = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

def two_qubit_U(g: float, dt: float) -> np.ndarray:
    """
    Build the 2-qubit unitary:
        U = exp(i dt Z⊗Z) @ exp(i dt (g/2) X⊗I) @ exp(i dt (g/2) I⊗X)
    Returns a 4x4 complex array.
    """
    ZZ = np.kron(_Z, _Z)
    X1 = np.kron(_X, _I)
    _1X = np.kron(_I, _X)

    U_zz = expm(1j * dt * ZZ)
    U_x1 = expm(1j * dt * (g / 2.0) * X1)
    U_1x = expm(1j * dt * (g / 2.0) * _1X)

    return U_zz @ U_x1 @ U_1x


import numpy as np

def trace_out_leftmost_qubit_4to3(rho: np.ndarray) -> np.ndarray:
    """
    Trace out qubit 0 from a 4-qubit RDM (16x16),
    returning 3-qubit RDM (8x8) for qubits (1,2,3).
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (16, 16):
        raise ValueError("rho must be 16x16")

    rho8 = rho.reshape(2,2,2,2, 2,2,2,2)  # i0,i1,i2,i3,j0,j1,j2,j3
    out = np.einsum("a b c d a f g h -> b c d f g h", rho8)
    return out.reshape(8, 8)


def trace_out_rightmost_qubit_4to3(rho: np.ndarray) -> np.ndarray:
    """
    Trace out qubit 3 from a 4-qubit RDM (16x16),
    returning 3-qubit RDM (8x8) for qubits (0,1,2).
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (16, 16):
        raise ValueError("rho must be 16x16")

    rho8 = rho.reshape(2,2,2,2, 2,2,2,2)  # i0,i1,i2,i3,j0,j1,j2,j3
    out = np.einsum("a b c d e f g d -> a b c e f g", rho8)
    return out.reshape(8, 8)

