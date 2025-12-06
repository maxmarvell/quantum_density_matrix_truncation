from src.qdmt.evolve import *
from src.qdmt.transfer_matrices import *
from qdmt.transfer_matrix import TransferMatrix
from scipy.linalg import sqrtm, norm

def matrix_sqrt(A, tol=1e-8):
    A = np.asarray(A)
    
    # Compute matrix square root
    S = sqrtm(A)
    
    # Check: S @ S should equal A
    err = norm(S @ S - A)
    if err > tol:
        raise ValueError(f"Matrix square root failed accuracy check: error={err}")
    
    return S



def hermitian_matrix_sqrt(A, tol=1e-12):
    A = np.asarray(A)

    # Symmetrize defensively
    A = 0.5 * (A + A.conj().T)

    # Eigen-decomposition (for Hermitian → guaranteed real vals)
    eigvals, eigvecs = np.linalg.eigh(A)

    # Check for negative eigenvalues
    if np.any(eigvals < -tol):
        raise ValueError("Matrix is not positive semidefinite; no real sqrt exists.")

    # Clip tiny negative numerical noise
    eigvals_clipped = np.clip(eigvals, 0, None)

    # Construct sqrt
    sqrt_diag = np.sqrt(eigvals_clipped)
    S = eigvecs @ np.diag(sqrt_diag) @ eigvecs.conj().T

    # Verify accuracy
    err = np.linalg.norm(S @ S - A) / max(1, np.linalg.norm(A))
    if err > 1e-10:
        raise ValueError(f"Inaccurate sqrt: relative error {err}")

    return S


def phys(A, s):
    """
    Return the matrix A[s] for a left-canonical MPS tensor A of shape (D, d, D).
    
    A: numpy array with shape (D, d, D)
    s: physical index (0 <= s < d)
    """
    A = np.asarray(A)
    if A.ndim != 3:
        raise ValueError(f"A must have shape (D,d,D), got {A.shape}")
    D1, d, D2 = A.shape
    if not (0 <= s < d):
        raise IndexError(f"physical index {s} out of bounds (0..{d-1})")
    
    # Return the (D × D) matrix for physical index s
    return A[:, s, :]

def gen_words(A: UniformMps, L: int, start, d=2):
    Atens = A.tensor
    all_words=[start]
    words_now=[start]
    for k in range(0,L):
        next_words=[]
        for word in words_now:
            for s in range(0,d):
                newword=phys(Atens,s)@word
                next_words.append(newword)
                all_words.append(newword)
        words_now=next_words
        # print(len(all_words))
    return all_words    


def vec(M):
    """Vectorize a matrix M by stacking its columns."""
    return M.reshape(-1, order='F')   # Fortran order = column-major

def cols_to_matrix(vectors):
    """Stack a list of 1D arrays into a matrix with those vectors as columns."""
    return np.column_stack(vectors)

def mats_to_vecmat(mats):
    """
    Given a list of matrices, return a matrix whose columns
    are the vectorized forms of those matrices.
    """
    vecs = [vec(M) for M in mats]     # vectorize each matrix
    return cols_to_matrix(vecs)       # stack as columns


def compress(A, U, D):
    I = np.eye(D)
    A_otimes_I = np.kron(A, I)
    return U.conj().T @ A_otimes_I @ U


from itertools import product

###############################################################################
# Helper: contract a sequence of physical indices into A
###############################################################################
def apply_A_sequence(A, seq):
    """
    Computes A^{s1} A^{s2} ... A^{sL}

    A has shape (D, d, D).
    seq is a list/tuple of physical indices [s1, s2, ..., sL].
    """
    D, d, _ = A.shape
    M = np.eye(D, dtype=complex)
    for s in seq:
        M = M @ A[:, s, :]   # (D,D) @ (D,D) -> (D,D)
    return M


###############################################################################
# Main: build the L-site reduced density matrix
###############################################################################
def reduced_density_matrix(A, R, L):
    """
    Computes the L-site reduced density matrix of an infinite uMPS.

    A: tensor of shape (D, d, D)
    R: right fixed point of shape (D, D)
    L: block size

    Returns:
        rho: np.array of shape (d**L, d**L)
    """
    D, d, _ = A.shape

    # All possible sequences of length L
    sequences = list(product(range(d), repeat=L))
    num = len(sequences)

    rho = np.zeros((num, num), dtype=complex)

    # Precompute left-multiplied blocks
    block_left = {}
    for idx, seq in enumerate(sequences):
        block_left[idx] = apply_A_sequence(A, seq)

    # Precompute right-multiplied blocks: A^{t_L} ... A^{t_1}
    block_right = {}
    for idx, seq in enumerate(sequences):
        M = np.eye(D, dtype=complex)
        # Note reversed order for bras
        for s in reversed(seq):
            M = A[:, s, :].conj().T @ M
        block_right[idx] = M

    # Fill rho
    for i in range(num):
        for j in range(num):
            rho[i, j] = np.trace(block_left[i] @ R @ block_right[j])

    return rho




###############################################################################
# Positive matrix square root
###############################################################################
def sqrt_pos(mat, eps=1e-14):
    """Positive square root of a Hermitian PSD matrix."""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.maximum(vals, eps)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


###############################################################################
# Compute left fixed point of the adjoint transfer map
###############################################################################

def compute_left_fixed_point(A, tol=1e-12, maxiter=10000):
    """
    Compute L satisfying  L = Σ_s A_s† L A_s,
    normalized so trace(L)=1.

    A.shape = (D, d, D)
    """
    D, d, _ = A.shape
    L = np.eye(D, dtype=complex) / D

    for _ in range(maxiter):
        L_new = np.zeros((D, D), dtype=complex)
        for s in range(d):
            As = A[:, s, :]
            L_new += As.conj().T @ L @ As

        tr = np.trace(L_new)
        if abs(tr) < 1e-14:
            raise ValueError("Left fixed point iteration became unstable (trace→0).")
        L_new /= tr

        if np.linalg.norm(L_new - L) < tol:
            L = L_new
            break

        L = L_new

    # enforce Hermiticity
    L = 0.5 * (L + L.conj().T)
    return L
def sqrt_pos(mat, eps=1e-14):
    """Positive square root of a Hermitian PSD matrix."""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.maximum(vals, eps)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T

def left_canonical_gauge(A):
    """
    Given A of shape (D, d, D), return:
      G       : (D,D) gauge
      A_tilde : (D,d,D) left-canonical tensors
      Q       : Σ_s A_tilde[s]† A_tilde[s]  (should be ≈ I)
    such that A_tilde[s] = G @ A[s] @ G^{-1}.
    """
    D, d, _ = A.shape

    # 1. Left fixed point of adjoint channel
    L = compute_left_fixed_point(A)

    # 2. G = L^{1/2}
    G = sqrt_pos(L)
    G_inv = np.linalg.inv(G)

    # 3. Gauge transform
    A_tilde = np.empty_like(A)
    for s in range(d):
        As = A[:, s, :]
        A_tilde[:, s, :] = G @ As @ G_inv

    # 4. Check left-canonical condition
    Q = np.zeros((D, D), dtype=complex)
    for s in range(d):
        As = A_tilde[:, s, :]
        Q += As.conj().T @ As

    return G, A_tilde, Q

def left_canonical_gauge_GAGinv(A):
    """
    Canonicalize via similarity transform A'_s = G A_s G^{-1}
    so that Σ_s A'_s† A'_s = I.
    Exactly matches UniformMps.is_isometry().
    """
    D, d, _ = A.shape

    # 1. Compute L = Σ A_s† A_s
    L = np.zeros((D, D), dtype=complex)
    for s in range(d):
        As = A[:, s, :]
        L += As.conj().T @ As

    # 2. Make Hermitian (numerical cleanup)
    L = 0.5 * (L + L.conj().T)

    # 3. Compute G = L^{1/2}
    G = sqrt_pos(L)        # positive Hermitian square root
    G_inv = np.linalg.inv(G)

    # 4. Apply gauge: A' = G A_s G^{-1}
    A_tilde = np.empty_like(A)
    for s in range(d):
        A_tilde[:, s, :] = G @ A[:, s, :] @ G_inv

    # 5. Check canonicality condition
    Q = np.zeros((D, D), dtype=complex)
    for s in range(d):
        As = A_tilde[:, s, :]
        Q += As.conj().T @ As

    return G, A_tilde, Q



def compress_rho_mps_old(A: UniformMps,L):
    rB = TransferMatrix.new(A, A).right_fixed_point()
    rB_mat = rB.tensor
    D = A.D
    # print(D)

    rho_pre=reduced_density_matrix(A.tensor, rB_mat, L)


    rB_half = hermitian_matrix_sqrt(rB_mat, tol=1e-15)
    # print(A.is_isometry())
    # print(L)
    words = gen_words(A, L, rB_half)
    hist_mat = mats_to_vecmat(words)

    # print(hist_mat.shape)
    U, S, Vh = np.linalg.svd(hist_mat, full_matrices=False)

    # print(words)
    # print(U.shape)

    As = []
    for s in range(d):
        As.append(compress(phys(A.tensor,s),U,D))

    # print(As)
    A_compress_tens=np.stack(As, axis=1)   

    A_renorm, lam0 = renormalize_transfer_spectrum(A_compress_tens)
    A_LC, _, _= left_canonical_gauge(A_renorm)
    print("test")
    A_renormm=UniformMps(A_LC)
    A_renormm.is_isometry()
    print("Original leading eigenvalue =", lam0)
    return A_renorm

    A_compress_tens = A_renorm
 
    A_compress=UniformMps(A_compress_tens)
    A_compress.is_isometry()

    G, A_compress_lc, Q = left_canonical_gauge_GAGinv(A_compress_tens)

    D_compress=UniformMps(A_compress_lc).D
    print(D_compress)

    UniformMps(A_compress_lc).is_isometry()
    
    # Compute compressed boundary vector omega
    # (from Omega = sqrt(R_original))
    Omega = rB_half                  # D×D
    Omega_vec = Omega.reshape(-1)    # (D^2,)
    omega = U.conj().T @ Omega_vec   # (D',)

    # After canonicalization:
    G_inv = np.linalg.inv(G)
    omega_lc = G_inv.conj().T @ omega     # correct transformed boundary
    R_compressed = np.outer(omega_lc, omega_lc.conj())

    rho_post = reduced_density_matrix(A_compress_lc, R_compressed, L)


import numpy as np

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def sqrt_pos(mat, eps=1e-14):
    """Positive square root of a Hermitian PSD matrix."""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.maximum(vals, eps)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def compute_left_fixed_point_restricted(B, tol=1e-12, maxiter=10000):
    """
    Compute the left fixed point L_S for the restricted tensors B_s:

        L_S = Σ_s B_s† L_S B_s

    B has shape (D_S, d, D_S).
    Returns L_S of shape (D_S, D_S), with trace normalized to 1.
    """
    D_S, d, _ = B.shape
    L = np.eye(D_S, dtype=complex) / D_S

    for _ in range(maxiter):
        L_new = np.zeros((D_S, D_S), dtype=complex)
        for s in range(d):
            Bs = B[:, s, :]
            L_new += Bs.conj().T @ L @ Bs

        tr = np.trace(L_new)
        if abs(tr) < 1e-14:
            raise ValueError("Left fixed point iteration on S became unstable (trace→0).")
        L_new /= tr

        if np.linalg.norm(L_new - L) < tol:
            L = L_new
            break

        L = L_new

    L = 0.5 * (L + L.conj().T)
    return L


def reachable_subspace(A_tens, omega, L):
    """
    Build the subspace S ⊂ C^{D'} reached from boundary omega
    by applying strings of A_s up to length L.

    A_tens : (D', d, D')
    omega  : (D',)
    L      : max depth
    Returns:
        U_S : (D', D_S) with orthonormal columns spanning S
    """
    Dp, d, _ = A_tens.shape

    vecs = []
    frontier = [omega]

    for depth in range(L + 1):
        new_frontier = []
        for v in frontier:
            vecs.append(v)
            for s in range(d):
                As = A_tens[:, s, :]
                new_v = As @ v
                new_frontier.append(new_v)
        frontier = new_frontier

    M = np.stack(vecs, axis=1)         # (D', N)
    U_S, _ = np.linalg.qr(M)           # (D', D_S)
    return U_S


def canonicalize_on_subspace(A_tens, omega, L_sub):
    """
    Canonicalize the compressed MPS tensors A_tens on the subspace
    reachable from omega up to length L_sub, using a similarity
    transform W = U_S G_S U_S† + P_perp so that within S the tensors
    satisfy a left-canonical condition (in the MPS sense).

    A_tens : (D', d, D')
    omega  : (D',)
    L_sub  : integer (e.g. use L_sub = L)

    Returns:
        W            : (D', D') gauge matrix
        W_inv        : (D', D') inverse of W
        A_canon_tens : (D', d, D') canonically gauged tensors
        omega_canon  : (D',) transformed boundary vector
    """
    Dp, d, _ = A_tens.shape

    # 1. Reachable subspace S from omega
    U_S = reachable_subspace(A_tens, omega, L_sub)  # (D', D_S)
    D_S = U_S.shape[1]

    # 2. Restricted tensors: B_s = U_S† A_s U_S
    B = np.empty((D_S, d, D_S), dtype=complex)
    for s in range(d):
        As = A_tens[:, s, :]
        B[:, s, :] = U_S.conj().T @ As @ U_S

    # 3. Left fixed point on S: L_S = Σ_s B_s† L_S B_s
    L_S = compute_left_fixed_point_restricted(B)

    # 4. G_S = L_S^{1/2}, similarity gauge on S: B' = G_S B G_S^{-1}
    G_S = sqrt_pos(L_S)
    G_S_inv = np.linalg.inv(G_S)

    # 5. Full-space similarity W
    P_S = U_S @ U_S.conj().T
    P_perp = np.eye(Dp, dtype=complex) - P_S

    W = U_S @ G_S @ U_S.conj().T + P_perp
    W_inv = U_S @ G_S_inv @ U_S.conj().T + P_perp

    # 6. Apply global similarity: A'_s = W A_s W^{-1}
    A_canon = np.empty_like(A_tens)
    for s in range(d):
        As = A_tens[:, s, :]
        A_canon[:, s, :] = W @ As @ W_inv

    # 7. Transform boundary vector:
    omega_canon = W_inv.conj().T @ omega

    # 8. Diagnostic: Q_S = U_S† (Σ A'† A') U_S should be ~ I_S
    Q_full = np.zeros((Dp, Dp), dtype=complex)
    for s in range(d):
        As = A_canon[:, s, :]
        Q_full += As.conj().T @ As
    Q_S = U_S.conj().T @ Q_full @ U_S
    print("‖Q_S − I_S‖ =", np.linalg.norm(Q_S - np.eye(D_S)))

    return W, W_inv, A_canon, omega_canon


# -------------------------------------------------------------------
# Your compression + subspace-canonicalization pipeline
# -------------------------------------------------------------------

def compress_rho_mps(A: "UniformMps", L: int):
    """
    Compress an infinite uMPS A to a smaller bond-dimension uMPS that
    approximately reproduces the same L-site reduced density matrix,
    and canonicalize it on the physically relevant reachable subspace.

    Uses your existing compression for the transfer structure,
    then:
      - constructs compressed boundary omega,
      - canonicalizes on the reachable subspace.
    """
    D = A.D
    d = A.d

    # 1. Original right fixed point and rho_L
    rB = TransferMatrix.new(A, A).right_fixed_point()
    rB_mat = rB.tensor                      # (D, D)

    rho_pre = reduced_density_matrix(A.tensor, rB_mat, L)

    # 2. Build history matrix in Liouville space (your code)
    rB_half = hermitian_matrix_sqrt(rB_mat, tol=1e-15)

    words = gen_words(A, L, rB_half)       # list of (D, D) matrices
    hist_mat = mats_to_vecmat(words)       # (D^2, N)

    U, S, Vh = np.linalg.svd(hist_mat, full_matrices=False)  # U: (D^2, D')

    # 3. Build compressed tensors A_compress via your compress() helper
    As = []
    for s in range(d):
        As.append(compress(phys(A.tensor, s), U, D))
    A_compress_tens = np.stack(As, axis=1)    # (D', d, D')

    return A_compress_tens

    # 4. Compressed boundary vector omega from Omega = sqrt(R)
    Omega = rB_half                           # (D, D)
    Omega_vec = Omega.reshape(-1)             # (D^2,)
    omega = U.conj().T @ Omega_vec            # (D',)

    # 5. Canonicalize on physically reachable subspace
    W, W_inv, A_canon_tens, omega_canon = canonicalize_on_subspace(
        A_compress_tens, omega, L_sub=L
    )

    # 6. Build compressed right boundary matrix from omega_canon
    #    (Note: this matches the vec(purified) representation we used;
    #    exact rho_L equality is not guaranteed with this simple R, but
    #    it should get closer with better compression.)
    R_compress = np.outer(omega_canon, omega_canon.conj())   # (D', D')

    # 7. Compare rho_L before / after
    rho_post = reduced_density_matrix(A_canon_tens, R_compress, L)
    print("‖rho_pre − rho_post‖ =", np.linalg.norm(rho_pre - rho_post))

    # 8. Wrap in UniformMps and check isometry on full space
    A_compress_mps = UniformMps(A_canon_tens)
    A_compress_mps.is_isometry()

    return A_compress_tens, rho_pre, rho_post


def renormalize_transfer_spectrum(A):
    """
    Renormalize an MPS tensor A (D',d,D') so that the leading eigenvalue
    of the transfer matrix has magnitude exactly 1.

    Returns:
        A_renorm  : renormalized tensor (D',d,D')
        lam_max   : original largest eigenvalue
    """
    # Build transfer matrix as linear operator
    # Vectorized: vec(X) -> vec(sum_s A_s X A_s^\dagger)
    Dp, d, _ = A.shape
    D2 = Dp * Dp

    # Construct full transfer matrix explicitly (small D only)
    T = np.zeros((D2, D2), dtype=complex)
    for s in range(d):
        As = A[:, s, :]
        # (As ⊗ As*) acting on vec(X)
        T += np.kron(As, As.conj())

    # Eigenvalues
    evals, _ = np.linalg.eig(T)
    idx = np.argmax(np.abs(evals))
    lam_max = evals[idx]

    # Scaling
    alpha = 1.0 / np.sqrt(abs(lam_max))
    A_renorm = alpha * A

    return A_renorm, lam_max


def transfer_matrix(A):
    """
    Construct the transfer matrix T = Σ_s A_s ⊗ conj(A_s)
    for tensor A of shape (D, d, D).

    Returns:
        T  : (D*D, D*D) ndarray
    """
    D, d, _ = A.shape
    T = np.zeros((D*D, D*D), dtype=complex)

    for s in range(d):
        As = A[:, s, :]         # (D, D)
        T += np.kron(As, As.conj())
    return T


def left_eigensystem(A):
    """
    Compute left eigenvalues and left eigenvectors of the transfer matrix
    for tensor A_s.

    Returns:
        evals : eigenvalue array
        Ls    : list of reshaped left eigenmatrices L of shape (D, D)
    """
    T = transfer_matrix(A)
    evals, evecs_right = np.linalg.eig(T)        # right eigenvectors
    # left eigenvectors are right eigenvectors of T†
    evals_L, evecs_L = np.linalg.eig(T.conj().T)

    D2 = A.shape[0] * A.shape[0]
    Ls = []
    for i in range(D2):
        Lvec = evecs_L[:, i]
        L = Lvec.reshape(A.shape[0], A.shape[0])
        Ls.append(L)

    return evals_L, Ls


def analyze_spectrum(evals, tol=1e-6):
    lam1 = [ev for ev in evals if abs(ev - 1) < tol]
    print(f"Number of eigenvalues ≈ 1: {len(lam1)}")
    if len(lam1):
        print("They are:")
        print(lam1[:10])




import numpy as np


# ---------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------

def sqrt_pos(mat, eps=1e-14):
    """Positive square root of a Hermitian positive semidefinite matrix."""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.maximum(vals, eps)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


# ---------------------------------------------------------------
# STEP 1: Build reachable subspace in operator (Kraus) space
# ---------------------------------------------------------------

def build_kraus_reachable_subspace(A_tens, rhoR_half, L):
    """
    Build the operator subspace spanned by { A_{s_k} ... A_{s_1} rhoR_half }.
    This is a D x D matrix space → flatten to D^2 vectors.

    A_tens     : (D, d, D)
    rhoR_half  : (D, D)
    L          : max depth
    returns:
        U      : (D^2, D') isometry (orthonormal basis of subspace)
    """
    D, d, _ = A_tens.shape

    mats = []

    frontier = [rhoR_half]          # list of D×D matrices

    for depth in range(L + 1):
        new_frontier = []
        for X in frontier:
            mats.append(X.reshape(D*D))
            for s in range(d):
                As = A_tens[:, s, :]
                new_frontier.append(As @ X)   # (D,D)
        frontier = new_frontier

    M = np.stack(mats, axis=1)      # shape (D^2, N)
    U, _ = np.linalg.qr(M)          # orthonormal basis
    return U                        # D^2 × D'


# ---------------------------------------------------------------
# STEP 2: Compress Kraus operators via U† K U
# ---------------------------------------------------------------
def compress_kraus_channel(A_tens, U):
    """
    Compresses the channel defined by A_s via the Kraus isometry U.

    Inputs:
        A_tens : array (D, d, D)
        U      : array (D*D, Dp)   isometry in Liouville space

    Returns:
        A_comp : (Dp, d, Dp)
        Dp     : compressed bond dimension
    """
    D, d, _ = A_tens.shape
    Dp = U.shape[1]

    # Identity for the right leg
    I = np.eye(D, dtype=complex)

    # Prepare output
    A_comp = np.zeros((Dp, d, Dp), dtype=complex)

    for s in range(d):
        As = A_tens[:, s, :]                     # (D, D)

        # Build A_s ⊗ I   (D² × D²)
        Ls = np.kron(As, I)

        # Compressed Kraus operator = U† (A_s ⊗ I) U
        A_comp[:, s, :] = U.conj().T @ (Ls @ U)

    return A_comp, Dp


# ---------------------------------------------------------------
# STEP 3: Left-canonicalize compressed Kraus ops (now CP guaranteed!)
# ---------------------------------------------------------------

def left_canonicalize(A_comp):
    """
    Given compressed Kraus ops A_s (D', d, D') that define a proper CP map,
    compute the UNIQUE positive left fixed point L, then gauge:
        A_s' = G A_s G^{-1}, where G = L^{1/2}.
    This gives left-canonical form: Σ_s A_s'† A_s' = I.
    """
    Dp, d, _ = A_comp.shape

    # Compute left fixed point of E*(X) = Σ A† X A
    L = np.eye(Dp, dtype=complex) / Dp

    for _ in range(10000):
        L_new = np.zeros_like(L)
        for s in range(d):
            As = A_comp[:, s, :]
            L_new += As.conj().T @ L @ As
        tr = np.trace(L_new)
        if abs(tr) < 1e-14:
            raise RuntimeError("Degenerate channel: fixed point trace→0.")
        L_new /= tr
        if np.linalg.norm(L_new - L) < 1e-12:
            L = L_new
            break
        L = L_new

    L = 0.5*(L + L.conj().T)

    G = sqrt_pos(L)
    Ginv = np.linalg.inv(G)

    A_canon = np.zeros_like(A_comp)
    for s in range(d):
        A_canon[:, s, :] = G @ A_comp[:, s, :] @ Ginv

    return A_canon, L


# ---------------------------------------------------------------
# TOP LEVEL FUNCTION
# ---------------------------------------------------------------
def compress_rho_mps_kraus(A: UniformMps, L: int):
    """
    Compress an MPS using Kraus-isometry compression:
    U† (A_s ⊗ I) U produces the compressed tensors.
    """
    D = A.D
    d = A.d

    # ---- Step 1: Build history matrix (your code) ----

    rB = TransferMatrix.new(A, A).right_fixed_point()
    rB_mat = rB.tensor

    rB_half = hermitian_matrix_sqrt(rB_mat)
    words = gen_words(A, L, rB_half)
    hist_mat = mats_to_vecmat(words)   # shape (D², N)

    # ---- Step 2: SVD to get U ----

    U, S, Vh = np.linalg.svd(hist_mat, full_matrices=False)
    # U has shape (D², Dp)
    Dp = U.shape[1]

    # ---- Step 3: Compress Kraus channel ----

    A_comp, Dp = compress_kraus_channel(A.tensor, U)

    # ---- Step 4: Wrap into UniformMps ----

    A_comp_mps = UniformMps(A_comp)
    A_comp_mps.is_isometry()

    return A_comp, A_comp_mps





import numpy as np


# ------------------------------------------------------------
# Helper: Positive sqrt
# ------------------------------------------------------------
def sqrt_pos(M, eps=1e-14):
    vals, vecs = np.linalg.eigh(M)
    vals = np.maximum(vals, eps)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


# ------------------------------------------------------------
# Extract slices A_s from MPS tensor (D,d,D)
# ------------------------------------------------------------
def phys(A, s):
    return A[:, s, :]


# ------------------------------------------------------------
# Stinespring compression of channel
# ------------------------------------------------------------
def stinespring_compress(A):
    """
    Input:
        A : (D, d, D)  MPS tensor
    Output:
        U : (d*D, D')  isometry truncating the Stinespring space
        A_comp : (D', d, D') compressed MPS tensor
    """

    D, d, _ = A.shape

    # Build Stinespring operator V = [A_0; A_1; ...; A_{d-1}]
    Vmat = np.vstack([phys(A, s) for s in range(d)])   # (d*D, D)

    # SVD to get dominant environment isometry
    U, S, Vh = np.linalg.svd(Vmat, full_matrices=False)

    # Choose rank = number of nonzero singular values
    Dp = np.sum(S > 1e-12)
    U_trunc = U[:, :Dp]                   # (d*D, D')

    # Now compress channel:
    # K'_s = U† (A_s ⊗ I) U
    A_comp = np.zeros((Dp, d, Dp), dtype=complex)

    # Prereshape U to (d, D, D')
    U_blocked = U_trunc.reshape(d, D, Dp)

    for s in range(d):
        A_s = phys(A, s)                  # (D, D)
        # (A_s ⊗ I)*U  acts as A_s on the D-index of each block
        A_comp_s = (A_s @ U_blocked[s])   # (D, D') -> reshape -> (Dp)
        # Project with U†:
        A_comp[:, s, :] = U_blocked[s].conj().T @ A_comp_s

    return A_comp, Dp


# ------------------------------------------------------------
# Canonicalize compressed MPS using G A G^{-1}
# ------------------------------------------------------------
def left_canonical(A):
    Dp, d, _ = A.shape

    # Left fixed point of adjoint transfer map
    L = np.eye(Dp, dtype=complex) / Dp

    for _ in range(2000):
        L_new = np.zeros_like(L)
        for s in range(d):
            As = A[:, s, :]
            L_new += As.conj().T @ L @ As
        L_new /= np.trace(L_new)
        if np.linalg.norm(L_new - L) < 1e-12:
            break
        L = L_new

    L = 0.5 * (L + L.conj().T)
    G = sqrt_pos(L)
    G_inv = np.linalg.inv(G)

    A_can = np.zeros_like(A)
    for s in range(d):
        A_can[:, s, :] = G @ A[:, s, :] @ G_inv

    return A_can, G, G_inv


# ------------------------------------------------------------
# Full Stinespring-MPS compressor
# ------------------------------------------------------------
def compress_rho_mps_stinespring(A, L=None):
    """
    Input:
        A : UniformMps tensor  (D,d,D)
    Output:
        A_compress : (D',d,D') canonical compressed tensor
    """

    A_tens = A.tensor
    A_comp, Dp = stinespring_compress(A_tens)

    print("Compressed bond dimension D' =", Dp)

    # Canonicalize
    A_can, G, G_inv = left_canonical(A_comp)

    # Check isometry
    V = A_can.reshape(Dp * A.d, Dp)
    err = np.linalg.norm(V.conj().T @ V - np.eye(Dp))
    print("‖V†V − I‖ =", err)

    return A_can


d=2
D_dim=20


A0=UniformMps.random(D_dim, d)


A_compress_tens = compress_rho_mps_old(A0,3)

evals, Lmats = left_eigensystem(A_compress_tens)

# sort by magnitude descending
idx = np.argsort(-np.abs(evals))
evals = evals[idx]
Lmats = [Lmats[i] for i in idx]

print("Leading eigenvalues:")
print(evals[:10])