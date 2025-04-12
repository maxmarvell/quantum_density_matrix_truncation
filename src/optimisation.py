from uniform_MPS import UniformMPS
from cost import transfer_blocks, cost, compute_contraction
import numpy as np
from ncon import ncon
from scipy.linalg import expm, polar


def derivate(A: UniformMPS, B: UniformMPS, L: int):

    ABdag, BAdag = transfer_blocks(A, B)
    BBdag, _ = transfer_blocks(B, B)

    res = np.zeros_like(B.value)

    for i in range(L):
        res -=  2*ncon(((ABdag ** i).value, (ABdag ** (L-i-1)).value, (BAdag ** L).value, B.r, A.r, A.value), ((1, 2, 7, -1), (8, -3, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6), (7, -2, 8)))

    for i in range(L):
        res +=  ncon(((BBdag ** i).value, (BBdag ** (L-i-1)).value, (BBdag ** L).value, B.r, B.r, B.value), ((1, 2, 7, -1), (8, -3, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6), (7, -2, 8)))
        res +=  ncon(((BBdag ** L).value, (BBdag ** i).value, (BBdag ** (L-i-1)).value, B.r, B.r, B.value), ((1, 2, 3, 4), (2, 1, 7, -1), (8, -3, 5, 6), (5, 4), (3, 6), (7, -2, 8)))

    return UniformMPS(res, normalise = False)


def project_tangent_grassman(B: UniformMPS, D: UniformMPS):
    return D - UniformMPS(ncon((B.value, np.conj(B.value), D.value), ((-1, -2, 3), (1, 2, 3), (1, 2, -3))), normalise=False)


def update_isometry(A: UniformMPS, B:UniformMPS, L:int, alpha: float = .001):
    D = derivate(A, B, L)
    G = project_tangent_grassman(B, D)
    B_prime = grassman_retraction(B.value.reshape(B.bond_dim*B.phys_dim, -1), G.value.reshape(B.bond_dim*B.phys_dim, -1), alpha)
    return B_prime.reshape() 

def simple_retraction(W, G, alpha):
    """
        Perform a simple retraction on isometry W
    """
    return W - G * alpha

def grassman_retraction(W, G, alpha):
    '''
        Peform a geodesic retraction on the Grassmann manifold based on the Euclidean metric.
    '''
    n, m = W.shape
    Q, R = np.linalg.qr((np.eye(n) - W @ W.conj().T) @ G, mode='reduced')
    Zero = np.zeros((m, m))

    a = np.block([W, Q])
    b = np.block([
        [Zero, -alpha*R.conj().T],
        [alpha*R, Zero]
    ])
    b = expm(b)[..., :m]
    return a @ b


def update_II(A: np.ndarray, M: np.ndarray, tol:float = 1e-3, max_iters=1e+5):

    def cost(A, M):
        return ncon((np.conj(A), M), ((1, 2, 3), (1, 2, 3)))
    
    def deriv(M):
        return M
    
    def update(A, M, alpha: float = 1e-4):
        D = deriv(M)
        G = ncon((A, np.conj(A), D), ((-1, -2, 3), (1, 2, 3), (1, 2, -3)))
        return A - alpha * (D - G)

    _A = A.copy()
    i = 0

    print(f"Initial cost: {cost(_A, M)}")

    while np.abs(cost(_A, M)) > tol:
        __A = update(_A, M)
        c1 = cost(_A, M)
        c2 = cost(__A, M)

        if abs(c2) > abs(c1):
            print(f"best convergence is {c1}")
            return __A

        if np.allclose(__A, _A):
            raise RuntimeError(f"not updating at iteration {i}")
        _A = __A
        i += 1
        
        if i > max_iters:
            raise RuntimeError("Did not converge in maximum number of iterations")

    print(f"converged with cost {c2}")
    return _A

if __name__ == "__main__":
    
    Da = 5
    Db = 3
    p = 2
    L = 4

    A = UniformMPS.random(Da, p)
    A.r_dominant()

    B = UniformMPS.random(Db, p)
    B.r_dominant()

    res = np.zeros_like(B.value)
    BBdag, _ = transfer_blocks(B, B)
    for i in range(L):
        res += ncon(((BBdag ** i).value, (BBdag ** (L-i)).value, B.r, B.value), ((1, 1, 5, -1), (6, -3, 3, 4), (3, 4), (5, -2, 6)))
    assert np.allclose(ncon((res, np.conj(B.value)), ((2, 1, 3), (2, 1, 3))) / L, (1+0j))

    ABdag, BAdag = transfer_blocks(A, B)
    assert np.allclose(ncon((derivate(B, B, L).value, np.conj(B.value)), ((2, 1, 3), (2, 1, 3))) / (2*L), cost(B, B, L))

    res = np.zeros_like(B.value)
    for i in range(L):
        res +=  ncon(((ABdag ** i).value, (ABdag ** (L-i-1)).value, (BAdag ** L).value, B.r, A.r, A.value), ((1, 2, 7, -1), (8, -3, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6), (7, -2, 8)))
    assert np.allclose(ncon((res, np.conj(B.value)), ((2, 1, 3), (2, 1, 3))) / L, compute_contraction(ABdag, BAdag, A.r, B.r, L))

    res = np.zeros_like(B.value)
    for i in range(L):
        res +=  ncon(((BBdag ** i).value, (BBdag ** (L-i-1)).value, (BBdag ** L).value, B.r, B.r, B.value), ((1, 2, 7, -1), (8, -3, 3, 4), (2, 1, 5, 6), (5, 4), (3, 6), (7, -2, 8)))
        res +=  ncon(((BBdag ** L).value, (BBdag ** i).value, (BBdag ** (L-i-1)).value, B.r, B.r, B.value), ((1, 2, 3, 4), (2, 1, 7, -1), (8, -3, 5, 6), (5, 4), (3, 6), (7, -2, 8)))
    assert np.allclose(ncon((res, np.conj(B.value)), ((2, 1, 3), (2, 1, 3))) / (2*L), compute_contraction(BBdag, BBdag, B.r, B.r, L))

    B_alpha = update_isometry(A, B, L, alpha=0.001)
    print(ncon((B_alpha.value, np.conj(B.value)), ((1, 2, -2), (1, 2, -1))))
    assert np.allclose(ncon((B_alpha.value, np.conj(B.value)), ((1, 2, -2), (1, 2, -1))), np.eye(Db, Db))

    M = np.random.rand(*A.value.shape) + 1j * np.random.rand(*A.value.shape)
    U, _, V = np.linalg.svd(M, full_matrices=True)

    _A = update_II(A.value, M)

    # print(np.allclose(_A.reshape((A.bond_dim*A.phys_dim, A.bond_dim))), ncon((U, V), ((-1, 1), (1, -2))))

    # print(ncon((B_alpha.value, np.conj(B_alpha.value)), ((1, 2, -2), (1, 2, -1))))