from examples.data_management import load_data, unfold_data, load_unfolded_data, sort_and_save_dataset, extend_unfolded_data
from examples.analyze_first import plot_unfolded_twofields, plot_over_time
import numpy as np
import matplotlib.pyplot as plt
from qdmt.analysis.tools import *

from scipy.integrate import quad
from scipy.optimize import least_squares

from pathlib import Path

import sys

def is_proportional_to_identity(X, tol=1e-10):
    """
    Checks whether X ≈ c I for some scalar c (complex allowed).
    Returns (ok, c, rel_err).
    """
    X = np.asarray(X)
    D1, D2 = X.shape
    assert D1 == D2
    D = D1

    c = np.trace(X) / D
    resid = X - c * np.eye(D, dtype=np.complex128)

    # scale-free error
    denom = max(np.linalg.norm(X.ravel()), 1e-300)
    rel_err = np.linalg.norm(resid.ravel()) / denom

    # print(f" c = {c}   rel_err = {rel_err:.3e}")
    return rel_err < tol, c, rel_err


def canonical_ok(A_obj, tol=1e-10):
    lf = TransferMatrix.new(A_obj, A_obj).left_fixed_point().tensor
    ok, c, rel = is_proportional_to_identity(lf, tol=tol)
    return ok, rel, lf

def gauge_transform_tensor(A, U):
    # A: (D,d,D), U: (D,D)
    A = np.asarray(A)
    D1, d, D2 = A.shape
    if D1 != D2:
        raise ValueError(f"Need square bonds, got {A.shape}")
    D = D1
    U = np.asarray(U)
    if U.shape != (D, D):
        raise ValueError(f"U shape {U.shape} incompatible with D={D}")
    Udag = U.conj().T

    A2 = np.empty_like(A, dtype=np.complex128)
    for i in range(d):
        A2[:, i, :] = U @ A[:, i, :] @ Udag
    return A2


def diagonalize_right_fp_gauge(A_obj, tol=1e-10):
    ok, err, _ = canonical_ok(A_obj, tol=tol)
    if not ok:
        raise ValueError(f"Tensor is not left-canonical (error={err}).")

   
    rf = TransferMatrix.new(A_obj, A_obj).right_fixed_point()
    R = np.asarray(rf.tensor)
    R = 0.5 * (R + R.conj().T)
    R = R / np.trace(R)

    lam, V = np.linalg.eigh(R)   # R = V diag(lam) V†
    U = V.conj().T

    A_new = gauge_transform_tensor(A_obj.tensor, U)
    return UniformMps(A_new), lam, U



def diagonalize_right_fp_gauge_continuous(A_obj, V_prev=None, tol=1e-10, gap_tol=1e-8):
    """
    Continuous gauge fixing for D=2.

    - Diagonalizes right fixed point R = V diag(lam) V†
    - Orders lam descending
    - If V_prev is provided:
        (i) choose best permutation (swap columns) by overlap with V_prev
        (ii) Procrustes align within the subspace: V <- V W with W unitary s.t. V W ~ V_prev
        (iii) if gap is tiny, freeze: V <- V_prev (avoids drift/branch noise)

    Returns: (A_new_mps, lam, U, V) with U = V†.
    """
    ok, err, _ = canonical_ok(A_obj, tol=tol)
    if not ok:
        raise ValueError(f"Tensor is not left-canonical (error={err}).")

    rf = TransferMatrix.new(A_obj, A_obj).right_fixed_point()
    R = np.asarray(rf.tensor)
    R = 0.5 * (R + R.conj().T)
    R = R / np.trace(R)

    lam, V = np.linalg.eigh(R)  # ascending
    idx = np.argsort(lam)[::-1]  # descending
    # lam = lam[idx]
    V = V[:, idx]

    gap = float(abs(lam[0] - lam[1]))

    if V_prev is not None:
        # If nearly degenerate, don't trust eigenvectors: keep previous basis
        if gap < gap_tol:
            V = V_prev.copy()
        else:
            # 1) pick best permutation
            V0 = V
            V1 = V[:, ::-1]
            score0 = np.abs(np.trace(V_prev.conj().T @ V0))
            score1 = np.abs(np.trace(V_prev.conj().T @ V1))
            V = V0 if score0 >= score1 else V1

            # 2) Procrustes: find W that best aligns V to V_prev
            # Solve min_{W unitary} || V W - V_prev ||_F
            M = V.conj().T @ V_prev
            U1, _, Vh = np.linalg.svd(M)
            W = U1 @ Vh
            V = V @ W

        # 3) Optional: stabilize column phases against V_prev (after Procrustes/freeze)
        for j in range(V.shape[1]):
            ov = np.vdot(V_prev[:, j], V[:, j])
            if np.abs(ov) > 0:
                V[:, j] *= ov / np.abs(ov)

    # Final stable phase convention: make V[0,j] real positive
    for j in range(V.shape[1]):
        phase = V[0, j] / (np.abs(V[0, j]) + 1e-300)
        V[:, j] /= phase

    U = V.conj().T
    A_new = gauge_transform_tensor(A_obj.tensor, U)
    return UniformMps(A_new), lam, U, V


# def diagonalize_right_fp_gauge(A_obj, tol=1e-10):
#     ok, err, _ = canonical_ok(A_obj, tol=tol)
#     if not ok:
#         raise ValueError(f"Tensor is not left-canonical (error={err}).")

#     rf = TransferMatrix.new(A_obj, A_obj).right_fixed_point()
#     R = np.asarray(rf.tensor)
#     R = 0.5 * (R + R.conj().T)
#     R = R / np.trace(R)

#     lam, V = np.linalg.eigh(R)   # ascending by default

#     # sort so lam[0] >= lam[1]
#     idx = np.argsort(lam)[::-1]
#     lam = lam[idx]
#     V = V[:, idx]

#     U = V.conj().T

#     A_new = gauge_transform_tensor(A_obj.tensor, U)
#     return UniformMps(A_new), lam, U



#   unitary construciton



def Rx(theta):
    c = np.cos(theta/2)
    s = np.sin(theta/2)
    return np.array([[c, -1j*s],
                     [-1j*s, c]], dtype=np.complex128)

def Rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=np.complex128)

def CNOT():
    # control = first (top), target = second (bottom)
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)



I2 = np.eye(2, dtype=np.complex128)


def U(phi1, psi1, phi2, psi2, phi3, psi3, phi4, psi4, phi5, psi5, phi6, psi6, psi7):
    """
    Builds 4x4 unitary with first 3 layers:
    
    Layer 1: Rx(phi1) ⊗ Rx(psi1)
    Layer 2: Rz(phi2) ⊗ Rz(psi2)
    Layer 3: I ⊗ Rx(psi3)
    """


    # Layer 1
    L1 = np.kron(Rx(phi1), Rx(psi1))

    # Layer 2
    L2 = np.kron(Rz(phi2), Rz(psi2))

    # Layer 3
    L3 = np.kron(I2, Rx(psi3))

    L4 = CNOT()

    L5=np.kron(Rx(phi3),Rz(psi4))


    L6=np.kron(Rz(phi4),Rx(psi5))

    L7=CNOT()

    L8 = np.kron(Rx(phi5), Rz(psi6))
    L9 = np.kron(Rz(phi6), Rx(psi7))

    return L9 @ L8 @ L7 @ L6 @ L5 @ L4 @ L3 @ L2 @ L1


def W(phi1, psi1, phi2, psi2, phi3, psi3, phi4, psi4, phi5, psi5, phi6, psi6, psi7):
    Umat = U(phi1, psi1, phi2, psi2, phi3, psi3, phi4, psi4, phi5, psi5, phi6, psi6, psi7)
    return Umat[:, :2]   # ancilla (phi strand) fixed to |0>, psi in {0,1}

def W_target_from_A(A_tensor):
    # A_tensor: (phi_out=DL, psi_out=d, psi_in=DR)
    # rows must be (phi_out, psi_out) to match U's basis
    DL, d, DR = A_tensor.shape
    return A_tensor.reshape(DL*d, DR, order="C")


def W_model(params):
    Umat = U(*params)
    return Umat[:, :2]  # cols = ancilla(phi_in)=0, psi_in=0/1



def A_from_params(params):
    Wm = W_model(params)                 # rows (phi_out, psi_out)
    return Wm.reshape(2,2,2, order="C")  # (DL,d,DR)


fixed_slightly_snug2 = {"phi4": 0.0, "phi5" : 0.0, "psi2" :  np.pi/2, "psi1": -1*np.pi/2, "psi5" : 0.0, "psi6": -np.pi/2, "psi4" : -np.pi/2}


def A_from_snug_params(params):
    Umat = U_reduced(*params)
    Wm = Umat[:, :2]                # rows (phi_out, psi_out)
    return Wm.reshape(2,2,2, order="C")  # (DL,d,DR)

def U_reduced(*params):
    if len(params)!=6:
        print("wrong number of parameters")
    else:    
        phi1=params[0]
        psi1=-1*np.pi/2
    
        phi2=params[1]
        psi2=np.pi/2

        phi3=params[2]
        psi3=params[3]

        phi4=0
        psi4=-1*np.pi/2

        phi5=0
        psi5=0 
        
        phi6=params[4]
        psi6=-1*np.pi/2

        psi7=params[5]
        return U(phi1, psi1, phi2, psi2, phi3, psi3, phi4, psi4, phi5, psi5, phi6, psi6, psi7)




def residual(params, A_tensor):
    Wt = W_target_from_A(A_tensor)
    Wm = W_model(params)

    # align global phase: minimize ||Wm - e^{iφ} Wt||
    inner = np.vdot(Wt, Wm)  # sum conj(Wt)*Wm
    phase = inner / (abs(inner) + 1e-300)
    R = (Wm - phase * Wt).ravel()
    return np.concatenate([R.real, R.imag])

def residual_no_phase(params, A_tensor):
    Wt = W_target_from_A(A_tensor)
    Wm = W_model(params)
    R = (Wm - Wt).ravel()
    return np.concatenate([R.real, R.imag])



def fit_params(A_tensor, x0=None, max_nfev=20000, verbose=2):
    if x0 is None:
        x0 = np.zeros(13)

    # optional: keep angles bounded
    lb = -2*np.pi * np.ones(13)
    ub =  2*np.pi * np.ones(13)

    res = least_squares(
        residual_no_phase, x0,
        args=(A_tensor,),
        bounds=(lb, ub),
        max_nfev=max_nfev,
        verbose=verbose
    )
    return res

def fit_fixed(A_tensor, fixed=None, x0=None, prior_strength=0.0):
    fixed = {} if fixed is None else dict(fixed)
    fixed = {NAME2IDX.get(k,k): v for k,v in fixed.items()}
    free = [i for i in range(N) if i not in fixed]
    k = len(free)

    def pack(pfree):
        full = np.zeros(N)
        for i,v in fixed.items(): full[i] = v
        for i,v in zip(free, pfree): full[i] = v
        return full

    def fun(pfree):
        r = residual_no_phase(pack(pfree), A_tensor)  # or residual(...) if you prefer
        if prior_strength > 0 and x0 is not None:
            r_prior = np.sqrt(prior_strength) * (pfree - x0)
            return np.concatenate([r, r_prior])
        return r

    if x0 is None:
        x0 = np.zeros(k)

    lb, ub = -2*np.pi*np.ones(k), 2*np.pi*np.ones(k)
    res = least_squares(fun, x0, bounds=(lb, ub), max_nfev=20000, verbose=0)
    res.full_params = pack(res.x)
    res.free_idx = free

    # print(res.full_params)
    return res



TWOPI = 2*np.pi

def wrap_0_2pi(x):
    return np.mod(x, TWOPI)


def run_all_As_warmstart(As, filename="snug_results_warm.npz", allowed_tolerance=1e-10, fixed = None, verbose=0,prior_strength=1e-5):
    params_full_list = []
    params_free_list = []
    costs = []
    lambdas = []
    dists = []

    x0_free = None
    V_prev = None            # NEW

    for i, A in enumerate(As):
        print(f"=== {i}/{len(As)-1} ===")

        p_full, p_free, c, lam, V_prev, dist = gauge_fix_and_find_parametrization(
            A,
            x0_free=x0_free,
            allowed_tolerance=allowed_tolerance,
            fixed=fixed,
            verbose=verbose,
            prior_strength=prior_strength,
            V_prev=V_prev,     # NEW
        )

        params_full_list.append(p_full)
        params_free_list.append(p_free)
        costs.append(c)
        lambdas.append(lam)
        dists.append(dist)

        # warm-start next step, wrapped to [0, 2pi)
        x0_free = p_free

    data = {
        "params_full": np.array(params_full_list),
        "params_free": np.array(params_free_list),
        "costs": np.array(costs),
        "lambdas": np.array(lambdas),
        "dists": np.array(dists),

    }
    np.savez(filename, **data)
    print("Saved:", filename)
    return data






def tensor_distance(params_full, A_target, phase_align=False):
    A_fit = A_from_params(params_full)
    if phase_align:
        inner = np.vdot(A_target, A_fit)
        c = inner / (abs(inner) + 1e-300)
        diff = A_fit - c * A_target
    else:
        diff = A_fit - A_target
    return float(np.linalg.norm(diff.ravel()))



def gauge_fix_and_find_parametrization(
    A_before,
    x0_free=None,
    allowed_tolerance=1e-10,
    fixed=None,
    verbose=0,
    prior_strength=0.0,
    V_prev=None,           # NEW
    tol=1e-10
):
    A_before_mps = UniformMps(A_before)

    # continuous gauge fix (NEW)
    # A_after_mps, lam, U, V = diagonalize_right_fp_gauge(
    #     A_before_mps, V_prev=V_prev, tol=tol
    # )
    A_after_mps, lam, U, V= diagonalize_right_fp_gauge_continuous(
        A_before_mps, V_prev=V_prev, tol=tol
    )


    A_target = A_after_mps.tensor
    # A_target = A_before_mps.tensor
    

    res = fit_fixed(
        A_target,
        fixed=fixed,
        x0=x0_free,
        prior_strength=prior_strength
    )

    # if you're using "pure cost" elsewhere, keep that logic; otherwise:
    cost = float(res.cost)

    if cost > allowed_tolerance:
        print(f"⚠️  WARNING: cost {cost:.3e} exceeds tolerance")

    dist=tensor_distance(res.full_params, A_target)    
    print(dist)
    # compare RDM
    A_params_mps=UniformMps(A_from_params(res.full_params))
    hs=HS_distance_L_mps(A_params_mps,A_before_mps,3)
    print(f"hs = {hs}")
    dist=hs
    print(res.full_params)
    return res.full_params, res.x, cost, lam, V, dist   # NEW: return V





# fixed_snug = {"phi4": 0.0, "phi5": 0.0, "psi5": 0.0}
# # fixed_snug = {"phi4": 0.0, "psi5": 0.0}
# fixed_snug = {"phi4": 0.0, "psi5": 0.0}

if __name__ == "__main__":

    N = 13
    NAME2IDX = {"phi1":0,"psi1":1,"phi2":2,"psi2":3,"phi3":4,"psi3":5,"phi4":6,
                "psi4":7,"phi5":8,"psi5":9,"phi6":10,"psi6":11,"psi7":12}




    # def A_from_W(W):
    #     # W: rows (phi_out, psi_out), cols (psi_in)
    #     return W.reshape(2,2,2, order="C")  # (DL,d,DR)


    # # --- load data as you already do ---
    # qm_dat_odd_even_average_raw = np.load(
    #     '/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/pre_split/benchmark_dt=0.001_steps=1999_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=-1).npz'
    # )
    # data_old = qm_dat_odd_even_average_raw


    # qm_dat_odd_even_average_split_raw = np.load(
    #     '/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.001_steps=1999_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=-1).npz'
    # )
    # data = qm_dat_odd_even_average_split_raw


    # qm_dat_unfolded_odd_only_avg_less_big = np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.1_steps=19_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=-1).npz')


    qm_dat_big_trotter_sym_raw_=np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.1_steps=19_tol=1e-12_it=10000_D=2_cut=3600(_Trotter_order=0).npz')

    qm_dat_huge_trotter_sym_raw_=np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM_debug/benchmark_dt=0.2_steps=9_tol=1e-14_it=10000_D=2_cut=3600(_Trotter_order=-1).npz')
    # dt=0.1 run, symmetric hamiltonian with averaged calcualtion. inject initial state as well and set cost to zero
    dat_sym_=qm_dat_huge_trotter_sym_raw_




    As = dat_sym_["state"]         # shape (N, 2, 2, 2)
    times = dat_sym_["time"]       # shape (N,)
    costs = dat_sym_["cost"]       # shape (N,)


 

    

    A0_tens = np.load("data/ground_state/tfim_AL_D2_g1.5.npz")["A"]
    A0 = UniformMps(A0_tens)
    assert A0.is_isometry()


    

    # prepend time
    times_full = np.concatenate(
        ([0.0], times)
    )

    # prepend cost (optional)
    costs_full = np.concatenate(
        ([0.0], costs)
    )

    print(As[0])
    print(A0_tens)

     # prepend cost (optional)
    As_full = np.concatenate(
        ([A0_tens], As)
    )
    



    def compute_loschmidt_parameter(parameters):
        n=len(parameters)
        p0=parameters[0]
        A0=UniformMps(A_from_snug_params(p0))  
        results = np.zeros(n)

        for i in range(n):
            p=parameters[i]
            A=UniformMps(A_from_snug_params(p))  
            results[i] = loschmidt_rate_per_site(A.tensor,A0.tensor) 
        return results 



    assert As_full.shape[0] == times_full.shape[0]
    print(As_full[0])
    print(A0_tens)




    np.savez(
        "sym3huge.npz",
        state=As_full,
        time=times_full,
        cost=costs_full,
        dt=0.1,
        description="Symmetric Hamiltonian, Trotter, includes injected A0 at t=0"
    )



    fixed_min= {"psi5": 0.0, "psi6" :  np.pi/2}
    min_labels=["phi1","psi1","phi2","psi2","phi3","psi3","phi4","psi4","phi5","phi6" ,"psi7"]


    fixed_super_snug = {"phi4": 0.0, "psi5": 0.0, "psi1": np.pi/2, "psi2" :  np.pi/2,  "psi4" :  np.pi/2,  "psi6" :  np.pi/2, "phi5" : 0.0}


    # fixed_super_snug = {"phi4": 0.0, "psi5": 0.0, "phi5" : 0.0, "psi2" :  np.pi/2,"psi4" :  np.pi/2, "psi6" :  np.pi/2,"psi1": -1*np.pi/2}
    super_snug_labels = ["phi1","phi2","phi3","psi3","phi6","psi7"]

    fixed_less_snug = {"phi4": 0.0, "phi5" : 0.0, "psi2" :  np.pi/2, "psi1": -1*np.pi/2}
    less_snug_labels = ["phi1","phi2","phi3","psi3","psi4","psi5","phi6","psi6" ,"psi7"]


    fixed_slightly_less_snug = {"phi4": 0.0, "phi5" : 0.0, "psi2" :  np.pi/2, "psi1": -1*np.pi/2, "psi5" : 0.0}
    slightly_less_snug_labels = ["phi1","phi2","phi3","psi3","psi4","phi6","psi6" ,"psi7"]

    fixed_slightly_snug = {"phi4": 0.0, "phi5" : 0.0, "psi2" :  np.pi/2, "psi1": -1*np.pi/2, "psi5" : 0.0, "psi6": -np.pi/2}
    slightly_snug_labels = ["phi1","phi2","phi3","psi3","psi4","phi6", "psi7"]

    fixed_slightly_snug2 = {"phi4": 0.0, "phi5" : 0.0, "psi2" :  np.pi/2, "psi1": -1*np.pi/2, "psi5" : 0.0, "psi6": -np.pi/2, "psi4" : -np.pi/2}
    slightly_snug_labels2 = ["phi1","phi2","phi3","psi3","phi6", "psi7"]


    # less_snug_labels = ["phi1","psi1","phi2","psi2","phi3","psi3","phi4","psi4","phi5","phi6" ,"psi7"]


    # less_snug_labels = ["psi1","phi2","psi2","phi3","psi3","phi4","psi4","phi5","phi6","psi6" ,"psi7"]

    labels=min_labels 
    fix=fixed_min


    labels=slightly_snug_labels2
    labels=["1","2","3","4","5","6","7","8","9","10","","","","","","","","","","","","","","","","",""]
    fix=fixed_slightly_snug2


    # for onemore in ["phi6"]:
    # # ["phi6", "phi2",  "phi1"]:
    #     for val in [1]:
    #     # [0.25*i for i in range(-15,5)]:


    #         fix={  "psi6":0.5*np.pi ,"phi5": 0.0 ,"psi4":0.5*np.pi, "psi5": 0.0, "psi1": 0.5*np.pi, "phi4": 0.0,  "psi2": -0.5*np.pi}
    #         fix={ "psi6":0.5*np.pi ,"phi5": 0.0 ,"psi4":0.5*np.pi, "psi5": 0.0, "psi1": 0.5*np.pi, "phi4": 0.0,  "psi2": -0.5*np.pi}
    #         # "phi3":0.5, 
    
    #         # {phi1, phi2, phi3, psi3, phi6, psi7}
    #         # "phi6": 1.4,

    #         filename = "/Users/phys2259/Documents/qdmt/super_snug_results_warm_huge_sym.npz"
    #         run_all_As_warmstart(As_full, filename=filename, allowed_tolerance=1e-10, fixed=fixed_slightly_snug2, prior_strength=0.5*1e-2)

            


    #         data = np.load(filename, allow_pickle=True)
    #         costs = data["dists"]
    #         params = data["params_free"]   # shape (N, P)
    #         full_params = data["params_full"]


    #         A_params = [A_from_params(x) for x in full_params]
    #         lsparam=compute_loschmidt(A_params)
    #         print(params)

    #         print(As_full)

    #         print("Loaded:", filename)
    #         print("costs shape:", costs.shape, "params shape:", params.shape)
    #         print("cost min/median/max:", float(costs.min()), float(np.median(costs)), float(costs.max()))

    #         # --- Plot 1: cost ---
    #         plt.figure()
    #         plt.plot(costs)
    #         plt.yscale("log")  # costs span many orders usually
    #         plt.xlabel("Index")
    #         plt.ylabel("Cost (log scale)")
    #         plt.title("Fit cost over sequence")
    #         plt.grid(True, which="both")
    #         plt.show()


    #         plt.figure()

    #         ls=compute_loschmidt(As_full)
    #         print(ls)
    #         plt.plot(lsparam)


    #         plt.plot(ls)
    #         # plt.yscale("log")  # costs span many orders usually
    #         plt.xlabel("time")
    #         plt.ylabel("LS")
    #         plt.title(onemore+str(val)+" LS")
    #         plt.grid(True, which="both")
    #         plt.show()
            


    #         plt.figure()
    #         for j in range(params.shape[1]):
    #             plt.plot(params[:, j],label=labels[j])
    #         plt.legend(ncol=3, fontsize=8)
    #         plt.grid(True)
    #         plt.title("Snug parameters")
    #         plt.show()


    # --- Plot 2: all parameters mod 2π ---

    def branch_fix_2pi(params):
        """
        params: (N, P) angles (rad), any range
        returns: (N, P) angles adjusted by +2π*k each step to minimize jumps
        """
        params = np.asarray(params, float)
        N, P = params.shape
        out = np.empty_like(params)
        out[0] = params[0]

        twopi = 2*np.pi
        for t in range(1, N):
            delta = params[t] - out[t-1]
            # choose integer k to bring delta into (-pi, pi]
            k = np.round(delta / twopi)
            out[t] = params[t] - k * twopi
        return out

    # params_smooth = branch_fix_2pi(params)




    # lams = data["lambdas"]
    # plt.figure()

    # plt.plot(lams[:, 0], label="lambda1")
    # plt.plot(lams[:, 1], label="lambda2")
    # plt.legend()
    # plt.title("Right fixed point eigenvalues")
    # plt.show()



    for A in As_full:
        print( A_from_snug_params)


# PLOT

    # ------------------------------------------------------------
    # Load exact MPS data
    # ------------------------------------------------------------
    qm_dat_huge_trotter_sym_raw_ = np.load(
        "/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM_debug/"
        "benchmark_dt=0.2_steps=9_tol=1e-14_it=10000_D=2_cut=3600(_Trotter_order=-1).npz"
    )

    dat_sym_ = qm_dat_huge_trotter_sym_raw_

    As = dat_sym_["state"]
    times = dat_sym_["time"]
    costs = dat_sym_["cost"]

    A0_tens = np.load("data/ground_state/tfim_AL_D2_g1.5.npz")["A"]

    As_full = np.concatenate(([A0_tens], As))
    times_full = np.concatenate(([0.0], times))

    # ------------------------------------------------------------
    # Load already fitted 6-parameter snug parametrization
    # ------------------------------------------------------------
    filename = "/Users/phys2259/Documents/qdmt/super_snug_results_warm_huge_sym.npz"
    data = np.load(filename, allow_pickle=True)

    params_free = data["params_free"]      # shape (N, 6)
    params_full = data["params_full"]      # shape (N, 13), optional
    dists = data["dists"]

    print("Loaded:", filename)
    print("params_free shape:", params_free.shape)
    print("dists min/median/max:",
          float(dists.min()),
          float(np.median(dists)),
          float(dists.max()))

    # ------------------------------------------------------------
    # Reconstruct MPS tensors from the 6 free parameters
    # ------------------------------------------------------------
    A_param = np.array([A_from_snug_params(p) for p in params_free])

    # Alternative, if you trust params_full more:
    # A_param = np.array([A_from_params(p) for p in params_full])

    # ------------------------------------------------------------
    # Compute Loschmidt echoes / rates
    # ------------------------------------------------------------
    ls_exact = compute_loschmidt(As_full)
    ls_param = compute_loschmidt(A_param)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(times_full, ls_exact, "o-", label="exact MPS")
    plt.plot(times_full, ls_param, "x--", label="6-param reconstruction")

    plt.xlabel("time")
    plt.ylabel("Loschmidt echo / rate")
    plt.title("Loschmidt comparison: exact vs 6-param snug")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional error plot
    plt.figure(figsize=(7, 4))
    plt.plot(times_full, np.abs(ls_param - ls_exact), "o-")
    plt.xlabel("time")
    plt.ylabel("|difference|")
    plt.yscale("log")
    plt.title("Loschmidt reconstruction error")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()