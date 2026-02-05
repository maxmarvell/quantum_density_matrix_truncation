from qdmt.uniform_mps import UniformMps
from qdmt.model import AbstractModel, TransverseFieldIsing
from qdmt.cost import EvolvedHilbertSchmidt, NaiveEvolvedHilbertSchmidt as Naive
from qdmt.optimisation import ConjugateGradient
from qdmt.manifold import Grassmann

import time
import numpy as np
import sys
from numpy.lib.npyio import NpzFile
import argparse
import os
from pathlib import Path


# import warnings
# warnings.simplefilter("error", RuntimeWarning)


def check_transfer_normalization(A):
    D, d, _ = A.shape
    # construct left and right "Gram" matrices
    G_left  = sum(A[:, s, :] @ A[:, s, :].conj().T for s in range(d))
    G_right = sum(A[:, s, :].conj().T @ A[:, s, :] for s in range(d))
    wL = np.linalg.eigvalsh(G_left)
    wR = np.linalg.eigvalsh(G_right)
    print("‖∑ A A† - I‖ =", np.linalg.norm(G_left - np.eye(D)))
    print("‖∑ A† A - I‖ =", np.linalg.norm(G_right - np.eye(D)))
    print("eigvals left  =", np.round(wL, 6))
    print("eigvals right =", np.round(wR, 6))



def enlarge_left_canonical_uniformmps(umps, D1: int, eps: float = 0.0) -> np.ndarray:
    """
    Enlarge `umps` to bond dim D1 >= umps.D and return a left-canonical tensor A
    with shape (D1, umps.d, D1):  sum_s A^s A^{s†} = I.
    """
    # Get tensor via class' accessor / reshape (matches your class)
    t = getattr(umps, "tensor")
    A0 = t() if callable(t) else t
    if A0 is None:
        d, D0 = umps.d, umps.D
        order = "C" if umps.V.flags.c_contiguous else ("F" if umps.V.flags.f_contiguous else "C")
        A0 = umps.V.reshape(D0, d, D0, order=order)
    D0, d, _ = A0.shape
    if D1 < D0:
        raise ValueError(f"D1 ({D1}) must be >= current D ({D0})")

    # Zero-change embed: A <- W A0 W†
    if D1 == D0:
        A = A0.copy()
    else:
        W = np.zeros((D1, D0), dtype=A0.dtype)
        W[:D0, :D0] = np.eye(D0, dtype=A0.dtype)
        Wdag = W.conj().T
        A = np.zeros((D1, d, D1), dtype=A0.dtype)
        for s in range(d):
            A[:, s, :] = W @ A0[:, s, :] @ Wdag

    # Tiny noise only in the new subspace (optional)
    if eps and eps > 0.0 and D1 > D0:
        if np.iscomplexobj(A):
            noise = np.random.randn(*A.shape) + 1j*np.random.randn(*A.shape)
        else:
            noise = np.random.randn(*A.shape)
        mask = np.zeros((D1, D1), dtype=bool)
        mask[D0:, :] = True
        mask[:, D0:] = True
        M = np.zeros_like(A, dtype=bool)
        for s in range(d):
            M[:, s, :] = mask
        A = A + eps * (noise * M)

    # RIGHT-canonical via QR on the right unfolding
    # Right unfolding: M = [A^1; A^2; ...; A^d] ∈ C^{(d*D1) × D1}
    M = np.concatenate([A[:, s, :] for s in range(d)], axis=0)   # (d*D1, D1)

    # QR of M: M = Q R, with Q ∈ C^{(d*D1) × D1}, Q†Q = I
    Q, R = np.linalg.qr(M, mode='reduced')

    # Split Q back into d blocks along the first axis
    A_rc = np.empty_like(A)
    for s in range(d):
        A_rc[:, s, :] = Q[s*D1:(s+1)*D1, :]

    # (optional) renormalize phases using det(R)/|det(R)| if you want consistency


    return A_rc


def check_left(A):
    D, d, _ = A.shape
    G = sum(A[:, s, :] @ A[:, s, :].conj().T for s in range(d))
    print("‖Σ_s A A† − I‖ =", np.linalg.norm(G - np.eye(D)))


def evolve(A0: UniformMps, 
           D: int,
           L: int, 
           model: AbstractModel, 
           delta_t: float, 
           max_t: float, 
           max_iter: int, 
           tol: float,
           cut_off: float,
           start_t: float = 0.0,
           trotterization_order: int = 2):

    # print("START")
    # print(cut_off)
    # print("START2")
    #  timing 
    start_time = time.time()

    print(model)

    d = A0.d
    if A0.D == D:
        A = A0
        print('no embedding')
    else:        
        Atens = enlarge_left_canonical_uniformmps(A0, D, eps=1e-8) 
        A = UniformMps(Atens, tol=1e-10)
        # A = UniformMps.random(D, d)
        # check_left(Atens)
        # check_transfer_normalization(Atens)
        # # print('yes')
        

        # print(A.is_isometry(1e-10))
        # print('we start now')
  

    times = np.arange(start_t + delta_t, max_t + delta_t/2, delta_t)

    # print(delta_t)
    cost = np.empty_like(times)
    norm = np.empty_like(times)
    duration = np.empty_like(times)
    maxiter = np.empty_like(times)
    state = np.empty((len(times), *A.tensor.shape), dtype=np.complex128)
    # print("do GM")
    M = Grassmann()
    # print("done GM")
    for i, t in enumerate(times):
        
        time_at_start_of_step=time.time()
        if (D**8*d**7*np.log(L) < d**(2*L)*L*D**2*d and D**6*d**6*np.log(L) < d**(2*L)):
            # print('do evolved')
            f = EvolvedHilbertSchmidt(A0, model, L, trotterization_order)
        else:
            f = Naive(A0, model, L, trotterization_order)
            # print('do naive?')
        # print('start gd')    
        gd = ConjugateGradient(f, M, A, max_iter, tol=tol, verbose=True)
        
        A, cost[i], norm[i], hist = gd.optimize()
        maxiter[i] = len(hist[0])-1

        time_at_end_of_step=time.time()
        # sys.exit()
        state[i] = A.tensor
        A0 = A

        duration[i]=-time_at_start_of_step+time_at_end_of_step
        print(duration)

        # check time
        if time.time() - start_time > cut_off:
            print(f"Time cut-off ({cut_off}s)")
            break

        print(f"\nEvolved the state to t={t}\n\n")
        # print(cost)

    return times, state, cost, norm, duration, maxiter

def parse(parser: argparse.ArgumentParser):
    parser.add_argument(
        'initial',
        type=str,
        help='Filepath to the initial state'
    )
    parser.add_argument(
        'savefile',
        type=str,
        help='Location to save the file'
    )
    parser.add_argument(
        '--max-time',
        type=float,
        help='Time to run the simulation until'
    )
    parser.add_argument(
        '-L',
        type=int,
        default=4,
        help='Length of the patch (default: 4)'
    )
    parser.add_argument(
        '-g',
        type=float,
        default=0.2,
        help='Hamiltonian Ising coupling strength (default: 0.2).'
    )
    parser.add_argument(
        '-D',
        type=int,
        default=4,
        help='Virtual dimension (default: 4).'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Learning rate for optimization (default: 0.1).'
    )
    parser.add_argument(
        '--max-iters',
        type=int,
        default=10000,
        help='Number of optimization steps (default: 10000).'
    )
    parser.add_argument(
        '--tol',
        type=float,
        default=1e-6,
        help='Optimization tolerance at each time step (default: 1e-6)'
    )
    parser.add_argument(
        '--delta-t',
        type=float,
        default=0.1,
        help='Trotterized time step (default: 0.1)'
    )
    return parser.parse_args()

def check_write_permission(filepath: str) -> bool:
    output_path = Path(filepath)
    output_dir = output_path.parent

    if output_dir and not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error: Cannot create directory '{output_dir}'. Permission denied or invalid path: {e}")
            return False

    if output_dir and not os.access(output_dir, os.W_OK):
        print(f"Error: Directory '{output_dir}' is not writable. Check permissions.")
        return False
        
    try:
        with open(filepath, 'w') as f:
            pass
        os.remove(filepath)
        # print(f"Successfully checked write permission for: {filepath}")
        return True
    except OSError as e:
        print(f"Error: Cannot write to '{filepath}'. Permission denied or invalid path: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while checking write permission: {e}")
        return False

def load_state(filepath: str):
    try:

        raw_data = np.load(filepath)

        if isinstance(raw_data, np.ndarray):
            print(f"Loaded a single NumPy array from: {filepath}")
            previous_data = {
                "time": [0],
                "state": [raw_data],
                "gradient_norm": [np.nan],
                "cost": [np.nan]
            }
            return UniformMps(raw_data), previous_data, 0
        
        elif isinstance(raw_data, NpzFile):

            with np.load(filepath) as data:

                while True :
                    try:
                        choice = float(input("What time will you start from? "))
                    except:
                        print("Was unable to parse value as number please try again.\n\n")
                        continue

                    idx = np.searchsorted(data['time'], choice, side='right')
                    start_time = data['time'][idx - 1]

                    choice = input(f"The closest start time is {start_time}? if this is correct type y: ").lower()
                    if choice != 'y':
                        continue

                    filtered_data = {}
                    for key in data.files:
                        filtered_data[key] = data[key][:idx]

                    print(f"Successfully loaded and filtered data from {filepath} up to time {start_time}.")
                    return UniformMps(data['state'][idx-1]), filtered_data, start_time
                
        else:
            print("Error: Unhandled file type")
            return None, None

    except FileNotFoundError:
        print(f"Error: File not found at location {filepath}.")
        return None, None
    except OSError as e:
        print(f"Error: Unable to read file at location {filepath}. Details: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return None, None

def run_simulation():

    parser = argparse.ArgumentParser(description="Evolve the state from an initial state under a NN transverse-field Hamiltonian.")
    args = parse(parser)

    start_time = 0.0
    model = TransverseFieldIsing(args.g, args.delta_t)

    # get initial state
    A, previous_data, start_time = load_state(args.initial)

    savefile = args.savefile
    while not check_write_permission(savefile):
        savefile = input("Choose another location to save the file: ")

    times, state, cost, norm = evolve(A, args.L, model, args.delta_t, args.max_time, args.max_iters, args.tol, start_t=start_time)

    if previous_data:
        times = np.concatenate((previous_data['time'], times))
        state = np.concatenate((previous_data['state'], state))
        cost = np.concatenate((previous_data['cost'], cost))
        norm = np.concatenate((previous_data['gradient_norm'], norm))

    np.savez_compressed(savefile,
                        time=times,
                        state=state,
                        gradient_norm=norm,
                        cost=cost)

if __name__ == "__main__":

    from qdmt.model import TransverseFieldIsing
    from qdmt.cost import EvolvedHilbertSchmidt
    from qdmt.manifold import Grassmann

    theta = phi = np.pi / 2

    psi = np.array([np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)])

    A = UniformMps(psi.reshape(1, 2, 1))

    filepath = 'data/non_integrable/bond_dimension_8_patch_4_Y_2'
    assert check_write_permission(filepath)


    dtt=0.001

    model = TransverseFieldIsing(g=1.05, delta_t=dtt, h=-0.5, J=-1)

   
    summary=[] 
    for D in [4]:
        print(D)
       
        t0 = time.time()
        times, state, cost, norm, _ = evolve(A, D, 4, model, dtt, 500*dtt, 1000, 1e-8,10)
        t1 = time.time()
        summary.append([D,cost,t1-t0])    
        # np.savez_compressed(filepath,
        #                     time=times,
        #                     state=state,
        #                     gradient_norm=norm,
        #                     cost=cost)
        print(summary)
        