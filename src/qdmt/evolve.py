from qdmt.uniform_mps import UniformMps
from qdmt.model import AbstractModel, TransverseFieldIsing
from qdmt.cost import EvolvedHilbertSchmidt
from qdmt.optimisation import GradientDescent
from qdmt.manifold import Grassmann

import numpy as np
from numpy.lib.npyio import NpzFile
import argparse
import os
from pathlib import Path

def evolve(A0: UniformMps, 
           L: int, 
           model: AbstractModel, 
           delta_t: float, 
           max_t: float, 
           max_iter: int, 
           tol: float,
           start_t: float = 0.0,
           trotterization_order: int = 2):

    
    A = UniformMps(A0.tensor)

    times = np.arange(start_t + delta_t, max_t + delta_t/2, delta_t)
    cost = np.empty_like(times)
    norm = np.empty_like(times)
    state = np.empty((len(times), *A.tensor.shape), dtype=np.complex128)

    M = Grassmann()

    for i, t in enumerate(times):
        f = EvolvedHilbertSchmidt(A, A, model, L, trotterization_order)
        gd = GradientDescent(f, M)

        A, cost[i], norm[i] = gd.optimize(max_iter, tol, True)
        state[i] = A.tensor

        print(f"\nEvolved the state to t={t}\n\n")

    return times, state, cost, norm

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
        print(f"Successfully checked write permission for: {filepath}")
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
                "cost": [np.nan],
                "tol" : [np.nan],
                "max_iters": [np.nan]
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
    max_iters = np.full_like(times, args.max_iters)
    tol = np.full_like(times, args.tol)

    if previous_data:
        times = np.concatenate((previous_data['time'], times))
        state = np.concatenate((previous_data['state'], state))
        cost = np.concatenate((previous_data['cost'], cost))
        norm = np.concatenate((previous_data['gradient_norm'], norm))
        max_iters = np.concatenate((previous_data['max_iters'], max_iters))
        tol = np.concatenate((previous_data['tol'], tol))

    np.savez_compressed(savefile,
                        time=times,
                        state=state,
                        gradient_norm=norm,
                        cost=cost,
                        max_iters=max_iters,
                        tol=tol)

if __name__ == "__main__":
    
    filename = 'data/ground_state/gstate_ising2_D6_g1.5.npy'
    A, _, _ = load_state(filename)
    model = TransverseFieldIsing(0.2, 0.25)
    evolve(A, 4, model, 0.25, 1, 1000, 1e-5)

