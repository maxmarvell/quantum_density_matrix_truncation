from qdmt.uniform_mps import UniformMps
from qdmt.model import AbstractModel
from qdmt.utils.geometry import AbstractUpdate
from qdmt.cost import EvolvedHilbertSchmidt
from qdmt.optimisation import OpimisationProblem
from qdmt.utils.geometry import GrassmanProjector, GradientDescent, AbstractUpdate

import numpy as np
import argparse
import os

def evolve(A0: UniformMps, 
           L: int, 
           model: AbstractModel, 
           delta_t: float, 
           max_t: float, 
           max_iter: int, 
           tol: float,
           alpha: float = 0.1, 
           start_t: float = 0.0,
           update: AbstractUpdate | None = None, 
           trotterization_order: int = 2):
    
    if update is None:    
        proj = GrassmanProjector()
        g = GradientDescent(proj, True)
    else:
        g = update

    
    A = UniformMps(A0.tensor)

    times = np.arange(start_t, max_t + delta_t/2, delta_t)
    cost = np.empty_like(times)
    norm = np.empty_like(times)
    state = np.empty((len(times), *A.tensor.shape), dtype=np.complex128)

    for i, t in enumerate(times):

        if i == 0:
            norm[0] = 0.0
            cost[0] = 0.0
            state[0] = A.tensor
            continue

        f = EvolvedHilbertSchmidt(A, A, model, L, trotterization_order)
        X = OpimisationProblem(f, g, alpha=alpha)
        A, cost[i], norm[i] = X.optimize(max_iter, tol, True)
        state[i] = A.tensor

        print(f"\nEvolved the state to t={t}\n\n")

    return times, state, cost, norm

def run_simulation():

    from qdmt.model import TransverseFieldIsing

    parser = argparse.ArgumentParser(description="Run the Lo-Schmidt Echo simulation.")

    parser.add_argument(
        '-L',
        type=int,
        default=4,
        help='Length of the patch (default: 4)'
    )
    parser.add_argument(
        '-g1',
        type=float,
        default=1.5,
        help='Ground state Ising coupling strength (default: 1.5).'
    )
    parser.add_argument(
        '-g2',
        type=float,
        default=0.2,
        help='Quench coupling strength (default: 0.2).'
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
        '--max-time',
        type=float,
        default=3.0,
        help='Time to run the simulation until'
    )
    parser.add_argument(
        '--tol',
        type=float,
        default=1e-8,
        help='Optimization tolerance at each time step (default: 1e-8)'
    )
    parser.add_argument(
        '--delta-t',
        type=float,
        default=0.1,
        help='Trotterized time step (default: 0.1)'
    )
    args = parser.parse_args()

    start_time = 0.0
    model = TransverseFieldIsing(args.g2, args.delta_t)
    filepath = f"data/transverse_field/g1_{args.g1}-g2_{args.g2}-D_{args.D}-L_{args.L}"
    concat = False

    # determine whether it is safe to write to file
    if os.path.exists(filepath+".npz"):
        while True:
            choice = input(f"⚠️  Warning: '{filepath}' already exists.\n"
                           "Type 'w' if you would like to write to the same file or 'n' to choose a new file location: ").lower()
            
            if choice == 'w':
                print("The file will be overwritten!")
                savefile = filepath
                break
            
            elif choice == 'n':
                savefile = input("Enter the new file path: ")
                break
            
            else:
                print("Invalid choice. Please try again.")

        while True:
            choice = input(f"Would you like to continue the simulation of start over?\n"
                           "Type 'y' if you would like to continue the simulation otherwise choose 'n': ").lower()
            
            if choice == 'y':
                while True :

                    try:
                        choice = float(input("What time will you start from? "))
                    except:
                        print("Was unable to parse value as number please try again.\n\n")
                        continue

                    data = np.load(filepath+".npz")

                    idx = np.searchsorted(data['time'], choice, side='right')

                    if idx == 0:
                        print(f"Given start time is smaller than smallest value {data['time'][0]}s.\n"
                                "Please provide another value!")
                        continue

                    start_time = data['time'][idx - 1]

                    choice = input(f"The closest start time is {start_time}, is this correct (y/n)?").lower()

                    if choice == 'y':
                        A = UniformMps(data['state'][idx - 1])
                        prev_times = data['time'][:idx - 1]
                        prev_state = data['state'][:idx - 1]
                        prev_cost = data['cost'][:idx - 1]
                        prev_norm = data['gradient_norm'][:idx - 1]
                        prev_tol = data['tol'][:idx - 1]
                        prev_max_iters = data['max_iters'][:idx - 1]
                        concat = True
                        break

                    else:
                        continue
                break

            elif choice == 'n':
                print("Simulating from the initial state at t=0.0s.")

                # load the ground state
                filename = f"data/ground_state/gstate_ising2_D{args.D}_g{args.g1}.npy"
                with open(filename, 'rb') as f:
                    A = UniformMps(np.load(f))
                break

            else:
                print("Invalid choice. Please try again.")
    else:
         # load the ground state
        filename = f"data/ground_state/gstate_ising2_D{args.D}_g{args.g1}.npy"
        with open(filename, 'rb') as f:
            A = UniformMps(np.load(f))

        savefile = filepath

    times, state, cost, norm = evolve(A, args.L, model, args.delta_t, args.max_time, args.max_iters, args.tol, args.alpha, start_t=start_time)
    max_iters = np.full_like(times, args.max_iters)
    tol = np.full_like(times, args.tol)

    if concat:
        times = np.concatenate((prev_times, times))
        state = np.concatenate((prev_state, state))
        cost = np.concatenate((prev_cost, cost))
        norm = np.concatenate((prev_norm, norm))
        max_iters = np.concatenate((prev_max_iters, max_iters))
        tol = np.concatenate((prev_tol, tol))

    np.savez_compressed(savefile,
                        time=times,
                        state=state,
                        gradient_norm=norm,
                        cost=cost,
                        max_iters=max_iters,
                        tol=tol)

if __name__ == "__main__":
    # run_simulation()

    from model import TransverseFieldIsing
    from utils.geometry import GrassmanProjector, Retraction

    d = 6
    A0 = UniformMps(np.load(f'data/ground_state/gstate_ising2_D{d}_g1.5.npy'))
    delta_t = 0.1
    tfim = TransverseFieldIsing(0.2, delta_t)
    L = 8
    max_t = 1.0
    tol = 1e-10
    alpha = 0.01

    P = GrassmanProjector()
    g = Retraction(P, True)

    _, states, _, _ = evolve(A0, L, tfim, delta_t, max_t, 10000, tol, update=g)
    A1 = UniformMps(states[-1])

    from ncon import ncon

    assert np.allclose(ncon([A1.tensor, A1.conj], [[1, 2, -2], [1, 2, -1]]), np.eye(d))