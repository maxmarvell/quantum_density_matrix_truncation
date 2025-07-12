import argparse
import numpy as np

from qdmt.lo_schmidt import LoSchmidtEcho
from qdmt.uniform_mps import UniformMps
from qdmt.model import TransverseFieldIsing

def run_simulation():

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
        help='Optimization tolerance at each time step (default: 1e-6)'
    )
    parser.add_argument(
        '--delta-t',
        type=float,
        default=0.1,
        help='Trotterized time step (default: 0.1)'
    )
    args = parser.parse_args()

    # load the ground state
    filename = f"data/ground_state/gstate_ising2_D{args.D}_g{args.g1}.npy"
    with open(filename, 'rb') as f:
        _A = np.load(f)
        A = UniformMps(_A)

    quench_model = TransverseFieldIsing(args.g2)
    lse = LoSchmidtEcho(A, args.L, quench_model)
    times, data, cost, norm, tensors = lse.run(args.max_time, args.delta_t, args.tol, args.max_iters)

    savefile = f"data/LoSchmidt/g1_{args.g1}-g2_{args.g2}-D_{args.D}-L_{args.L}.npy"
    with open(savefile, 'wb') as f:
        np.save(f, times)
        np.save(f, data)
        np.save(f, tensors)
        np.save(f, cost)
        np.save(f, norm)


def continue_simulation(args: argparse.Namespace):

    L = args.L
    g1 = args.g1
    g2 = args.g2
    D = args.D
    start_time = args.start_time

    alpha = args.alpha
    max_iters = args.max_iters
    max_time = args.max_time
    tol = args.tol
    delta_t = args.delta_t

    try:
        with open(f"data/ground_state/gstate_ising2_D{D}_g{g1}.npy") as f:
            _A = np.load(f)
            A0 = np.load(_A)
    except IOError as e:
        print(f"Error: Could not read file: {e}")

    try:
        with open(f"data/LoSchmidt/g1_{g1}-g2_{g2}-D_{D}-L_{L}.npy") as f:
            times = np.load(f)
            np.load(f)
            np.load(f)
            np.load(f)
            tensors = np.load(f)
    except IOError as e:
        print(f"Error: Could not read file: {e}")

def plot_results():

    import qdmt.utils.graph
    import matplotlib.pyplot as plt

    # parser the command line arguments
    parser = argparse.ArgumentParser(description="Plot Lo-Schmidt echo simulation.")
    parser.add_argument(
        '--filename',
        type=str,
        help='File containing data'
    )
    parser.add_argument(
        '--type',
        type=int,
        default=1,
        help='Plot type'
    )
    args = parser.parse_args()

    filename = args.filename
    type = args.type

    try:
        with open(f"data/LoSchmidt/{filename}", 'rb') as f:
            times = np.load(f)
            data = np.load(f)
            _ = np.load(f)
            _ = np.load(f)
            _ = np.load(f)

            if type == 1:
                _, ax = plt.subplots(figsize=(10, 6))
                ax.plot(times, data, marker='o', markersize=2)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(r'$|\langle\psi|\psi\rangle|^2$')

            elif type == 2:
                pass

            plt.show()

    except IOError as e:
        print(f"Error: Could not read file: {e}")
    pass


