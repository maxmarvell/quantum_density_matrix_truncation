from qdmt.model import AbstractModel
from qdmt.uniform_mps import UniformMps
from qdmt.cost import EvolvedHilbertSchmidt
from qdmt.optimisation import OpimisationProblem
from qdmt.utils.geometry import GrassmanProjector, GradientDescent, AbstractUpdate
from qdmt.model import TransverseFieldIsing
import numpy as np
import argparse

class LoSchmidtEcho():

    def __init__(self, A: UniformMps, L: int, quench_model: AbstractModel):
        self.A = A
        self.A0 = UniformMps(A.tensor)

        self.model = quench_model
        self.L = L

    def _compute_echo(self) -> float:
        return -np.log(self.A0.fidelity(self.A)**2)

    def run(self, t_max: float, delta_t: float, update: AbstractUpdate | None = None):
        """
            Evolve the uMPS 'A' for a time delta_t, updates A
        """

        if update is None:    
            proj = GrassmanProjector()
            g = GradientDescent(proj, True)
        else:
            g = update

        self.model.delta_t = delta_t
        times = np.arange(0, t_max + delta_t/2, delta_t)
        cost = np.empty_like(times)
        norm = np.empty_like(times)
        data = np.empty_like(times)
        tensors = np.zeros((len(times), *self.A.tensor.shape), dtype=np.complex128)
        
        for i, t in enumerate(times):

            f = EvolvedHilbertSchmidt(self.A, self.A, self.model, self.L)
            X = OpimisationProblem(f, g, alpha = 0.1)
            self.A, cost[i], norm[i] = X.optimize(1000, 1e-4, True)
            tensors[i] = self.A.tensor

            echo = self._compute_echo()
            print(f"\nCurrent LoSchmidt Echo value at t={t}: {echo}\n\n")
            data[i] = echo

        self.A = self.A0
        return times, data, cost, norm, tensors

def run_simulation(args):

    L = args.L
    g1 = args.g1
    g2 = args.g2
    D = args.D
    alpha = args.alpha
    max_iters = args.max_iters
    max_time = args.max_time
    tol = args.tol
    delta_t = args.delta_t

    # load the ground state
    filename = f"data/ground_state/gstate_ising2_D{D}_g{g1}.npy"
    with open(filename, 'rb') as f:
        _A = np.load(f)
        A = UniformMps(_A)
        A0 = UniformMps(_A)

    times = np.arange(0, max_time + delta_t/2, delta_t) + delta_t
    cost = np.empty_like(times)
    norm = np.empty_like(times)
    data = np.empty_like(times)
    tensors = np.zeros((len(times), *A.tensor.shape), dtype=np.complex128)

    quench_model = TransverseFieldIsing(g2, delta_t)
    proj = GrassmanProjector()
    g = GradientDescent(proj, True)

    for i, t in enumerate(times):

        f = EvolvedHilbertSchmidt(A, A, quench_model, L)
        X = OpimisationProblem(f, g, alpha)
        A, cost[i], norm[i] = X.optimize(max_iters, tol, True)
        tensors[i] = A.tensor

        echo = -np.log(A0.fidelity(A)**2)
        print(f"\nCurrent LoSchmidt Echo value at t={t}: {echo}\n\n")
        data[i] = echo

    savefile = f"data/LoSchmidt/g1_{g1}-g2_{g2}-D_{D}-L_{L}.npy"
    with open(savefile, 'wb') as f:
        np.save(f, times)
        np.save(f, data)
        np.save(f, cost)
        np.save(f, norm)
        np.save(f, tensors)

def main():
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
    run_simulation(args)

if __name__ == "__main__":
    main()