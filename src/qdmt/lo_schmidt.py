from qdmt.model import AbstractModel
from qdmt.uniform_mps import UniformMps
from qdmt.cost import EvolvedHilbertSchmidt
from qdmt.optimisation import OpimisationProblem
from qdmt.utils.geometry import GrassmanProjector, GradientDescent, AbstractUpdate
import numpy as np

class LoSchmidtEcho():

    def __init__(self, A: UniformMps, L: int, quench_model: AbstractModel):
        self.A = A
        self.A0 = UniformMps(A.tensor)

        self.model = quench_model
        self.L = L

    def _compute_echo(self) -> float:
        return -np.log(self.A0.fidelity(self.A)**2)

    def run(self, t_max: float, delta_t: float, tol: float, max_iters: int, alpha: float = 0.1, update: AbstractUpdate | None = None):
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
            X = OpimisationProblem(f, g, alpha = alpha)
            self.A, cost[i], norm[i] = X.optimize(max_iters, tol, True)
            tensors[i] = self.A.tensor

            echo = self._compute_echo()
            print(f"\nCurrent LoSchmidt Echo value at t={t}: {echo}\n\n")
            data[i] = echo

        self.A = self.A0
        return times, data, cost, norm, tensors