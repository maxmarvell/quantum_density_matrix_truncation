from model import AbstractModel
from uniform_mps import UniformMps
from cost import EvolvedHilbertSchmidt
from optimisation import OpimisationProblem
from utils.geometry import Retraction, GrassmanProjector, GradientDescent, AbstractUpdate
import numpy as np

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

        t = .0
        self.model.delta_t = delta_t
        times = np.arange(0, t_max + delta_t/2, delta_t)
        cost = np.empty_like(times)
        norm = np.empty_like(times)
        data = np.empty_like(times)
        tensors = np.zeros((len(times), *self.A.tensor.shape), dtype=np.complex128)
        
        for i, _ in enumerate(times):

            f = EvolvedHilbertSchmidt(self.A, self.A, self.model, self.L)
            X = OpimisationProblem(f, g, alpha = 0.1)
            self.A, cost[i], norm[i] = X.optimize(1000, 1e-4, True)
            tensors[i] = self.A.tensor

            echo = self._compute_echo()
            print(f"\nCurrent LoSchmidt Echo value at t={t}: {echo}\n\n")
            data[i] = echo

        self.A = self.A0
        return times, data, cost, norm, tensors

if __name__ == "__main__":

    from model import TransverseFieldIsing

    with open('data/ground_state_d6.npy', 'rb') as f:
        _A = np.load(f)
    
    L = 8
    A = UniformMps(_A)
    quench_model = TransverseFieldIsing(.2)
    LSE = LoSchmidtEcho(A, L, quench_model)
    times, data, cost, norm, tensors = LSE.run(10)
    
    import datetime
    now = datetime.datetime.now()
    with open(f'data/LoSchmidt/{now}.npy', 'wb') as f:
        np.save(f, times)
        np.save(f, data)
        np.save(f, cost)
        np.save(f, norm)
        np.save(f, tensors)

    print(data)

