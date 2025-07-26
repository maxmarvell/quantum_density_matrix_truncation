from qdmt.utils.geometry import *
from qdmt.cost import AbstractCostFunction
from qdmt.uniform_mps import UniformMps

class OpimisationProblem():
    
    def __init__(self, f: AbstractCostFunction, g: AbstractUpdate, alpha: float = 0.1):
        self.f = f
        self.alpha = alpha
        self.g = g

    def update(self):
        D = self.f.derivative()
        B, norm = self.g.update(self.f.B.tensor, D, self.alpha, self.f.rB.tensor)
        return UniformMps(B), norm
    
    def optimize(self, max_iters: int, tol: float, verbose: bool = False, rtol: float = 1e-3) -> tuple[UniformMps, np.complex128]:
        print(f"Initial cost: {abs(self.f.cost())}")

        norm = tol
        cost = []
        gradient_norm = []

        for i in range(max_iters):

            tmp_cost = self.f.cost()
            cost.append(tmp_cost)

            self.f.B, norm = self.update()
            gradient_norm.append(norm)

            if verbose and i % 100 == 0:
                print(f"Iteration {i}:")
                print(f"\tCost - {abs(tmp_cost)}")
                print(f"\tGradient Norm - {norm}")

            if i > 1:
                if abs(cost[i-1] - cost[i-2]) / abs(cost[i-2]) < rtol:
                    print("\nConverged, cost function not sufficiently decreasing!")
                    print(f"Iteration {i}:")
                    print(f"\tCost - {abs(tmp_cost)}")
                    print(f"\tGradient Norm - {norm}")
                    break

            if norm < tol:
                print("\nConverged!")
                print(f"Iteration {i}:")
                print(f"\tCost - {abs(tmp_cost)}")
                print(f"\tGradient Norm - {norm}")
                break


        return self.f.B, np.abs(self.f.cost()), np.abs(norm)
                    
if __name__ == "__main__":

    d = 4
    p = 2

    A = UniformMps.new(d, p)
    
    from qdmt.model import TransverseFieldIsing
    from qdmt.cost import EvolvedHilbertSchmidt
    from utils.geometry import ConjugateGradient
    tfim = TransverseFieldIsing(0.1, 0.1)

    proj = GrassmanProjector()
    g = ConjugateGradient(proj, True)
    f = EvolvedHilbertSchmidt(A, A, tfim, 4, trotterization_order=2)
    Op = OpimisationProblem(f, g, .1)
    Op.optimize(1000, 1e-10, verbose=True)
    print(np.allclose(ncon((Op.f.B.conj, Op.f.B.tensor), ((1, 2, -1), (1, 2, -2))), np.eye(d, dtype=np.complex128), rtol=1e-12))

