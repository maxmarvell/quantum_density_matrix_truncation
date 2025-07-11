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
    
    def optimize(self, max_iters: int, tol: float, verbose: bool = False) -> tuple[UniformMps, np.complex128]:
        print(f"Initial cost: {self.f.cost()}")

        norm = tol
        for i in range(max_iters):
            self.f.B, norm = self.update()

            if verbose and i % 100 == 0:
                print(f"Iteration {i}:")
                print(f"\tCost - {np.abs(self.f.cost())}")
                print(f"\tGradient Norm - {norm}")

            if norm < tol:
                print("\nConverged!")
                print(f"Iteration {i}:")
                print(f"\tCost - {np.abs(self.f.cost())}")
                print(f"\tGradient Norm - {norm}")
                break


        return self.f.B, np.abs(self.f.cost()), np.abs(norm)
                    
if __name__ == "__main__":

    d = 4
    p = 2

    A = UniformMps.new(d, p)
    
    from qdmt.model import TransverseFieldIsing
    from qdmt.cost import EvolvedHilbertSchmidt
    tfim = TransverseFieldIsing(0.1, 0.1)

    proj = GrassmanProjector()
    g = GradientDescent(proj, True)
    f = EvolvedHilbertSchmidt(A, A, tfim, 4, trotterization_order=2)
    Op = OpimisationProblem(f, g, .1)
    Op.optimize(1000, 1e-10, verbose=True)
    print(np.allclose(ncon((Op.f.B.conj, Op.f.B.tensor), ((1, 2, -1), (1, 2, -2))), np.eye(d, dtype=np.complex128), rtol=1e-12))

