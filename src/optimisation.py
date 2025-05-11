from utils.geometry import *
from cost import AbstractCostFunction
from scipy.linalg import polar

class OpimisationProblem():
    
    def __init__(self, f: AbstractCostFunction, alpha: float = 0.1):
        self.f = f
        self.alpha = alpha

    def update(self, retraction):
        D = self.f.derivative()
        B = retraction(self.f.B, D, self.alpha, self.f.rB)
        n, p, _ = B.shape
        _B, _ = polar(B.reshape(n*p, n))
        return UniformMPS(_B.reshape(n, p, n))
    
    def optimize(self, retraction, n_iter: int, tol: float):
        print(f"Initial cost: {self.f.cost()}")

        for i in range(n_iter):
            self.f.B = self.update(retraction)
            self.f.rB = None 

            if i % 100 == 0:
                C = self.f.cost()
                print(f"Iteration {i}: {C}")
                if C < tol:
                    break
                    
if __name__ == "__main__":

    Da = 6
    Db = 6
    p = 2

    A = UniformMPS.from_random(Da, p)
    B = UniformMPS.from_random(Db, p)

    # generate unitary
    from utils.unitary import transverse_ising
    U = transverse_ising(0.1)

    from cost import EvolvedHilbertSchmidt, HilbertSchmidt

    f = EvolvedHilbertSchmidt(A, B, U, U, 2)
    Op = OpimisationProblem(f)
    Op.optimize(grassman_retraction, 1000, 1e-5)

    f = HilbertSchmidt(A, B, 2)
    Op = OpimisationProblem(f)
    Op.optimize(grassman_retraction, 1000, 1e-10)
