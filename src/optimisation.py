from utils.geometry import *
from cost import AbstractCostFunction
from uniform_MPS import UniformMPS

class OpimisationProblem():
    
    def __init__(self, f: AbstractCostFunction, g: AbstractUpdate, alpha: float = 0.1):
        self.f = f
        self.alpha = alpha
        self.g = g

    def update(self):
        D = self.f.derivative()
        B = self.g.update(self.f.B.tensor, D, self.alpha, self.f.rB.tensor)
        return UniformMPS(B)
    
    def optimize(self, n_iter: int, tol: float):
        print(f"Initial cost: {self.f.cost()}")

        for i in range(n_iter):
            self.f.B = self.update()
            self.f.rB = None

            # if i > 0 and i % 500 == 0:
            #     self.alpha /= 5.0 
            #     print(f"Reducing learning rate to {self.alpha}")

            if i % 100 == 0:
                C = self.f.cost()
                print(f"Iteration {i}: {C}")
                if C < tol:
                    break
                    
if __name__ == "__main__":

    Da = 5
    Db = 5
    p = 2

    A = UniformMPS.from_random(Da, p)
    B = UniformMPS.from_random(Db, p)

    # generate unitary
    from utils.unitary import transverse_ising
    U = transverse_ising(0.1)

    from cost import EvolvedHilbertSchmidt, HilbertSchmidt

    # f = EvolvedHilbertSchmidt(A, A, U, U, 10)
    # Op = OpimisationProblem(f)
    # Op.optimize(grassman_retraction, 5000, 1e-5)

    # f = EvolvedHilbertSchmidt(A, B, U, U, 4)
    # Op = OpimisationProblem(f)
    # Op.optimize(grassman_retraction, 1000, 1e-5)

    proj = GrassmanProjector()
    g = TestRetraction(proj, True)

    # f = EvolvedHilbertSchmidt(A, B, U, U, 2)
    # Op = OpimisationProblem(f, retr, 0.01)
    # Op.optimize(10000, 1e-5)

    f = HilbertSchmidt(A, B, L=2)
    Op = OpimisationProblem(f, g, 0.1)
    Op.optimize(1000, 1e-10)

    # print(np.allclose(ncon((Op.f.B.conj, Op.f.B.tensor), ((1, 2, -1), (1, 2, -2))), np.eye(Db, dtype=np.complex128), rtol=1e-12))

    g = GradientDescent(proj, True)
    f = HilbertSchmidt(A, B, L=2)
    Op = OpimisationProblem(f, g, 0.1)
    Op.optimize(1000, 1e-10)

    g = Retraction(proj, False)
    f = HilbertSchmidt(A, B, L=2)
    Op = OpimisationProblem(f, g, 0.1)
    Op.optimize(1000, 1e-10)

    print(np.allclose(ncon((Op.f.B.conj, Op.f.B.tensor), ((1, 2, -1), (1, 2, -2))), np.eye(Db, dtype=np.complex128), rtol=1e-12))
