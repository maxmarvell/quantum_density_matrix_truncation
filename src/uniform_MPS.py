from ncon import ncon
import numpy as np
from scipy.linalg import polar
from scipy.sparse.linalg import eigs
from scipy.stats import unitary_group
from typing import Self
# from transfer_matrix import TransferMatrix

class UniformMPS():

    def __init__(self, A: np.ndarray):
        self.d, self.p, _ = A.shape
        self.tensor = A
        # set transfer matrix in __init__
        # self.E = 

    def __add__(self, other: Self):
        return UniformMPS(self.tensor + other.tensor, normalise=False)
    
    def __sub__(self, other: Self):
        return UniformMPS(self.tensor - other.tensor, normalise=False)
    
    def __mul__(self, other):
        return UniformMPS(self.tensor * other)
    
    __rmul__ = __mul__

    @classmethod
    def from_random(cls, d: int, p: int):
        """
            Initialise a random normalised uniform MPS states in left canonical form.

            args:
                d: Bond dimension of the uniform matrix product state
                p: Physical dimension of the uniform matrix product state

            returns:
                Random normalised uniform matrix product states, shape (d, p, d)
        """
        A = np.random.normal(size=(d*p, d)) + 1j * np.random.normal(size=(d*p, d))
        A, _ = polar(A)
        return cls(A.reshape(d, p, d))
    
    @classmethod
    def from_file(cls, file_path: str):
        """
            Create a UniformMPS instance by reading tensor data from a file.

            args:
                file_path: Path to the file containing the tensor data.

            returns:
                A UniformMPS instance initialized with the tensor data from the file.
        """
        # Load the tensor data from the file
        tensor = np.load(file_path)

        # Ensure the tensor has the correct shape
        if len(tensor.shape) != 3:
            raise ValueError("The loaded tensor must have three dimensions (d, p, d).")

        return cls(tensor)
    
    @property
    def matrix(self):
        return self.tensor.reshape(self.d * self.p, self.d)
    
    @property 
    def conj(self):
        return self.tensor.conj()
    
    # @property
    # def is_isometry(self):
    #     A = self.matrix
    #     is_isometry = np.allclose(A @ np.conj(A).T, np.eye(self.d))
    #     return is_isometry
    
    def right_fixed_point(self):
        """
            Find the dominant right eigenvector 
        """

        if self.E is None:
            self.E = self.transfer_matrix

        _, r = eigs(self.transfer_matrix.tensor.reshape((self.d**2, self.d**2)), k=1, which='LM')
        r = r.reshape((self.d, self.d))
        self.r = r / np.trace(r)
    
    def gauge(self, U: np.ndarray = None):

        if U is None:
            U = random_unitary(self.d)

        # check unitarity
        assert np.allclose(ncon((np.conj(U).T, U), ((-1, 1), (1, -2))), np.eye(self.d)), 'U should satisfy unitary constraint U^dagger*U=1'

        A = ncon((U, self.tensor, np.conj(U).T), ((-1, 1), (1, -2, 3), (3, -3)))
        r = ncon((U, self.r, np.conj(U).T), ((-1, 1), (1, 2), (2, -2)))

        gauged = UniformMPS(A)
        gauged.r = r

        return gauged
    
    def mps_chain(self, L: int):
        tensors = [self.tensor for i in range(L)]
        indices = [[-i, -(i+1), i] if i == 1 else [i-1, -(i+1), -(i+2)] if i == L else [i-1, -(i+1), i] for i in range(1, L+1)]
        return ncon(tensors, indices)
    
    def reduced_density_mat(self, L:int):
        """
            Returns the reduced density matrix in the form:

            e.g    1   3   5   7
                   |   |   |   |
                 - A - A - A - A - 
                |                 R
                 - A*- A*- A*- A*-
                   |   |   |   |
                   2   4   6   8
        """

        mps = self.mps_chain(L)
        tensors = (mps, self.r, np.conj(mps))
        indices = ([1, *[i for i in range(-1, -(2*L), -2)], 2], [2, 3], [1, *[i for i in range(-2, -(2*L+1), -2)], 3])
        return ncon(tensors, indices)

    def fidelity(self, other:Self):
        E = TransferMatrix(ncon((self.tensor, other.tensor), ([-1, 1, -3], [-4, 1, -2])))
        return E.leading_r_eigen()
    

# class TransferMatrix():
    
#     def __init__(self, tensor: np.ndarray):
#         self.tensor = tensor
#         self.dims = (tensor.shape[0], tensor.shape[1])

#     def __matmul__(self, other):
#         res = ncon((self.tensor, other.tensor), ((-1, -2, 1, 2), (1, 2, -3, -4)))
#         return TransferMatrix(res)
    
#     def __pow__(self, n: int):

#         d1, d2 = self.dims
#         if n == 0:
#             return TransferMatrix(np.identity(d1*d2, self.tensor.dtype).reshape(d1, d2, d1, d2))

#         order = int(np.log2(n))
#         E = TransferMatrix(self.tensor)
#         for _ in range(order):
#             E = E @ E

#         rmd = n - 2**order
#         if rmd > 0:
#             E = E @ self ** rmd

#         return E
    
#     @staticmethod
#     def construct_transfer_matrix(*args: np.ndarray):

#         if len(args) == 1:
#             A, B = args[0], args[0]
#         elif len(args) == 2:
#             A, B = args[0], args[1]
#         else:
#             raise TypeError("TransferMatrix.__init__() must provide 1 or 2 positional arguments")

#         E = ncon((A, np.conj(B)), ([-1, 1, -3], [-2, 1, -4]))

#         return TransferMatrix(E)
    
#     def leading_r_eigen(self):
#         d1, d2 = self.dims
#         r, _ = eigs(self.tensor.reshape((d1*d2, d1*d2)), k=1, which='LM')
#         return r[0]
        

def random_unitary(D: int):
    U = unitary_group.rvs(D)
    assert np.allclose(ncon((np.conj(U).T, U), ((-1, 1), (1, -2))), np.eye(D))
    return U
            
if __name__ == "__main__":

    Da = 5
    p = 2   

    # uMPS = UniformMPS.random(Da, p)
    # uMPS.r_dominant()
    # r = uMPS.r
    # E = uMPS.transfer_matrix

    # # check normalisation
    # assert np.abs(ncon((r, ), ((1, 1))) - 1) < 1e-12

    # # check left subspace
    # assert np.allclose(ncon((E.tensor,), ((1, 1, -1, -2))), np.eye(Da))

    # # check right subspace
    # assert np.allclose(r, np.conj(r).T)

    # assert np.allclose(r, ncon((E.tensor, r), ((-1, -2, 1, 2), (1, 2))))

    # # check power function is working
    # assert np.allclose((E ** 4).tensor, ncon((E.tensor, E.tensor, E.tensor, E.tensor), ((-1, -2, 3, 4), (3, 4, 5, 6), (5, 6, 7, 8), (7, 8, -3, -4))))

    # # check gauge transform
    # p = uMPS.reduced_density_mat(3)
    # Up = uMPS.gauge().reduced_density_mat(3)

    # assert np.allclose(p, Up)

    # assert np.allclose(
    #     ncon((p, Up), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))),
    #     ncon((p, p), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5))),
    #     ncon((Up, Up), ((1, 2, 3, 4, 5, 6), (2, 1, 4, 3, 6, 5)))
    # )

    # assert np.allclose(E.tensor, ncon(((E ** 0).tensor, E.tensor), ((-1, -2, 1, 2), (1, 2, -3, -4))))

    # print("All assertions passed!")


    A = UniformMPS.from_file("data/isometries/A.npy")
    B = UniformMPS.from_file("data/isometries/B.npy")