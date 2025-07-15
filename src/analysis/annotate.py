from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrix import TransferMatrix
from qdmt.cost import HilbertSchmidt
from qdmt.model import Pauli
from qdmt.fixed_point import RightFixedPoint

import numpy as np
from ncon import ncon
from scipy.sparse.linalg import eigs
import argparse

def compute_lo_schmidt(A: UniformMps, A0: UniformMps) -> np.float64:
    E = TransferMatrix.new(A, A0)
    fidl = E.fidelity()
    return -np.log(fidl*fidl.conj())

def compute_subsystem_lo_schmidt(A: UniformMps, A0: UniformMps, L: int) -> np.float64:
    f = HilbertSchmidt(A, A0, L)
    return f.cost()

def compute_transverse_magnetization(A: UniformMps) -> np.float64:
    E = TransferMatrix.new(A, A)
    r = RightFixedPoint(E)
    return np.abs(ncon([A.tensor, Pauli.Sx,  A.conj, r.tensor], [[1, 2, 3], [2, 4], [1, 4, 5], [3, 5]]))

def compute_correlation_length(A) -> np.float64:
    E = TransferMatrix.new(A, A)
    M = E.to_matrix()
    r = eigs(M, k=2, which='LM', return_eigenvectors=False)
    return -1.0/np.log(np.abs(r[1]/r[0]))


def parse():
    parser = argparse.ArgumentParser(description="Annotate time evolved data.")

    parser.add_argument(
        '--filepath',
        type=str,
        help='Relative file path to time evolved data.'
    )
    return parser.parse_args()

def main():
    args = parse()

    try:
        data = np.load(args.filepath)
    except OSError:
        print(f"Error: unable to read file at location {args.filepath}.")

    lo_schmidt_echo = np.empty_like(data['time'])
    subsystem_lo_schmidt_echo = np.empty_like(data['time'])
    transverse_magnetization = np.empty_like(data['time'])
    correlation_length = np.empty_like(data['time'])

    A0 = UniformMps(data['state'][0])

    for i in range(len(data['time'])):
        A = UniformMps(data['state'][i])
        lo_schmidt_echo[i] = compute_lo_schmidt(A, A0)
        subsystem_lo_schmidt_echo[i] = compute_subsystem_lo_schmidt(A, A0, L=8)
        transverse_magnetization[i] = compute_transverse_magnetization(A)
        correlation_length[i] = compute_correlation_length(A)

    np.savez_compressed(args.filepath,
                        **data,
                        lo_schmidt_echo=lo_schmidt_echo,
                        subsystem_lo_schmidt_echo=subsystem_lo_schmidt_echo,
                        transverse_magnetization=transverse_magnetization,
                        correlation_length=correlation_length)