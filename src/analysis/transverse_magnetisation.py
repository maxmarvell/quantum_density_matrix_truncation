from ncon import ncon
import numpy as np
import matplotlib.pyplot as plt
import argparse

from qdmt.transfer_matrix import TransferMatrix
from qdmt.fixed_point import RightFixedPoint
from qdmt.uniform_mps import UniformMps
from qdmt.model import Pauli

def compute_transverse_magnetization(A: UniformMps) -> np.float64:
    E = TransferMatrix.new(A, A)
    r = RightFixedPoint(E)
    return np.abs(ncon([A.tensor, Pauli.Sx,  A.conj, r.tensor], [[1, 2, 3], [2, 4], [1, 4, 5], [3, 5]]))

def parse():
    parser = argparse.ArgumentParser(description="Annotate time evolved data.")

    parser.add_argument(
        'filepath',
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

    transverse_mag = np.empty_like(data['time'])

    for i in range(len(data['time'])):
        A = UniformMps(data['state'][i])
        transverse_mag[i] = compute_transverse_magnetization(A)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['time'], transverse_mag, marker='o', markersize=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\langle\sigma^z\rangle$')

    plt.show()
    
if __name__ == "__main__":
    main()