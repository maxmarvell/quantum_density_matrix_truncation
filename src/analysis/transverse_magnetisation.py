from ncon import ncon
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.integrate import quad

from qdmt.transfer_matrix import TransferMatrix
from qdmt.fixed_point import RightFixedPoint
from qdmt.uniform_mps import UniformMps
from qdmt.model import Pauli

from .utils.graph import *

def compute_transverse_magnetization(A: UniformMps) -> np.float64:
    E = TransferMatrix.new(A, A)
    r = RightFixedPoint(E)
    return ncon([A.tensor, Pauli.Sx,  A.conj, r.tensor], [[1, 2, 3], [2, 4], [1, 4, 5], [3, 5]])

def transverse_magnetization(t, g0, g1):
    def theta(k, g):
        return 0.5 * np.arctan2(np.sin(k), g - np.cos(k))

    def delta(k):
        return theta(k, g1) - theta(k, g0)

    def epsilon(k):
        return 2 * np.sqrt((g1 - np.cos(k))**2 + np.sin(k)**2)

    def integrand(k):
        th1 = theta(k, g1)
        d = delta(k)
        ek = epsilon(k)
        return (
            np.cos(2 * th1) * np.cos(2 * d) +
            np.sin(2 * th1) * np.sin(2 * d) * np.cos(2 * ek * t)
        ) / np.pi

    return quad(integrand, 0, np.pi)[0]

def parse():
    parser = argparse.ArgumentParser(description="Annotate time evolved data.")

    parser.add_argument(
        'filepath',
        type=str,
        nargs='+',
        help='Relative file path to time evolved data.'
    )
    return parser.parse_args()

def main():
    args = parse()

    num_files = len(args.filepath)
    _, axs = plt.subplots(num_files, 1, figsize=(10, 4 * num_files), sharex=True)

    if num_files == 1:
        axs = [axs]

    for i, filepath in enumerate(args.filepath):
        try:
            data = np.load(filepath)
        except OSError:
            print(f"Error: unable to read file at location {filepath}. Skipping this file.")
            continue

        transverse_mag = np.empty_like(data['time'])
        exact_times = np.linspace(0.0, data['time'][-1], int(20*data['time'][-1]))
        exact = [transverse_magnetization(t, 1.5, 0.2) for t in exact_times]

        for j in range(len(data['time'])):
            A = UniformMps(data['state'][j])
            transverse_mag[j] = compute_transverse_magnetization(A)

        # axs[i].plot(exact_times, exact, label='Exact')
        axs[i].plot(data['time'], transverse_mag, label=f'Approximate - {filepath}')
        axs[i].set_ylim(-.5, .5)
        axs[i].set_ylabel(r'$\langle\sigma^z\rangle$')
        axs[i].legend()
        axs[i].grid(True) # Add grid for better readability

    # Set common x-label for the bottom-most subplot
    axs[-1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()