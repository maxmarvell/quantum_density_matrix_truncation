import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.integrate import quad
from .utils.graph import *

from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrix import TransferMatrix

def compute_lo_schmidt(A: UniformMps, A0: UniformMps) -> np.float64:
    E = TransferMatrix.new(A, A0)
    fidl = E.fidelity()
    return -np.log(fidl*fidl.conj())

def f(z, g0, g1):
    def theta(k, g):
        return np.arctan2(np.sin(k), g-np.cos(k))/2
    def phi(k, g0, g1):
        return theta(k, g0)-theta(k, g1)
    def epsilon(k, g1):
        return -2*np.sqrt((g1-np.cos(k))**2+np.sin(k)**2)
    def integrand(k):
        return -1/(2*np.pi)*np.log(np.cos(phi(k, g0, g1))**2 + np.sin(phi(k, g0, g1))**2 * np.exp(-2*z*epsilon(k, g1)))

    return quad(integrand, 0, np.pi)[0]

def loschmidt_paper(t, g0, g1):
    return (f(t*1j, g0, g1)+f(-1j*t, g0, g1))

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

    args = parse()

    num_files = len(args.filepath)


    if num_files == 1:
        try:
            data = np.load(args.filepath[0])
        except OSError:
            print(f"Error: unable to read file at location {args.filepath[0]}.")

        A0 = UniformMps(data['state'][0])
        lo_schmidt_echo = np.empty_like(data['time'])

        exact_times = np.linspace(0.0, data['time'][-1], int(20*data['time'][-1]))
        exact = [loschmidt_paper(t, 1.5, 0.2) for t in exact_times]

        for i in range(len(data['time'])):
            A = UniformMps(data['state'][i])
            lo_schmidt_echo[i] = compute_lo_schmidt(A, A0)
            
        _, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(exact_times, exact, label='Exact')
        axs[0].plot(data['time'], lo_schmidt_echo, '+', label='Approximate')
        axs[2].set_xlabel('Time (s)')
        axs[0].set_ylabel(r'$-\log\Bigl\{|\langle\psi|\psi\rangle|^2\Bigr\}$')

        axs[1].plot(data['time'], data['cost'])
        axs[1].set_ylabel(r'$C(A)$')
        axs[1].set_yscale('log')

        axs[2].plot(data['time'], data['gradient_norm'], label='Gradient Norm')
        axs[2].set_ylabel(r'$|G|$')
        axs[2].set_yscale('log')

    else: 
        _, axs = plt.subplots(num_files, 1, figsize=(10, 4 * num_files), sharex=True)   
        for i, filepath in enumerate(args.filepath):
            try:
                data = np.load(filepath)
            except OSError:
                print(f"Error: unable to read file at location {filepath}. Skipping this file.")
                continue

            A0 = UniformMps(data['state'][0])
            loschmidt = np.empty_like(data['time'])

            for j in range(len(data['time'])):
                A = UniformMps(data['state'][j])
                loschmidt[j] = compute_lo_schmidt(A, A0)

            exact_times = np.linspace(0.0, data['time'][-1], int(20*data['time'][-1]))
            exact = [loschmidt_paper(t, 1.5, 0.2) for t in exact_times]

            axs[i].plot(exact_times, exact, label='Exact')
            axs[i].plot(data['time'], loschmidt, '+', label=f'File: {filepath}')
            axs[i].set_ylabel(r'$-\log\bigl\{|\langle\psi|\psi\rangle|^2\bigr\}$')
            axs[i].legend()
            axs[i].grid(True) # Add grid for better readability

    # Set common x-label for the bottom-most subplot
    axs[-1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()