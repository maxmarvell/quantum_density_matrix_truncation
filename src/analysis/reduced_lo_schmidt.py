import numpy as np
import matplotlib.pyplot as plt

from qdmt.uniform_mps import UniformMps
from qdmt.fixed_point import RightFixedPoint
from qdmt.cost import HilbertSchmidt
from analysis.lo_schmidt_echo import compute_lo_schmidt, loschmidt_paper

def compute_reduced_lo_schmidt(A: UniformMps, B: UniformMps, L: int) -> np.float64:
    f = HilbertSchmidt(A, B, L)
    rB = RightFixedPoint.from_mps(B)
    return -np.log(f._compute_rho_A_rho_B(B, rB))/L

def main():

    files = ['data/07-30/D_8-L_08.npz']
    L = 2

    num_files = len(files)

    if num_files == 1:
        try:
            data = np.load(files[0])
        except OSError:
            print(f"Error: unable to read file at location {files[0]}.")

        A0 = UniformMps(data['state'][0])
        reduced_lo_schmidt_echo = np.empty_like(data['time'])
        lo_schmidt_echo = np.empty_like(data['time'])

        exact_times = np.linspace(0.0, data['time'][-1], int(20*data['time'][-1]))
        exact = [loschmidt_paper(t, 1.5, 0.2) for t in exact_times]

        for i in range(len(data['time'])):
            A = UniformMps(data['state'][i])
            reduced_lo_schmidt_echo[i] = compute_reduced_lo_schmidt(A, A0, L)
            
        _, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(exact_times, exact, label='Exact')
        axs[0].plot(data['time'], reduced_lo_schmidt_echo, '+', label='Reduced Lo Schmidt')
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
        for i, filepath in enumerate(files):
            try:
                data = np.load(filepath)
            except OSError:
                print(f"Error: unable to read file at location {filepath}. Skipping this file.")
                continue

            A0 = UniformMps(data['state'][0])
            loschmidt = np.empty_like(data['time'])

            for j in range(len(data['time'])):
                A = UniformMps(data['state'][j])
                loschmidt[j] = compute_reduced_lo_schmidt(A, A0)

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