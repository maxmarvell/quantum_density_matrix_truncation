import numpy as np
import matplotlib.pyplot as plt

from qdmt.uniform_mps import UniformMps
from qdmt.cost import HilbertSchmidt
from qdmt.fixed_point import RightFixedPoint
from analysis.lo_schmidt_echo import compute_lo_schmidt
from analysis.utils.graph import *

def compute_trace_distace(A1, A2, L):
    f = HilbertSchmidt(A1, L)
    rA = RightFixedPoint.from_mps(A2)
    return f.cost(A2, rA)

def main():

    files = ['data/integrable/experiment_trotter_error/D_4-L_8-delta_t_0.02.npz']
    L = 160

    num_files = len(files)

    if num_files == 1:
        try:
            data = np.load(files[0])
        except OSError:
            print(f"Error: unable to read file at location {files[0]}.")

        A0 = UniformMps(data['state'][0])
        A1 = A0
        distance = np.empty_like(data['time'][:-1])
        lo_schmidt = np.empty_like(data['time'][:-1])

        for i in range(len(data['time']) - 1):
            A = UniformMps(data['state'][i])
            distance[i] = compute_trace_distace(A, A1, L)
            lo_schmidt[i] = compute_lo_schmidt(A, A0)
            A1 = A
            
        _, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(data['time'][:-1], distance)
        axs[1].set_xlabel('Time (s)')
        axs[0].set_ylabel(r'$Tr\bigl(\rho(t)-\rho(t+dt)\bigr)^2$')

        axs[1].plot(data['time'][:-1], lo_schmidt)
        axs[1].set_ylabel(r'$-\log\Bigl\{|\langle\psi|\psi\rangle|^2\Bigr\}$')

    else: 
        _, axs = plt.subplots(num_files, 1, figsize=(10, 6), sharex=True)   
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