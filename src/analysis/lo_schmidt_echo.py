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
        help='Relative file path to time evolved data.'
    )
    return parser.parse_args()

def main():
    args = parse()

    try:
        data = np.load(args.filepath)
    except OSError:
        print(f"Error: unable to read file at location {args.filepath}.")

    A0 = UniformMps(data['state'][0])
    lo_schmidt_echo = np.empty_like(data['time'])

    for i in range(len(data['time'])):
        A = UniformMps(data['state'][i])
        lo_schmidt_echo[i] = compute_lo_schmidt(A, A0)

    print(data['gradient_norm'])
    print(data['cost'])
    print(data['max_iters'])
    print(data['tol'])

    exact_times = np.linspace(0.0, 10.0, 400)
    exact = [loschmidt_paper(t, 1.5, 0.2) for t in exact_times]

    _, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(exact_times, exact, label='Exact')
    ax[0].plot(data['time'], lo_schmidt_echo, '+', label='Approximate')
    ax[2].set_xlabel('Time (s)')
    ax[0].set_ylabel(r'$|\langle\psi|\psi\rangle|^2$')

    ax[1].plot(data['time'], data['cost'])
    ax[1].set_yscale('log')

    ax[2].plot(data['time'], data['gradient_norm'], label='Gradient Norm')
    ax[2].plot(data['time'], data['tol'], '--', label='Tolerance')
    ax[2].set_yscale('log')

    plt.show()
    

if __name__ == "__main__":
    main()