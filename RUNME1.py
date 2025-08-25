import numpy as np

from qdmt.uniform_mps import UniformMps
from qdmt.model import TransverseFieldIsing
from qdmt.evolve import check_write_permission, evolve

'''
    TRY AND GET THE BEST RESULTS: NON-INTEGRABLE
'''

def main():
    theta = phi = np.pi / 2

    psi = np.array([np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)])

    A = UniformMps(psi.reshape(1, 2, 1))

    filepath = 'data/non_integrable/bond_dimension_12_patch_6'
    assert check_write_permission(filepath)

    model = TransverseFieldIsing(g=1.05, delta_t=0.01, h=-0.5, J=-1)
    times, state, cost, norm = evolve(A, 12, 6, model, 0.01, 10, 1000, 1e-8)

    np.savez_compressed(filepath,
                        time=times,
                        state=state,
                        gradient_norm=norm,
                        cost=cost)

if __name__ == "__main__":
    main()
