import numpy as np

from qdmt.uniform_mps import UniformMps
from qdmt.model import TransverseFieldIsing
from qdmt.evolve import check_write_permission, evolve

def main():
    theta = phi = np.pi / 2

    psi = np.array([np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)])

    A = UniformMps(psi.reshape(1, 2, 1))

    filepath = 'data/non_integrable/bond_dimension_10_patch_4'
    assert check_write_permission(filepath)

    model = TransverseFieldIsing(g=1.05, delta_t=0.1, h=-0.5, J=-1)
    times, state, cost, norm = evolve(A, 10, 4, model, 0.1, 10, 1000, 1e-8)

    np.savez_compressed(filepath,
                        time=times,
                        state=state,
                        gradient_norm=norm,
                        cost=cost)

if __name__ == "__main__":
    main()
