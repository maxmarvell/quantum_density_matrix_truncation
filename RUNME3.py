import numpy as np

from qdmt.uniform_mps import UniformMps
from qdmt.model import TransverseFieldIsing
from qdmt.evolve import load_state, check_write_permission, evolve

'''
    TRY AND GET BEST RESULTS: INTEGRABLE
'''

def main():

    loadfile = 'data/ground_state/gstate_ising2_D6_g1.5.npy'
    A, prev_data, start_time = load_state(loadfile)

    filepath = 'data/integrable/bond_dimension_8_patch_size_24'
    assert check_write_permission(filepath)

    model = TransverseFieldIsing(g=0.2, delta_t=0.01)
    times, state, cost, norm = evolve(A, 8, 24, model, 0.01, 25, 1000, 1e-6, start_time)

    np.savez_compressed(filepath,
                        time=times,
                        state=state,
                        gradient_norm=norm,
                        cost=cost)

if __name__ == "__main__":
    main()
