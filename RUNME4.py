import numpy as np

from qdmt.uniform_mps import UniformMps
from qdmt.model import TransverseFieldIsing
from qdmt.evolve import load_state, check_write_permission, evolve

'''
    EXPERIMENT: RUN WITH SAME PARAMETERS AS LESLIE
'''

def main():

    loadfile = 'data/ground_state/gstate_ising2_D2_g1.5.npy'
    A, prev_data, start_time = load_state(loadfile)

    filepath = 'data/non_integrable/bond_dimension_2_patch_4'
    assert check_write_permission(filepath)

    model = TransverseFieldIsing(g=0.2, delta_t=0.1)
    times, state, cost, norm = evolve(A, 2, 4, model, 0.05, 20, 100, 1e-8, start_time)

    if prev_data:
        times = np.concatenate((prev_data['time'], times))
        state = np.concatenate((prev_data['state'], state))
        cost = np.concatenate((prev_data['cost'], cost))
        norm = np.concatenate((prev_data['gradient_norm'], norm))

    np.savez_compressed(filepath,
                        time=times,
                        state=state,
                        gradient_norm=norm,
                        cost=cost)

if __name__ == "__main__":
    main()
