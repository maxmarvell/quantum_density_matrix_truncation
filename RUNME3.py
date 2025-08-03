import numpy as np

from qdmt.uniform_mps import UniformMps
from qdmt.model import TransverseFieldIsing
from qdmt.evolve import load_state, check_write_permission, evolve

def main():

    loadfile = 'data/integrable/experiment_bond_dimension/D_8-L_08.npz'
    A, prev_data, start_time = load_state(loadfile)

    filepath = 'data/integrable/experiment_bond_dimension/D_8-L_08--2'
    assert check_write_permission(filepath)

    model = TransverseFieldIsing(g=0.2, delta_t=0.25)
    times, state, cost, norm = evolve(A, 8, 8, model, 0.25, 20, 1000, 1e-8, start_time)

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
