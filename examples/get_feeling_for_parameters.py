from src.qdmt.evolve import *
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from examples.data_management import *

import os
import matplotlib.pyplot as plt




from pathlib import Path

import numpy as np
import re
from pathlib import Path

def merge_temp_files(tmp_dir: Path, output_file: Path):
    tmp_dir = Path(tmp_dir)

    # --- Get all matching chunk files ---
    files = list(tmp_dir.glob("chunk*.npz"))
    if not files:
        print("WARNING: No temporary files found in", tmp_dir)
        return

    # --- Numeric sorting: chunk0, chunk1, ..., chunk10 ---
    def extract_chunk_number(path):
        m = re.search(r"chunk(\d+)", path.stem)
        return int(m.group(1)) if m else -1

    files = sorted(files, key=extract_chunk_number)

    # --- Collect everything ---
    times_all = []
    states_all = []
    costs_all = []
    norms_all = []
    durations_all = []
    maxiters_all = []

    for f in files:
        print("Reading:", f)
        data = np.load(f, allow_pickle=True)

        times_all.append(np.asarray(data["time"]))
        states_all.append(np.asarray(data["state"]))
        costs_all.append(np.asarray(data["cost"]))
        norms_all.append(np.asarray(data["gradient_norm"]))
        durations_all.append(np.asarray(data["duration"]))
        maxiters_all.append(np.asarray(data["maxiter"]))

    # --- Concatenate ---
    times_all = np.concatenate(times_all)
    states_all = np.concatenate(states_all, axis=0)
    costs_all = np.concatenate(costs_all)
    norms_all = np.concatenate(norms_all)
    durations_all = np.concatenate(durations_all)
    maxiters_all = np.concatenate(maxiters_all)

    # --- Sort everything by time ---
    idx = np.argsort(times_all)

    times_sorted = times_all[idx]
    states_sorted = states_all[idx]
    costs_sorted = costs_all[idx]
    norms_sorted = norms_all[idx]
    durations_sorted = durations_all[idx]
    maxiters_sorted = maxiters_all[idx]

    # --- Save final merged file ---
    np.savez_compressed(
        output_file,
        time=times_sorted,
        state=states_sorted,
        cost=costs_sorted,
        gradient_norm=norms_sorted,
        duration=durations_sorted,
        maxiter=maxiters_sorted
    )

    print(f"✓ Successfully merged {len(files)} chunks → {output_file}")
    print(f"✓ time range: {times_sorted[0]} → {times_sorted[-1]}")

def Execute_Run(dt: float, steps: int, tolerance: float, iterations: int, D: int, cut_off: float, save = True, debug = False, A_init = None, save_after_steps = 50, g=1.05,h=-0.5, J=-1, L=4, theta =  np.pi / 2, phi =  np.pi / 2, run_folder: str | Path = None):
    if save == False:
        print("WARNING, NOT SAVING THE DATA!")

    ##### fixed things

    psi = np.array([np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)])
    if A_init is None:
        A_init = UniformMps(psi.reshape(1, 2, 1))


    ##### fixed things

    filepath = filepath_gen(dt, steps, tolerance, iterations, D, cut_off, run_folder)

    # make sure the parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)


    print(str(filepath))

    if debug:
        filepath = Path(str(filepath) + "_debug")


        

    assert check_write_permission(filepath)

    # model = TransverseFieldIsing(g=0, delta_t=dt, h=0, J=0)
    model = TransverseFieldIsing(g=g, delta_t=dt, h=h, J=J)

    print([g,J,h])
    time_total = steps*dt
    time_chunk = save_after_steps*dt

    # ------------------------------
    # Create UNIQUE temp directory for this run
    # ------------------------------
    tmp_dir = filepath.parent / (".tmp_" + filepath.name)
    tmp_dir.mkdir(exist_ok=True)
    print("Temporary directory:", tmp_dir)

    start_times = np.arange(0, time_total, time_chunk)

    chunk_id=0
    for this_start in start_times:
        max_t=this_start+time_chunk
        times, state, cost, norm, duration, maxiter = evolve(A_init, D, L, model, dt, max_t, iterations, tolerance, cut_off,start_t=this_start)
        # print(state)
        # print(this_start)
        A_init=UniformMps(state[-1])
       
       # Save chunk
        chunk_file = tmp_dir / f"chunk{chunk_id}.npz"
        print("Writing:", chunk_file)

        np.savez_compressed(
            chunk_file,
            time=times,
            state=state,
            cost=cost,
            gradient_norm=norm,
            duration=duration,
            maxiter=maxiter
        )

        chunk_id += 1

    # ------------------------------
    # Merge all chunks AFTER the loop
    # ------------------------------
    print("Merging temporary files...")
    merge_temp_files(tmp_dir, filepath)

    print("✓ Finished run.")
  
    # if save:
    #     np.savez_compressed(filepath,time=times, state=state, gradient_norm=norm, cost=cost, duration=duration, maxiter=maxiter)







# def Resume_Run(dt: float, steps: int, tolerance: float, iterations: int, D: int, cut_off_previous: float, cut_off_now: float):
#     data = load_data(dt, steps, tolerance, iterations, D, cut_off_previous)

#     valid_len = count_initial_nonzero(data["cost"])-2
#     keys = ["time", "state", "cost", "gradient_norm", "duration"]
#     times, state, cost, norm, duration = [data[k][:valid_len] for k in keys]

#     filepath = filepath_gen(dt, steps, tolerance, iterations, D, cut_off_now)+"test"
#     assert check_write_permission(filepath)

#     model = TransverseFieldIsing(g=1.05, delta_t=dt, h=-0.5, J=-1)
#     time_total = steps*dt

#     t_start = times[-1]

#     print(cut_off_now)

#     print(cost[-1])
    
#     A_init = UniformMps(state[-1])
#     print("go")
#     new_times, new_state, new_cost, new_norm, new_duration = evolve(A_init, D, L, model, dt, time_total, iterations, tolerance, cut_off_now, t_start)
    
    
#     combi_times = np.concatenate(times,new_times)
#     combi_state = np.concatenate(state,new_state)
#     combi_cost = np.concatenate(cost,new_cost)
#     combi_norm = np.concatenate(norm,new_norm)
#     combi_duration = np.concatenate(duration,new_duration)

#     # np.savez_compressed(filepath,time=combi_times, state=combi_state, gradient_norm=combi_norm, cost=combi_cost, duration=combi_duration)



if __name__ == "__main__":
    # print('test')


    from qdmt.model import TransverseFieldIsing
    from qdmt.cost import EvolvedHilbertSchmidt
    from qdmt.manifold import Grassmann


    dt = 1e-3
    steps = int(0.2//dt)
    steps = 3
    tol =1e-9
    iter=10000
    Ddim = 12

    hour = 3600
    days = 24*hour
    
    cut_off = hour

    # Execute_Run(dt, steps, tol, iter, Ddim, cut_off,True,False,save_after_steps=5)
    # A0, _ , _ = load_state("data/ground_state/gstate_ising2_D8_g1.5.npy")


    A0_tens = np.load("data/ground_state/tfim_AL_D12_g1.5.npz")["A"]
    A0=UniformMps(A0_tens)
    A0.is_isometry()
    start1 = time.time()


    Execute_Run(dt, steps, tol, iter, Ddim, cut_off, save = False, debug = True, A_init = A0, save_after_steps = 3, g=-0.2,h=0, J=-1, L=4, theta =  np.pi / 2, phi =  np.pi / 2, run_folder="integrable_test")
    end1 = time.time()

    dt = 1e-4
    steps = int(0.2//dt)
    steps = 30
    start = time.time()


    Execute_Run(dt, steps, tol, iter, Ddim, cut_off, save = False, debug = True, A_init = A0, save_after_steps = 30, g=-0.2,h=0, J=-1, L=4, theta =  np.pi / 2, phi =  np.pi / 2, run_folder="integrable_test")
    end = time.time()
    print("Elapsed time for small:", end - start, "seconds")
    print("Elapsed time for big:", end1 - start1, "seconds")

