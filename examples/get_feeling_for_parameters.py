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

def Execute_Run(dt: float, steps: int, tolerance: float, iterations: int, D: int, cut_off: float, save = True, debug = False, A_init = None, save_after_steps = 50):
    if save == False:
        print("WARNING, NOT SAVING THE DATA!")

    ##### fixed things

    L = 4

    theta = phi = np.pi / 2
    psi = np.array([np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)])
    if A_init is None:
        A_init = UniformMps(psi.reshape(1, 2, 1))


    ##### fixed things

    filepath = filepath_gen(dt, steps, tolerance, iterations, D, cut_off)

    print(str(filepath))

    if debug:
        filepath = Path(str(filepath) + "_debug")


        

    assert check_write_permission(filepath)

    # model = TransverseFieldIsing(g=0, delta_t=dt, h=0, J=0)
    model = TransverseFieldIsing(g=1.05, delta_t=dt, h=-0.5, J=-1)

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







def Resume_Run(dt: float, steps: int, tolerance: float, iterations: int, D: int, cut_off_previous: float, cut_off_now: float):
    data = load_data(dt, steps, tolerance, iterations, D, cut_off_previous)

    valid_len = count_initial_nonzero(data["cost"])-2
    keys = ["time", "state", "cost", "gradient_norm", "duration"]
    times, state, cost, norm, duration = [data[k][:valid_len] for k in keys]

    filepath = filepath_gen(dt, steps, tolerance, iterations, D, cut_off_now)+"test"
    assert check_write_permission(filepath)

    model = TransverseFieldIsing(g=1.05, delta_t=dt, h=-0.5, J=-1)
    time_total = steps*dt

    t_start = times[-1]

    print(cut_off_now)

    print(cost[-1])
    
    A_init = UniformMps(state[-1])
    print("go")
    new_times, new_state, new_cost, new_norm, new_duration = evolve(A_init, D, L, model, dt, time_total, iterations, tolerance, cut_off_now, t_start)
    
    
    combi_times = np.concatenate(times,new_times)
    combi_state = np.concatenate(state,new_state)
    combi_cost = np.concatenate(cost,new_cost)
    combi_norm = np.concatenate(norm,new_norm)
    combi_duration = np.concatenate(duration,new_duration)

    # np.savez_compressed(filepath,time=combi_times, state=combi_state, gradient_norm=combi_norm, cost=combi_cost, duration=combi_duration)



if __name__ == "__main__":
    # print('test')


    from qdmt.model import TransverseFieldIsing
    from qdmt.cost import EvolvedHilbertSchmidt
    from qdmt.manifold import Grassmann



    # benchmarks, some trotter steps, toleramces, iterations and bond dimensions



    # FIXED FOR NOW

    

    # steps = 100
    # # steps=100
    # cut_off= 1*60*60+4
    # # cut_off=60

    # delta_ts_ = [0.1*1e-2]
    # tolerances_ = [1e-10]
    # iterations_ = [1000]
    # Ds_ = [10]
    # iterations_ = [500]
    # Ds_ = [4,8]
    # delta_ts_=[1e-2]
    # tolerances_ = [1e-8]

    dt = 1e-3
    steps = int(20//dt)
    tol =1e-11
    iter=40000
    Ddim = 12

    hour = 3600
    days = 24*hour
    
    cut_off = 2*days

    # Execute_Run(dt, steps, tol, iter, Ddim, cut_off,True,False,save_after_steps=5)

    data=load_data(dt, steps, tol, iter, Ddim, cut_off)
    print(data["time"])
    print(data["cost"])
    print(data["maxiter"])
    print(data["duration"])

    

    times = data["time"]
    cost = data["cost"]
    duration = data["duration"]
    maxiter = data["maxiter"]

    print(len(times), len(cost), len(duration), len(maxiter))
    print(times[:20])

    time_arr = np.array(times)
    cost_arr = np.array(cost)

    plt.figure()
    # plt.plot(cost)
    plt.plot(duration)
    # plt.plot(maxiter)
    plt.xlabel("index")
    plt.ylabel("cost")
    plt.title("cost vs index")
    plt.tight_layout()
    plt.show()

    sys.exit()

    print("time_arr.shape:", time_arr.shape)
    print("cost_arr.shape:", cost_arr.shape)
    print("first element of time:", time_arr[0])
    print("first element of cost:", cost_arr[0])
    # plt.figure(figsize=(10,6))

    plt.plot(times, cost, label="cost")
    # plt.plot(time, duration, label="duration")
    # plt.plot(time, maxiter, label="maxiter")

    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Cost, Duration, Maxiter over Time")
    plt.legend()
    # plt.tight_layout()
    plt.show()

    # # Run with timeout
    # for tolerance in tolerances_:
    #     for iterations in iterations_:
    #         for dt in delta_ts_:
    #             for bondD in Ds_:
    #                 # print(f"benchmark_dt={dt}_steps={steps}_tol={tolerance}_it={iterations}_D={bondD}_cut={cut_off}")
    #                 # A_0 = spit_out_state(dt, steps, tolerance, iterations, bondD, cut_off, 0.79)
    #                 # Execute_Run(dt, 1, 1e-10, 1000, bondD, cut_off,False,True,A_0)
                    
    #                 # Execute_Run(dt, 100000, tolerance, 1000, 8, cut_off,True,False)
    #                 # Execute_Run(dt, 100000, tolerance, 1000, 12, cut_off,True,False)
    #                 Execute_Run(dt, 1, 1e-11, 10, 4, cut_off,True,True)
                    # A_0 = spit_out_state(dt, 1000, tolerance, 1000, 16, cut_off, 0.41)
                    # Execute_Run(dt, 100, tolerance, 1000, 16, cut_off,True,False)


                    # Resume_Run(dt, steps, tolerance, iterations, bondD, cut_off,60)
    # filepath = filepath_gen(dt: float, steps: int, tolerance: float, iterations: int, D: int, cut_off: float)
    # assert check_write_permission(filepath)


    # dtt=0.001

    # model = TransverseFieldIsing(g=1.05, delta_t=dtt, h=-0.5, J=-1)

    # theta = phi = np.pi / 2

    # psi = np.array([np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)])

    # A = UniformMps(psi.reshape(1, 2, 1))
   
    # summary=[] 
    # for D in [4]:
    #     print(D)
       
    #     t0 = time.time()
    #     times, state, cost, norm = evolve(A, D, 4, model, dtt, 2*dtt, 1000, 1e-8)
    #     t1 = time.time()
    #     summary.append([D,cost,t1-t0])    
    #     np.savez_compressed(filepath,
    #                         time=times,
    #                         state=state,
    #                         gradient_norm=norm,
    #                         cost=cost)
    #     print(summary)