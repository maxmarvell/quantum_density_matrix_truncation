from src.qdmt.evolve import *
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from examples.data_management import *

import os




from pathlib import Path

def merge_temp_files(tmp_dir, output_file):
    files = sorted(tmp_dir.glob("*chunk*.npz"))

    all_times = []
    all_states = []
    all_costs = []
    all_norms = []
    all_durations = []
    all_maxiters = []

    for f in files:
        data = np.load(f)
        all_times.append(data["time"])
        all_states.append(data["state"])
        all_costs.append(data["cost"])
        all_norms.append(data["gradient_norm"])
        all_durations.append(data["duration"])
        all_maxiters.append(data["maxiter"])

    times = np.concatenate(all_times)
    state = np.concatenate(all_states, axis=0)
    cost = np.concatenate(all_costs)
    norm = np.concatenate(all_norms)
    duration = np.concatenate(all_durations)
    maxiter = np.concatenate(all_maxiters)

    
    np.savez_compressed(
        output_file,
        time=times,
        state=state,
        gradient_norm=norm,
        cost=cost,
        duration=duration,
        maxiter=maxiter
    )


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


    tmp_dir = filepath.parent / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    print(tmp_dir)



    start_times = np.arange(0, time_total, time_chunk)

    chunk_id=0
    for this_start in start_times:
        max_t=this_start+time_chunk
        times, state, cost, norm, duration, maxiter = evolve(A_init, D, L, model, dt, max_t, iterations, tolerance, cut_off,start_t=this_start)
        # print(state)
        # print(this_start)
        A_init=UniformMps(state[-1])
       
        tmp_file = tmp_dir / (f"benchmark_dt={dt}_steps={steps}_tol={tolerance}_it={iterations}_D={D}_cut={cut_off}"+f"_chunk{chunk_id}.npz")

        chunk_id+=1
        print(tmp_file)
        np.savez_compressed(tmp_file ,time=times, state=state, gradient_norm=norm, cost=cost, duration=duration, maxiter=maxiter)
        merge_temp_files(tmp_dir, filepath)

  
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

    Execute_Run(dt, steps, tol, iter, Ddim, cut_off,True,False,save_after_steps=5)

    data=load_data(dt, steps, tol, iter, Ddim, cut_off)
    print(data["time"])
    print(data["cost"])
    print(data["maxiter"])
    print(data["duration"])


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