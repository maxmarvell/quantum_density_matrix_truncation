from src.qdmt.evolve import *
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from examples.data_management import *

import os
import matplotlib.pyplot as plt




from pathlib import Path

import numpy as np
import re


import time
import re
import numpy as np
from pathlib import Path

def merge_temp_files(tmp_dir: Path, output_file: Path):
    tmp_dir = Path(tmp_dir)

    files = list(tmp_dir.glob("chunk*.npz"))
    if not files:
        print("WARNING: No temporary files found in", tmp_dir)
        return

    def extract_chunk_number(path: Path) -> int:
        m = re.search(r"chunk(\d+)", path.stem)
        return int(m.group(1)) if m else -1

    files = sorted(files, key=extract_chunk_number)

    times_all, states_all, costs_all, norms_all, durations_all, maxiters_all = [], [], [], [], [], []

    for f in files:
        print("Reading:", f)
        data = np.load(f, allow_pickle=True)

        times_all.append(np.asarray(data["time"]))
        states_all.append(np.asarray(data["state"]))
        costs_all.append(np.asarray(data["cost"]))
        norms_all.append(np.asarray(data["gradient_norm"]))
        durations_all.append(np.asarray(data["duration"]))
        maxiters_all.append(np.asarray(data["maxiter"]))

    times_all = np.concatenate(times_all)
    states_all = np.concatenate(states_all, axis=0)
    costs_all = np.concatenate(costs_all)
    norms_all = np.concatenate(norms_all)
    durations_all = np.concatenate(durations_all)
    maxiters_all = np.concatenate(maxiters_all)

    idx = np.argsort(times_all)
    times_sorted = times_all[idx]
    states_sorted = states_all[idx]
    costs_sorted = costs_all[idx]
    norms_sorted = norms_all[idx]
    durations_sorted = durations_all[idx]
    maxiters_sorted = maxiters_all[idx]

    np.savez_compressed(
        output_file,
        time=times_sorted,
        state=states_sorted,
        cost=costs_sorted,
        gradient_norm=norms_sorted,
        duration=durations_sorted,
        maxiter=maxiters_sorted,
    )

    print(f"✓ Successfully merged {len(files)} chunks → {output_file}")
    print(f"✓ time range: {times_sorted[0]} → {times_sorted[-1]}")


def Execute_Run(
    dt: float,
    steps: int,
    tolerance: float,
    iterations: int,
    D: int,
    cut_off: float,  # used BOTH as evolve cut_off and as wall-clock cutoff (seconds)
    save: bool = True,
    debug: bool = False,
    A_init=None,
    save_after_steps: int = 50,
    g: float = 1.05,
    h: float = -0.5,
    J: float = -1,
    L: int = 4,
    theta: float = np.pi / 2,
    phi: float = np.pi / 2,
    run_folder: str | Path = None,
):
    """
    Runs evolution in chunks, saving chunk*.npz into a temp directory and merging at the end.

    NOTE (as requested): `cut_off` is used in two ways:
      1) passed into evolve(..., cut_off=...)
      2) interpreted as a WALL-CLOCK limit in seconds; after each chunk is saved, we stop if elapsed >= cut_off

    This is intentionally inconsistent but convenient for your workflow.
    """

    if save is False:
        print("WARNING: save=False (this function still writes chunk files + merged output).")

    wall_start = time.time()

    # --- initial state ---
    psi = np.array([np.cos(theta / 2), np.exp(phi * 1j) * np.sin(theta / 2)])
    if A_init is None:
        A_init = UniformMps(psi.reshape(1, 2, 1))

    filepath = filepath_gen(dt,
                             steps, tolerance, iterations, D, cut_off, run_folder)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if debug:
        filepath = Path(str(filepath) + "_debug")

    print("Output file:", filepath)
    assert check_write_permission(filepath)

    model = TransverseFieldIsing(g=g, delta_t=dt, h=h, J=J)
    print("Model params:", [g, J, h])

    time_total = steps * dt
    time_chunk = save_after_steps * dt

    tmp_dir = filepath.parent / (".tmp_" + filepath.name)
    tmp_dir.mkdir(exist_ok=True)
    print("Temporary directory:", tmp_dir)

    start_times = np.arange(0.0, time_total, time_chunk)

    chunk_id = 0
    for this_start in start_times:
        max_t = min(this_start + time_chunk, time_total)

        times, state, cost, norm, duration, maxiter = evolve(
            A_init,
            D,
            L,
            model,
            dt,
            max_t,
            iterations,
            tolerance,
            cut_off,          # evolve still receives cut_off
            start_t=this_start,
        )

        A_init = UniformMps(state[-1])

        chunk_file = tmp_dir / f"chunk{chunk_id}.npz"
        print("Writing:", chunk_file)
        np.savez_compressed(
            chunk_file,
            time=times,
            state=state,
            cost=cost,
            gradient_norm=norm,
            duration=duration,
            maxiter=maxiter,
        )
        chunk_id += 1

        # --- wall-clock cutoff check AFTER saving a chunk ---
        elapsed = time.time() - wall_start
        if cut_off is not None and elapsed >= cut_off:
            print(f"⏱ Wall-time cutoff reached ({elapsed:.1f}s >= {cut_off:.1f}s).")
            print("Merging partial results and stopping cleanly...")
            merge_temp_files(tmp_dir, filepath)
            print("✓ Stopped due to wall-time cutoff (partial run saved).")
            return

    print("Merging temporary files...")
    merge_temp_files(tmp_dir, filepath)
    print("✓ Finished run.")


def _extract_chunk_number(path: Path) -> int:
    m = re.search(r"chunk(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def Execute_Run_Resume(
    dt: float,
    steps: int,
    tolerance: float,
    iterations: int,
    D: int,
    cut_off: float,  # used BOTH as evolve cut_off and as wall-clock cutoff (seconds), per your request
    save: bool = True,
    debug: bool = False,
    A_init=None,
    save_after_steps: int = 50,
    g: float = 1.05,
    h: float = -0.5,
    J: float = -1,
    L: int = 4,
    theta: float = np.pi / 2,
    phi: float = np.pi / 2,
    run_folder: str | Path = None,
    tmp_dir_override: str | Path | None = None,  # NEW: optional location of chunk*.npz
):
    """
    Resume a previously started chunked run.

    - Determines output filepath via filepath_gen(...) (same as Execute_Run).
    - Uses tmp_dir = filepath.parent / (".tmp_" + filepath.name) unless tmp_dir_override is provided.
    - Finds the last VALID chunk file, resumes from its final time and final state.
    - Continues writing chunk{n}.npz.
    - Merges at the end (or merges partial results and returns when wall-clock cutoff is reached).

    NOTE: `cut_off` is intentionally used inconsistently:
      (1) passed into evolve(...)
      (2) treated as wall-clock limit in seconds, checked after each saved chunk
    """

    if save is False:
        print("WARNING: save=False (this function still writes chunk files + merged output).")

    wall_start = time.time()

    # Output filepath consistent with Execute_Run
    filepath = filepath_gen(dt, steps, tolerance, iterations, D, cut_off, run_folder)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if debug:
        filepath = Path(str(filepath) + "_debug")

    print("Output file:", filepath)
    assert check_write_permission(filepath)

    time_total = steps * dt
    time_chunk = save_after_steps * dt

    # Decide temp directory containing chunk files
    if tmp_dir_override is not None:
        tmp_dir = Path(tmp_dir_override)
    else:
        tmp_dir = filepath.parent / (".tmp_" + filepath.name)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    print("Temporary directory:", tmp_dir)

    # Find existing chunks
    files = sorted(tmp_dir.glob("chunk*.npz"), key=_extract_chunk_number)

    # If no chunks exist, start a fresh run
    if not files:
        print("No chunk files found — starting fresh run.")
        return Execute_Run(
            dt, steps, tolerance, iterations, D, cut_off,
            save=save, debug=debug, A_init=A_init, save_after_steps=save_after_steps,
            g=g, h=h, J=J, L=L, theta=theta, phi=phi, run_folder=run_folder
        )

    # Load last valid chunk (robust to corrupted/empty last file)
    last_data = None
    last_file = None
    for f in reversed(files):
        try:
            data = np.load(f, allow_pickle=True)
            t = np.asarray(data["time"])
            s = np.asarray(data["state"])
            if t.size == 0 or s.shape[0] == 0:
                continue
            last_data = data
            last_file = f
            break
        except Exception as e:
            print("Skipping unreadable chunk:", f, "error:", e)

    if last_data is None:
        raise RuntimeError(f"Found chunk files in {tmp_dir}, but none were readable/valid.")

    last_chunk_id = _extract_chunk_number(last_file)
    last_times = np.asarray(last_data["time"])
    last_state = np.asarray(last_data["state"])

    resume_t = float(last_times[-1])

    print(f"Found last chunk: chunk{last_chunk_id} ({last_file.name}), last time t={resume_t:.6g}")

    # If already complete, just merge and return
    if resume_t >= time_total:
        print(f"Run already complete (t={resume_t:.6g} >= time_total={time_total:.6g}). Merging...")
        merge_temp_files(tmp_dir, filepath)
        return

    # Build A_init from last saved state
    A_init = UniformMps(last_state[-1])

    # Model
    model = TransverseFieldIsing(g=g, delta_t=dt, h=h, J=J)
    print("Model params:", [g, J, h])

    # Determine next chunk id
    next_chunk_id = last_chunk_id + 1
    chunk_id = next_chunk_id

    # Continue chunks: start at the last saved time, step by time_chunk
    this_start = resume_t


    while this_start < time_total:
        max_t = min(this_start + time_chunk, time_total)

        times, state, cost, norm, duration, maxiter = evolve(
            A_init,
            D,
            L,
            model,
            dt,
            max_t,
            iterations,
            tolerance,
            cut_off,          # evolve still receives cut_off
            start_t=this_start,
        )

        # Safety: avoid infinite loop if evolve doesn't advance time
        times_arr = np.asarray(times)
        if times_arr.size == 0:
            raise RuntimeError("evolve() returned empty time array during resume.")
        new_t = float(times_arr[-1])
        if new_t <= this_start + 1e-14:
            raise RuntimeError(
                f"Time did not advance during resume (start={this_start}, end={new_t}). "
                "Check evolve() output."
            )

        A_init = UniformMps(state[-1])

        chunk_file = tmp_dir / f"chunk{chunk_id}.npz"
        print("Writing:", chunk_file)
        np.savez_compressed(
            chunk_file,
            time=times,
            state=state,
            cost=cost,
            gradient_norm=norm,
            duration=duration,
            maxiter=maxiter,
        )

        chunk_id += 1
        this_start = new_t

        # Wall-clock cutoff check AFTER saving a chunk
        elapsed = time.time() - wall_start
        if cut_off is not None and elapsed >= cut_off:
            print(f"⏱ Wall-time cutoff reached ({elapsed:.1f}s >= {cut_off:.1f}s).")
            print("Merging partial results and stopping cleanly...")
            merge_temp_files(tmp_dir, filepath)
            print("✓ Stopped due to wall-time cutoff (partial run saved).")
            return

    print("Merging temporary files...")
    merge_temp_files(tmp_dir, filepath)
    print("✓ Finished resumed run.")



if __name__ == "__main__":
    # print('test')


    # from qdmt.model import TransverseFieldIsing
    # from qdmt.cost import EvolvedHilbertSchmidt
    # from qdmt.manifold import Grassmann



    # from pathlib import Path
    # import numpy as np

    # # Your chunk directory
    # tmp_dir = Path("/Users/phys2259/Documents/qdmt/results/temp_imports/temp_runs")

    # # Choose where to store the merged file
    # # (pick any name you like)
    # output_file = RESULTS_DIR / "integrable_test/merged_temp_imports.npz"

    # # Call your existing merge function
    # merge_temp_files(tmp_dir, output_file)

    # # Load merged data to verify
    # data = np.load(output_file, allow_pickle=True)

    # print("Merged file loaded.")
    # print("time shape:", data["time"].shape)
    # print("state shape:", data["state"].shape)
    # print("cost shape:", data["cost"].shape)
    # print("gradient_norm shape:", data["gradient_norm"].shape)

    # sys.exit()

    dt = 1e-3
    steps = int(20//dt)
    # # steps = 3
    tol =1e-7
    iter=300
    Ddim = 12

    hour = 3600
    days = 24*hour
    
    cut_off = 4*days
    # Execute_Run_Resume(dt, steps, tol, iter, Ddim, cut_off, run_folder="integrable_test")

    # Execute_Run(dt, steps, tol, iter, Ddim, cut_off,True,False,save_after_steps=5)
    # A0, _ , _ = load_state("data/ground_state/gstate_ising2_D8_g1.5.npy")


    A0_tens = np.load("data/ground_state/tfim_AL_D12_g1.5.npz")["A"]
    A0=UniformMps(A0_tens)
    A0.is_isometry()
    
    # start1 = time.time()


    # Execute_Run(dt, steps, tol, iter, Ddim, cut_off, save = False, debug = True, A_init = A0, save_after_steps = 3, g=-0.2,h=0, J=-1, L=4, theta =  np.pi / 2, phi =  np.pi / 2, run_folder="integrable_test")
    # end1 = time.time()

    # dt = 1e-4
    # steps = int(0.2//dt)
    # steps = 30
    # start = time.time()


    # Execute_Run(dt, steps, tol, iter, Ddim, cut_off, save = True, debug = False, A_init = A0, save_after_steps = 50, g=-0.2,h=0, J=-1, L=4, theta =  np.pi / 2, phi =  np.pi / 2, run_folder="integrable_test")
    # Execute_Run_Resume(dt, steps, tol, iter, Ddim, cut_off, save_after_steps = 50, g=-0.2,h=0, J=-1, L=4, theta =  np.pi / 2, phi =  np.pi / 2, run_folder="integrable_test")
    
    # THE LONG HIGH PRECISION D=8 run which had to be aborted because it took forever.
    # Execute_Run_Resume(dt, steps, tol, iter, Ddim, cut_off, save_after_steps = 50, g=-0.2,h=0, J=-1, L=4, theta =  np.pi / 2, phi =  np.pi / 2, run_folder="integrable_test", tmp_dir_override='/Users/phys2259/Documents/qdmt/results/integrable_test/.tmp_benchmark_dt=0.001_steps=19999_tol=1e-09_it=10000_D=8_cut=345600')
    
    
    # end = time.time()
    # print("Elapsed time for small:", end - start, "seconds")
    # print("Elapsed time for big:", end1 - start1, "seconds")

