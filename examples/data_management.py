import os
import numpy as np
from qdmt.analysis import tools
from qdmt.analysis import conserved_quantities as cq 
from src.qdmt.uniform_mps import UniformMps
from qdmt.analysis import magnetization as mag

from src.qdmt.evolve import *


from pathlib import Path

# Compute the absolute project root directory:
# If this file lives in qdmt/examples/, then parents[1] = qdmt/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print(PROJECT_ROOT)
# The results folder lives under project root
RESULTS_DIR = PROJECT_ROOT / "results"


def save_results(data, foldername, filename,override=False):
    """
    Saves data as a .npy file into results/
    filename: name without extension (e.g. "run1")
    """
    folder = RESULTS_DIR  / foldername
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, filename + ".npy")

    if os.path.exists(path) and not override:
        print(f"File {path} already exists. Not overwriting (override=False).")
        return
    np.save(path, data)
    print(f"Saved to {path}")


def load_results(foldername,filename):
    """
    Loads data from results/
    """

    folder = RESULTS_DIR  / foldername
    path = os.path.join(folder, filename + ".npy")
    data = np.load(path, allow_pickle=True)
    print(f"Loaded from {path}")
    return data




def filepath_gen(dt, steps, tolerance, iterations, D, cut_off):
    filename = f"benchmark_dt={dt}_steps={steps}_tol={tolerance}_it={iterations}_D={D}_cut={cut_off}"
    return RESULTS_DIR  / filename


# def filepath_gen(dt, steps, tolerance, iterations, D, cut_off):
#     filename = f"benchmark_dt={dt}_steps={steps}_tol={tolerance}_it={iterations}_D={D}_cut={cut_off}_trimmed_unfolded.npz"
#     return RESULTS_DIR / "datasets_unfolded" / filename



def load_data(dt, steps, tolerance, iterations, D, cut_off):
    # Build the filename
    filename = (
        f"benchmark_dt={dt}_steps={steps}_tol={tolerance}"
        f"_it={iterations}_D={D}_cut={cut_off}.npz"
    )

    # Full absolute path to the .npz file
    filepath = RESULTS_DIR / filename

    # Load and return the data
    return np.load(filepath)


def count_initial_nonzero(arr, cutoff=1e-16):
    """
    Count how many leading entries in `arr` have |value| > cutoff.
    Stops at the first element below the cutoff.
    """
    arr = np.asarray(arr)
    for i, val in enumerate(arr):
        if not np.isfinite(val) or abs(val) <= cutoff or abs(val) > 1 or val < 0:
            print(val)
            print(i)
            return i
            
    return len(arr)




def trim_data(data, l):
    """
    Trim all arrays in an np.load(...) result (npz file) to length l.
    Returns a new dict of trimmed arrays.
    """
    trimmed = {}
    for key in data.files:
        arr = data[key]
        trimmed[key] = arr[:l]
    return trimmed

def save_trimmed_data(data, dt, steps, tolerance, iterations, D, cut_off):
    # Get the original dataset path as a Path object
    original_path = filepath_gen(dt, steps, tolerance, iterations, D, cut_off)

    # Folder that will hold trimmed datasets: qdmt/results/datasets_trimmed
    trimmed_folder = RESULTS_DIR / "datasets_trimmed"
    trimmed_folder.mkdir(parents=True, exist_ok=True)

    # Same base name but with "_trimmed.npz"
    base_name = original_path.stem         # filename without .npz
    trimmed_name = base_name + "_trimmed.npz"

    save_path = trimmed_folder / trimmed_name

    # OPTIONAL: skip if already present
    # if save_path.exists():
    #     print(f"Trimmed dataset already exists: {save_path}")
    #     return

    # Save compressed
    np.savez_compressed(
        save_path,
        time=data["time"],
        state=data["state"],
        gradient_norm=data["gradient_norm"],
        cost=data["cost"],
        duration=data["duration"],
    )

    print(f"Trimmed dataset saved to: {save_path}")





def prep_data(dt, steps, tolerance, iterations, D, cut_off, timerange = None):
    data = load_data(dt, steps, tolerance, iterations, D, cut_off)   
    if timerange is None:
        valid_len = count_initial_nonzero(data["cost"])
        timerange = valid_len*dt
    else:
        valid_len = int(timerange // dt)      
    print(valid_len)    
    data = trim_data(data, valid_len)
    # --- save trimmed version ---
    save_trimmed_data(
        data,
        dt, steps, tolerance, iterations, D, cut_off
    )

    return data    


def load_trimmed_data(dt, steps, tolerance, iterations, D, cut_off):
    # Original base file path (Path object)
    original_path = filepath_gen(dt, steps, tolerance, iterations, D, cut_off)

    # Always use the global RESULTS_DIR / "datasets_trimmed"
    trimmed_folder = RESULTS_DIR / "datasets_trimmed"

    # Remove ".npz" from base filename, append "_trimmed.npz"
    trimmed_name = original_path.stem + "_trimmed.npz"
    load_path = trimmed_folder / trimmed_name

    # Check existence
    if not load_path.exists():
        print(f"Trimmed dataset not found: {load_path}")
        return None

    # Load the trimmed dataset
    data = np.load(load_path)

    print(f"Loaded trimmed dataset: {load_path}")

    return {
        "time": data["time"],
        "state": data["state"],
        "gradient_norm": data["gradient_norm"],
        "cost": data["cost"],
        "duration": data["duration"]
    }

def unfold_data(
    data_trimmed,
    dt,
    steps,
    tolerance,
    iterations,
    D,
    cut_off,
    overwrite=False,
    L=4,
    g=1.05,
    h=-0.5,
    J=-1
):
    """
    Take trimmed data (dict) and compute derived quantities:
    norm, energy, renyi_entropy, etc. Save as a new NPZ file in
    results/datasets_unfolded/ and return the extended dictionary.
    """

    state = data_trimmed["state"]
    cost = data_trimmed["cost"]

    # --- Derived quantities ---
    model_H = TransverseFieldIsing(g=g, delta_t=0.1, h=h, J=J)

    print("compute norm")
    norm = np.array([cq.compute_norm(UniformMps(x)) for x in state])

    print("compute energy")
    energy = np.array([cq.compute_energy(UniformMps(x), model_H) for x in state])

    print("compute renyi")
    renyi_entropy = np.array([tools.compute_second_Reyni(UniformMps(x), L) for x in state])

    print("compute log cost")
    log_cost = np.log(cost)

    # print("compute magnetization")
    # t_magnetization = np.array([mag.transverse_magnetization(UniformMps(x)) for x in state])

    # --- Build extended dict ---
    data_unfolded = {
        "time": data_trimmed["time"],
        "state": data_trimmed["state"],
        "gradient_norm": data_trimmed["gradient_norm"],
        "cost": data_trimmed["cost"],
        "duration": data_trimmed["duration"],
        "norm": norm,
        "energy": energy,
        "renyi_entropy": renyi_entropy,
        "log_cost": log_cost,
        # "t_magnetization": t_magnetization
    }

    # --- Save path construction ---

    # Base path for the original dataset (Path object)
    original_path = filepath_gen(dt, steps, tolerance, iterations, D, cut_off)

    # Save folder: qdmt/results/datasets_unfolded
    unfolded_folder = RESULTS_DIR / "datasets_unfolded"
    unfolded_folder.mkdir(parents=True, exist_ok=True)

    # Add suffix "_trimmed_unfolded"
    base_name = original_path.stem
    unfolded_name = base_name + "_trimmed_unfolded.npz"

    save_path = unfolded_folder / unfolded_name

    # Skip if exists
    if save_path.exists() and not overwrite:
        print(f"Unfolded dataset already exists: {save_path}")
        return data_unfolded

    # Save
    np.savez_compressed(
        save_path,
        **data_unfolded
    )

    print(f"Unfolded dataset saved to: {save_path}")

    return data_unfolded

def load_unfolded_data(dt, steps, tolerance, iterations, D, cut_off, L=4, g=1.05, h=-0.5, J=-1):
    """
    Load unfolded data saved in results/datasets_unfolded/.
    Returns a dict or None if file not found.
    """

    original_path = filepath_gen(dt, steps, tolerance, iterations, D, cut_off)

    # Correct folder
    unfolded_folder = RESULTS_DIR / "datasets_unfolded"

    # Avoid double suffix
    base = original_path.stem.replace("_trimmed_unfolded", "")
    unfolded_name = base + "_trimmed_unfolded.npz"

    load_path = unfolded_folder / unfolded_name

    if not load_path.exists():
        print(f"Unfolded dataset not found: {load_path}")
        return None

    data = np.load(load_path)

    print(f"Loaded unfolded dataset: {load_path}")

    return {
        "time": data["time"],
        "state": data["state"],
        "gradient_norm": data["gradient_norm"],
        "cost": data["cost"],
        "duration": data["duration"],
        "norm": data["norm"],
        "energy": data["energy"],
        "renyi_entropy": data["renyi_entropy"],
        "log_cost": data["log_cost"],
    }



def generate_trimmed_and_unfolded(
    dt,
    steps,
    tolerance,
    iterations,
    D,
    cut_off,overwrite = False,
    timerange=None,L=4,g=1.05, h=-0.5, J=-1
):
    """
    Full processing pipeline:
    1. Load + trim data  (prep_data)
    2. Save trimmed dataset
    3. Compute derived quantities (norm, energy, renyi)
    4. Save unfolded dataset
    Returns (trimmed_data, unfolded_data).
    """

    print("\n--- Generating trimmed dataset ---")
    trimmed = prep_data(
        dt,
        steps,
        tolerance,
        iterations,
        D,
        cut_off,
        timerange=timerange
    )

    print("\n--- Generating unfolded dataset ---")
    unfolded = unfold_data(
        trimmed,
        dt,
        steps,
        tolerance,
        iterations,
        D,
        cut_off,overwrite
    )

    print("\n--- Done ---")
    return trimmed, unfolded



def spit_out_state(dt, steps, tolerance, iterations, D, cut_off, timerange = None):
    # RETURN THE STATE ONE STEP BEFORE TIMERANGE
    data = prep_data(dt, steps, tolerance, iterations, D, cut_off, timerange)
    state=data["state"]
    cost=data["cost"]
    print(f"cost here={cost[-1]}\nprevious cost was={cost[-2]}")
    return UniformMps(state[-2])