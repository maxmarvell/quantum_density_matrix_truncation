from pathlib import Path
from examples.get_feeling_for_parameters import merge_temp_files
import numpy as np
from examples.data_management import *


if __name__ == "__main__":


    # Your chunk directory
    tmp_dir = Path("/Users/phys2259/Documents/qdmt/results/temp_imports/d12_low_partial")

    # Choose where to store the merged file
    # (pick any name you like)
    output_file = RESULTS_DIR / "integrable_test/merged_d12_low_partial.npz"

    # Call your existing merge function
    merge_temp_files(tmp_dir, output_file)

    # Load merged data to verify
    data = np.load(output_file, allow_pickle=True)

    print("Merged file loaded.")
    print("time shape:", data["time"].shape)
    print("state shape:", data["state"].shape)
    print("cost shape:", data["cost"].shape)
    print("gradient_norm shape:", data["gradient_norm"].shape)

    # sys.exit()