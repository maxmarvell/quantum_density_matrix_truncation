from examples.data_management import load_data, unfold_data, load_unfolded_data, sort_and_save_dataset, extend_unfolded_data
from examples.analyze_first import plot_unfolded_twofields
import numpy as np
import matplotlib.pyplot as plt
from qdmt.analysis.tools import *

from pathlib import Path
import sys


hour = 3600
days = 24*hour


# === Dataset 1 ===

dt = 1e-3
steps = int(20//dt)
tol =1e-11
iter=40000
Ddim = 12
cut_off = 2*days
# D12_high_precision_raw=load_data(dt, steps, tol, iter, Ddim, cut_off)


# data_sorted, outpath = sort_and_save_dataset(D12_high_precision_raw, filepath)

# D12_high_precision_sorted=load_data(dt, steps, tol, iter, Ddim, cut_off, dir="sorted")
# D12_high_precision_=unfold_data(D12_high_precision_sorted,dt,steps,tolerance=tol,iterations=iter,D=12,cut_off=cut_off, overwrite=True)


# plot_cost_vs_index(D12_high_precision_sorted)


# sys.exit()

# D12_high_precision_=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off)

# extend_unfolded_data(
#     dt, steps, tol, iter, Ddim, cut_off,
#     new_key="trace_distance_dt",
#     trajectory_fn=lambda states:compute_trace_distance_successive(states, L=4),
#     overwrite=True
# )


# extend_unfolded_data(
#     dt, steps, tol, iter, Ddim, cut_off,
#     new_key="trace_distance_to_avg",
#     trajectory_fn=lambda states: compute_trace_distance_to_average(
#         states,
#         dt=dt,
#         t_cut=10.0,   # example
#         L=4
#     ),
#     overwrite=True
# )


D12_high_precision_=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off)
# data_trimmed=D12_high_precision_raw
# load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off)


# print("Raw time shape:", data_trimmed["time"].shape)
# print("First 5 raw times:", data_trimmed["time"][:5])
# print("Last 5 raw times:", data_trimmed["time"][-5:])
# print("Max time:", np.max(data_trimmed["time"]))

# sys.exit()


if __name__ == "__main__":

    data=D12_high_precision_

    left="trace_distance_to_avg"
    # right="dist_steady"
    # left="von_neumann_entropy"
    # left="l_magnetization"

    # right="renyi_entropy"
    # right="von_neumann_entropy"
    # right="energy"
    right="trace_distance_dt"

    plot_unfolded_twofields(
    (D12_high_precision_, "D=12"),
    field_left=left,
    field_right=right,
    title=left+" and "+right
    )