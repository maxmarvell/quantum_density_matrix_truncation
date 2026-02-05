from examples.data_management import load_data, unfold_data, load_unfolded_data, sort_and_save_dataset, extend_unfolded_data
from examples.analyze_first import plot_unfolded_twofields, plot_over_time
import numpy as np
import matplotlib.pyplot as plt
from qdmt.analysis.tools import *

from pathlib import Path
import sys



###### desired data sets
# D=16 non integrable
# integrable:  L=4, D=2,4,8,12,16 
# smaller dt 
# ground state of the TFIM for g0 = 1.5 and a quench TFIM parametrised by g1 = 0.2

hour = 3600
days = 24*hour


# === Dataset 1.1 Non-integrable model, g=1.05, h=09.5, J=-1 ===

dt = 1e-3
steps = int(20//dt)
tol =1e-11
iter=40000
Ddim = 12
cut_off = 2*days
# D12_high_precision_raw=load_data(dt, steps, tol, iter, Ddim, cut_off)


# # data_sorted, outpath = sort_and_save_dataset(D12_high_precision_raw, filepath)

# # D12_high_precision_sorted=load_data(dt, steps, tol, iter, Ddim, cut_off, dir="sorted")
# # D12_high_precision_=unfold_data(D12_high_precision_sorted,dt,steps,tolerance=tol,iterations=iter,D=12,cut_off=cut_off, overwrite=True)

non_integrable_D12_high_precision_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off)






# === Dataset 1.2 Non-integrable model ===

dt = 1e-3
steps = int(20//dt)
tol =1e-9
iter=40000
Ddim = 8
cut_off = 2*days


non_integrable_D8_dat=np.load('/Users/phys2259/Documents/qdmt/results/sorted/benchmark_dt=0.001_steps=19999_tol=1e-09_it=10000_D=8_cut=172800.npz')

# unfold_data(non_integrable_D8_dat,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=True,
#     dir = "sorted"
# )
non_integrable_D8_unfolded=load_unfolded_data(dt, steps, tol, iter, Ddim, cut_off, dir = "sorted")

# sys.exit()


# === Dataset 2 ===
#  integrable work laptop D=12 run, high accuracy, finished

dt = 1e-3
steps = int(20//dt)
tol =1e-9
iter=10000
Ddim = 12


cut_off = 2*days
g=-0.2
h=0
J=-1 
L=4



# integrable_D12_dat=np.load("/Users/phys2259/Documents/qdmt/results/integrable_test/benchmark_dt=0.001_steps=19999_tol=1e-09_it=10000_D=12_cut=172800.npz")

# unfold_data(integrable_D12_dat,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=True,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_test"
# )

integrable_D12_unfolded=load_unfolded_data(dt, steps, tol, iter, Ddim, cut_off, dir = "integrable_test")






# === Dataset 3 ===
#  integrable personal laptop D=8 run, high accuracy, unfinished

dt = 1e-3
steps = int(20//dt)
tol =1e-9
iter=10000
Ddim = 8


cut_off = 4*days
g=-0.2
h=0
J=-1 
L=4




# integrable_D8_dat=np.load("/Users/phys2259/Documents/qdmt/results/integrable_test/benchmark_dt=0.001_steps=19999_tol=1e-09_it=10000_D=8_cut=345600.npz")
# unfold_data(integrable_D8_dat,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=True,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_test"
# )

integrable_D8_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off, dir="integrable_test")





# === Dataset 4 ===
#  integrable, laptop D=8 run, low accuracy, unfinished

dt = 1e-3
steps = int(20//dt)
tol =1e-7
iter=300
Ddim = 8


cut_off = 4*days
g=-0.2
h=0
J=-1 
L=4




# integrable_D8_dat_low=np.load("/Users/phys2259/Documents/qdmt/results/integrable_test/benchmark_dt=0.001_steps=19999_tol=1e-07_it=300_D=8_cut=345600.npz")
# unfold_data(integrable_D8_dat_low,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=True,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_test"
# )

integrable_D8_low_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off, dir="integrable_test")




# === Dataset 5 ===
#  integrable, laptop D=4 run, low accuracy, unfinished

dt = 1e-3
steps = int(20//dt)
tol =1e-7
iter=300
Ddim = 4


cut_off = 4*days
g=-0.2
h=0
J=-1 
L=4




# integrable_D4_dat_low=np.load("/Users/phys2259/Documents/qdmt/results/integrable_test/benchmark_dt=0.001_steps=19999_tol=1e-07_it=300_D=4_cut=345600.npz")
# unfold_data(integrable_D4_dat_low,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=True,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_test"
# )

integrable_D4_low_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off, dir="integrable_test")




# === Dataset 6 ===
#  integrable, laptop D=12 run, low accuracy, unfinished

dt = 1e-3
steps = int(20//dt)
tol =1e-7
iter=300
Ddim = 12


cut_off = 4*days
g=-0.2
h=0
J=-1 
L=4



# integrable_D12_dat_low=np.load("/Users/phys2259/Documents/qdmt/results/integrable_test/merged_d12_low_partial.npz")
integrable_D12_dat_low=np.load("/Users/phys2259/Documents/qdmt/results/integrable_test/benchmark_dt=0.001_steps=19999_tol=1e-07_it=300_D=12_cut=345600.npz")
# unfold_data(integrable_D12_dat_low,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=True,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_test"
# )

integrable_D12_low_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off, dir="integrable_test")






if __name__ == "__main__":

    # plot_over_time

    data=non_integrable_D8_unfolded
    # data=integrable_D8_unfolded

    plot_over_time(
        (data, "cost", "D=8 high precision cost"),
        (data, "energy", "D=8 high precision energy"),
        (data,"von_neumann_entropy", "entroy"),
        dt=0.001,
        t_max=20,
        title="Renyi over time"
    )


    # # data=integrable_D8_unfolded
    # data=integrable_D8_low_unfolded

    # left="energy"
    # # right="dist_steady"
    # # left="von_neumann_entropy"
    # # left="l_magnetization"

    # # right="renyi_entropy"
    # right="norm"
    # # right="energy"
    # # right="trace_distance_dt"
    # right="dist_steady"

    # plot_unfolded_twofields(
    # (data, "D=8"),
    # field_left=left,
    # field_right=right,
    # title=left+" and "+right
    # )