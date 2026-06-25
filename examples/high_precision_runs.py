from examples.data_management import load_data, unfold_data, load_unfolded_data, sort_and_save_dataset, extend_unfolded_data
from examples.analyze_first import plot_unfolded_twofields, plot_over_time
import numpy as np
import matplotlib.pyplot as plt
from qdmt.analysis.tools import *

from scipy.integrate import quad


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

# non_integrable_D12_high_precision_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off)






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
# non_integrable_D8_unfolded=load_unfolded_data(dt, steps, tol, iter, Ddim, cut_off, dir = "sorted")

# sys.exit()





# === Dataset 1.3 Non-integrable model ===

dt = 1e-3
steps = int(20//dt)
tol =1e-11
iter=40000
Ddim = 8
cut_off = 2*days


non_integrable_D8_high_dat=np.load('/Users/phys2259/Documents/qdmt/results/sorted/benchmark_dt=0.001_steps=19999_tol=1e-11_it=10000_D=8_cut=172800.npz')

# unfold_data(non_integrable_D8_high_dat,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=True,
#     dir = "sorted"
# )

# non_integrable_D8_high_unfolded=load_unfolded_data(dt, steps, tol, iter, Ddim, cut_off, dir = "sorted")

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

# integrable_D12_unfolded=load_unfolded_data(dt, steps, tol, iter, Ddim, cut_off, dir = "integrable_test")






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

# integrable_D8_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off, dir="integrable_test")





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

#  integrable_D8_low_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off, dir="integrable_test")




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

#  integrable_D4_low_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off, dir="integrable_test")




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



#
# 
#  integrable_D12_dat_low=np.load("/Users/phys2259/Documents/qdmt/results/integrable_test/merged_d12_low_partial.npz")


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

# integrable_D12_low_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off, dir="integrable_test")


# ====== QM RUN DATASETS ====



dt = 1e-3
steps = int(2//dt)
# # steps = 3
tol =1e-11
iter=10000
Ddim = 2

hour = 3600
days = 24*hour

cut_off = 1*hour


qm_dat_raw=np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.001_steps=1999_tol=1e-11_it=10000_D=2_cut=3600.npz')

# unfold_data(qm_dat_raw,
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
#     dir = "integrable_D2_forQM"
# )

# sys.exit()
qm_dat_unfolded=load_unfolded_data(dt, steps, tol,iter,Ddim,cut_off=cut_off, dir="integrable_D2_forQM")

# odd H only


dt = 1e-3
steps = int(2//dt)
# # steps = 3
tol =1e-11
iter=10000
Ddim = 2

hour = 3600
days = 24*hour

cut_off = 1*hour

qm_dat_odd_only_raw=np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.001_steps=1999_tol=1e-11_it=10000_D=2_cut=3600_debug.npz')

# qm_dat_unfolded_odd_only = unfold_data(qm_dat_odd_only_raw,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=False,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_D2_forQM"
# )



qm_dat_odd_only_sym_raw=np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.001_steps=1999_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=0).npz')

# qm_dat_unfolded_odd_sym_only = unfold_data(qm_dat_odd_only_sym_raw,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=False,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_D2_forQM"
# )




# odd and even H average 
qm_dat_odd_even_average_raw=np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.001_steps=1999_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=-1).npz')

# qm_dat_unfolded_odd_even_average = unfold_data(qm_dat_odd_even_average_raw,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=False,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_D2_forQM"
# )


# odd and even H average with giant trotter

dt=0.2
steps = int(2//dt)
# # steps = 3
tol =1e-11
iter=10000
Ddim = 2

hour = 3600
days = 24*hour

cut_off = 1*hour
qm_dat_odd_even_average_big_trotter_raw_=np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.2_steps=9_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=-1).npz')

# qm_dat_odd_even_average_big_trotter_ = unfold_data(qm_dat_odd_even_average_big_trotter_raw_,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=False,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_D2_forQM"
# )


qm_dat_buffer_big_trotter_raw_=np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.2_steps=9_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=1).npz')

# qm_dat_buffer_big_trotter_ = unfold_data(qm_dat_buffer_big_trotter_raw_,
#     dt,
#     steps,
#     tol,
#     iter,
#     Ddim,
#     cut_off,
#     overwrite=False,
#     L=L,
#     g=g,
#     h=h,
#     J=J,
#     dir = "integrable_D2_forQM"
# )

qm_dat_big_trotter_sym_raw_=np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM_debug/benchmark_dt=0.1_steps=19_tol=1e-12_it=10000_D=2_cut=3600(_Trotter_order=0).npz')

trotter_huge=np.load('benchmark_dt=0.2_steps=9_tol=1e-14_it=10000_D=2_cut=3600(_Trotter_order=-1).npz')

# sys.exit()



#  exact LS


def f(z, g0, g1):
    def theta(k, g):
        return np.arctan2(np.sin(k), g-np.cos(k))/2
    def phi(k, g0, g1):
        return theta(k, g0)-theta(k, g1)
    def epsilon(k, g1):
        return -2*np.sqrt((g1-np.cos(k))**2+np.sin(k)**2)
    def integrand(k):
        return -1/(2*np.pi)*np.log(np.cos(phi(k, g0, g1))**2 + np.sin(phi(k, g0, g1))**2 * np.exp(-2*z*epsilon(k, g1)))

    return quad(integrand, 0, np.pi)[0]

def loschmidt_paper(t, g0, g1):
    return (f(t*1j, g0, g1)+f(-1j*t, g0, g1))





if __name__ == "__main__":

    # plot_over_time

    # data=non_integrable_D8_unfolded
    # data2=non_integrable_D8_high_unfolded
    # data=integrable_D8_unfolded

    # data=qm_dat_unfolded
    # data2=qm_dat_unfolded_odd_only
    # data3=qm_dat_unfolded_odd_even_average



    # data_big=qm_dat_odd_even_average_big_trotter_


    # data_big_buffer=qm_dat_buffer_big_trotter_
    g0=1.5
    g1=0.2 
    data_exact = {}
    dt=0.2
    # t = 0.0 + dt* np.arange(len(data_big["loschmidt"]))  # match your plot time grid
    # data_exact["loschmidt"] = np.array([loschmidt_paper(tt, g0, g1) for tt in t])



    # plot_over_time(
    #     # (data, "loschmidt", "LM 2-layer brickwall with buffer"),
    #     # (data2, "loschmidt", "LM 1-layer, H_even only with rescaled time step"),
    #     # (data3, "loschmidt", "LM 1-layer, H_even and H_odd average"),
    #     (data_big,"loschmidt", "LM BIG AVG trotter"),
    #     (data_big_buffer,"loschmidt", "LM BIG BUFFER trotter"),
    #     (data_exact, "loschmidt", "LM exact"),
    # dt=dt,
    # t_max=2,
    # title="Loschmidt over time"
    # )


    dt_exact = 0.1
    t_exact = np.arange(0, 2 + 1e-12, dt_exact)
    data_exact["loschmidt"] = np.array([loschmidt_paper(tt, g0, g1) for tt in t_exact])

    qm_dat_unfolded_odd_only_sym_big = np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.2_steps=9_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=0).npz')
    qm_dat_unfolded_odd_only_avg_big = np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.2_steps=9_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=-1).npz')
    qm_dat_unfolded_odd_only_avg_less_big = np.load('/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM/benchmark_dt=0.1_steps=19_tol=1e-11_it=10000_D=2_cut=3600(_Trotter_order=-1).npz')
    

    plot_over_time(
    # (qm_dat_unfolded_odd_only_sym_big,         "cost", "const even only",    dt_exact,       {"marker":"o", "linestyle":"-","markersize":3}),
    # (qm_dat_unfolded_odd_only_avg_big,  "cost", "const avg",    dt_exact,       {"marker":"o", "linestyle":"-","markersize":3}),
    (qm_dat_unfolded_odd_only_avg_less_big, "cost", "const avg",    dt_exact,       {"marker":"o", "linestyle":"-","markersize":3}),
    # (qm_dat_unfolded_odd_sym_only, "loschmidt", "LM trotter even only sym", dt_exact,       {"marker":"o", "linestyle":"-","markersize":3}),
    # (data_big, "cost", "cost BIG BUFFER trotter", dt,       {"marker":"o", "linestyle":"-","markersize":3}),

    # (data_exact,      "loschmidt", "LM exact",              dt_exact, {"linestyle":"-"}),  # smooth line
    dt=dt,
    t_max=2,
    title="Cost over time",
)


    plot_over_time(
    (qm_dat_unfolded_odd_only,        "loschmidt", "LM trotter even only",    dt_exact,       {"marker":"o", "linestyle":"-","markersize":3}),
    (qm_dat_unfolded_odd_sym_only, "loschmidt", "LM trotter even only sym", dt_exact,       {"marker":"o", "linestyle":"-","markersize":3}),
    # (data_big, "cost", "cost BIG BUFFER trotter", dt,       {"marker":"o", "linestyle":"-","markersize":3}),

    (data_exact,      "loschmidt", "LM exact",              dt_exact, {"linestyle":"-"}),  # smooth line
    dt=dt,
    t_max=2,
    title="Loschmidt over time",
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