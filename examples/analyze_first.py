
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from qdmt.analysis import tools
from qdmt.analysis import conserved_quantities as cq 

from examples.data_management import *

from src.qdmt.uniform_mps import UniformMps
from src.qdmt.evolve import *

import numpy as np
import matplotlib.pyplot as plt





def plot_cost_and_duration(time, cost, duration, basename="run"):
    fig, ax1 = plt.subplots()
    ax1.plot(time, cost, color="tab:blue", marker="o", label="Cost")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cost", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()  # secondary y-axis
    ax2.plot(time, duration, color="tab:orange", marker="x", label="Duration")
    ax2.set_ylabel("Duration (s)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(f"Cost and Duration over Time\n{basename}")
    fig.tight_layout()
    plt.grid(True)
    plt.show()


def plot_one_vs_the_other_over_time(time, the_one, the_one_name, the_other,  the_other_name, basename="run"):
    fig, ax1 = plt.subplots()
    ax1.plot(time, the_one, color="tab:blue", marker="o", label="Cost")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(str(the_one_name), color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()  # secondary y-axis
    ax2.plot(time, the_other, color="tab:orange", marker="x", label="Duration")
    ax2.set_ylabel(str(the_other_name), color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(f"{the_one_name} and {the_other_name} over Time\n{basename}")
    fig.tight_layout()
    plt.grid(True)
    plt.show()



def get_valid_time_range(data):
    cost = data["cost"]
    valid_len = max([5,count_initial_nonzero(cost)])-2
    # print(valid_len)

# --- Trim all 1D arrays to that length ---
    time = data["time"][:valid_len]
    cost = data["cost"][:valid_len]
    duration = data["duration"][:valid_len]
    return [time, cost, duration]


def show_cost_duration(dt, steps, tolerance, iterations, D, cut_off,timerange = None):
    params = f"dt={dt} D={D} tol={tolerance} iter={iterations} cutoff={cut_off}"
    data = prep_data(dt, steps, tolerance, iterations, D, cut_off,timerange)    
    time, cost, duration = get_valid_time_range(data)
    plot_cost_and_duration(time, cost, duration, basename=params)





def plot_entanglement_and_cost(data,L,params):
    time = data["time"]
    cost = data["cost"]
    state = data["state"]
    # duration = data["duration"]
    renyi2 = np.array(list(map(lambda x: tools.compute_second_Reyni(UniformMps(x), L), state)))
    logcost = np.array(list(map(lambda x: np.log(x), cost)))
    plot_one_vs_the_other_over_time(time, logcost, "logcost", renyi2, "renyi2", basename=params)


def plot_energy_and_cost(data,params):
    time = data["time"]
    cost = data["cost"]
    state = data["state"]
    # duration = data["duration"]

    model_H = TransverseFieldIsing(g=1.05, delta_t=dt, h=-0.5, J=-1)
    

    energy = np.array(list(map(lambda x: cq.compute_energy(UniformMps(x), model_H), state)))
    plot_one_vs_the_other_over_time(time, cost, "cost", energy, "energy", basename=params) 


def plot_energy_and_norm(data,params):
    time = data["time"]
    # cost = data["cost"]
    state = data["state"]
    # duration = data["duration"]
    dt=0.1
    model_H = TransverseFieldIsing(g=1.05, delta_t=dt, h=-0.5, J=-1)
    
    energy = np.array(list(map(lambda x: cq.compute_energy(UniformMps(x), model_H), state)))
    norm = np.array(list(map(lambda x: cq.compute_norm(UniformMps(x)), state)))
    # print(norm)
    # print(energy)

    plot_one_vs_the_other_over_time(time, norm, "norm", energy, "energy", basename=params)     

# def plot_energy_and_norm(data,params):
#     time = data["time"]
#     cost = data["cost"]
#     state = data["state"]
#     # duration = data["duration"]

   
#     energy = np.array(list(map(lambda x: cq.compute_normalization(UniformMps(x)), state)))
#     plot_one_vs_the_other_over_time(time, cost, "cost", energy, "energy", basename=params)   


def show_entanglement_cost(dt, steps, tolerance, iterations, D, cut_off, L, timerange = None):
    data = prep_data(dt, steps, tolerance, iterations, D, cut_off, timerange)
    params = f"dt={dt} D={D} tol={tolerance} iter={iterations} cutoff={cut_off}"
    plot_entanglement_and_cost(data,L,params)


def show_e_c_test(dt, steps, tolerance, iterations, D, cut_off, L, timerange = None):
    params = f"dt={dt} D={D} tol={tolerance} iter={iterations} cutoff={cut_off}"
    data=load_trimmed_data(dt, steps, tolerance, iterations, D, cut_off)
    plot_entanglement_and_cost(data,L,params)


def show_energy_cost(dt, steps, tolerance, iterations, D, cut_off, timerange = None):
    data = prep_data(dt, steps, tolerance, iterations, D, cut_off, timerange)
    params = f"dt={dt} D={D} tol={tolerance} iter={iterations} cutoff={cut_off}"
    plot_energy_and_cost(data,params)


def show_energy_norm(dt, steps, tolerance, iterations, D, cut_off, timerange = None):
    data = prep_data(dt, steps, tolerance, iterations, D, cut_off, timerange)
    params = f"dt={dt} D={D} tol={tolerance} iter={iterations} cutoff={cut_off}"
    plot_energy_and_norm(data,params)    

# def show_energy_norm(dt, steps, tolerance, iterations, D, cut_off, timerange = None):
#     data = prep_data(dt, steps, tolerance, iterations, D, cut_off, timerange)
#     params = f"dt={dt} D={D} tol={tolerance} iter={iterations} cutoff={cut_off}"
#     plot_energy_and_norm(data,params)



def plot_unfolded_energy_and_norm(dt, steps, tolerance, iterations, D, cut_off, params=None):
    """
    Load unfolded data and plot precomputed energy and norm.
    """

    data = load_unfolded_data(dt, steps, tolerance, iterations, D, cut_off)
    if data is None:
        return

    time = data["time"]
    energy = data["energy"]
    norm = data["norm"]

    # Use your existing plotting utility
    basename = params if params is not None else "unfolded_plot"

    plot_one_vs_the_other_over_time(time, norm, "norm", energy, "energy", basename=basename)



import matplotlib.pyplot as plt

def plot_unfolded_datasets(*datasets_with_labels, fields):
    """
    Plot selected fields (e.g. 'cost', 'renyi_entropy') for multiple unfolded datasets.
    Each argument is of the form: (dataset_dict, label_string).

    Example call:
        plot_unfolded_datasets(
            (ds1, "run 1"),
            (ds2, "run 2"),
            fields=["cost", "renyi_entropy"]
        )
    """

    # Number of subplots equals number of fields requested
    n_fields = len(fields)
    fig, axes = plt.subplots(n_fields, 1, figsize=(8, 4 * n_fields), sharex=False)

    # If only one field, axes is not a list
    if n_fields == 1:
        axes = [axes]

    for ax, field in zip(axes, fields):
        for data, label in datasets_with_labels:

            if data is None:
                print(f"Warning: dataset for label '{label}' is None, skipping.")
                continue

            time = data["time"]

            if field not in data:
                print(f"Warning: field '{field}' missing in dataset '{label}', skipping.")
                continue

            values = data[field]

            # Plot dataset
            ax.plot(time, values, label=label)

        ax.set_title(f"{field} over time")
        ax.set_xlabel("time")
        ax.set_ylabel(field)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()



def plot_unfolded_datasets_singleplot(*datasets_with_labels, fields):
    """
    Plot several fields (cost, renyi_entropy, etc.) from multiple datasets
    all on ONE SINGLE PLOT.

    Call like:
        plot_unfolded_datasets_singleplot(
            (ds1, "dt=0.005"),
            (ds2, "dt=0.01"),
            fields=["cost", "renyi_entropy"]
        )
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for data, label in datasets_with_labels:

        if data is None:
            print(f"Warning: dataset '{label}' is None, skipping.")
            continue

        time = data["time"]

        for field in fields:
            if field not in data:
                print(f"Warning: field '{field}' missing in dataset '{label}', skipping.")
                continue

            values = data[field]

            # Plot the curve with combined label
            ax.plot(time, values, label=f"{label} – {field}")

    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_title("Comparison of datasets and fields")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_unfolded_twofields(
    *datasets_with_labels,
    field_left=None,
    field_right=None,
    xlabel="Time",
    title=None,
    figsize=(10, 6),
    t_max=None,
):
    """
    Plot one or two fields from multiple unfolded datasets.
    - datasets_with_labels: (data_dict, label_string), ...
    - field_left:  name of field to plot on left y-axis (or None)
    - field_right: name of field to plot on right y-axis (or None)
    - t_max: if not None, only plot data with time <= t_max
    """

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = None
    if field_right is not None:
        ax_right = ax_left.twinx()

    # leave space on the right for the legend
    fig.subplots_adjust(right=0.75)

    # left axis: blue/teal/green, up to 6 distinct
    left_colors = [
        "#0072B2",  # strong blue
        "#009E73",  # bluish green
        "#56B4E9",  # sky blue
        "#00A2C2",  # teal-blue
        "#6ACE6A",  # medium green
        "#228833",  # dark green
    ]

    # right axis: yellow/orange/red, up to 6 distinct
    right_colors = [
        "#E69F00",  # orange
        "#F0E442",  # yellow
        "#F39C12",  # darker orange
        "#D55E00",  # orange-red
        "#CC6677",  # soft red
        "#B2182B",  # deep red
    ]

    left_handles = []
    right_handles = []

    # ----- LEFT FIELD -----
    if field_left is not None:
        for i, (data, label) in enumerate(datasets_with_labels):
            if data is None:
                print(f"Warning: dataset '{label}' is None, skipping (left).")
                continue
            if field_left not in data:
                print(f"Field '{field_left}' missing in dataset '{label}', skipping (left).")
                continue

            # time = np.asarray(data["time"]).flatten()
            # values = np.asarray(data[field_left]).flatten()
            #  buggy time array needs fixing
            values = np.asarray(data[field_left]).flatten()
            time = np.arange(len(values))


            if t_max is not None:
                mask = time <= t_max
                time = time[mask]
                values = values[mask]

            color = left_colors[i % len(left_colors)]
            h, = ax_left.plot(
                time,
                values,
                color=color,
                marker="o",
                linestyle="",
                markersize=3,
                linewidth=1.5,
                label=f"{label} – {field_left}",
            )
            left_handles.append(h)

    # ----- RIGHT FIELD -----
    if field_right is not None and ax_right is not None:
        for i, (data, label) in enumerate(datasets_with_labels):
            if data is None:
                print(f"Warning: dataset '{label}' is None, skipping (right).")
                continue
            if field_right not in data:
                print(f"Field '{field_right}' missing in dataset '{label}', skipping (right).")
                continue

            # time = np.asarray(data["time"]).flatten()
            # values = np.asarray(data[field_right]).flatten()
            #  fix for buggy time array
            values = np.asarray(data[field_right]).flatten()
            time = np.arange(len(values))


            if t_max is not None:
                mask = time <= t_max
                time = time[mask]
                values = values[mask]

            color = right_colors[i % len(right_colors)]
            h, = ax_right.plot(
                time,
                values,
                color=color,
                marker="x",
                linestyle="",
                markersize=3,
                linewidth=1.5,
                label=f"{label} – {field_right}",
            )
            right_handles.append(h)

    # ----- AXIS LABELS -----
    ax_left.set_xlabel(xlabel)

    if field_left is not None:
        ax_left.set_ylabel(field_left, color="tab:blue")
        ax_left.tick_params(axis="y", labelcolor="tab:blue")

    if field_right is not None and ax_right is not None:
        ax_right.set_ylabel(field_right, color="tab:orange")
        ax_right.tick_params(axis="y", labelcolor="tab:orange")

    # ----- TITLE -----
    if title is None:
        if field_left and field_right:
            base_title = f"{field_left} and {field_right} over Time"
        elif field_left:
            base_title = f"{field_left} over Time"
        elif field_right:
            base_title = f"{field_right} over Time"
        else:
            base_title = "Time Plot"

        if t_max is not None:
            title = f"{base_title} (t ≤ {t_max})"
        else:
            title = base_title

    fig.suptitle(title)

    # ----- LEGEND OUTSIDE (RIGHT) -----
    handles = left_handles + right_handles
    labels = [h.get_label() for h in handles]
    if handles:
        ax_left.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

    # ----- STYLE -----
    ax_left.grid(True, linewidth=0.5, linestyle="--", alpha=0.7)

    print("DEBUG:", label)
    # print("time shape:", time.shape)
    print("values shape:", values.shape)
    print("first 5 times:", time[:5])
    print("first 5 values:", values[:5])
    print("last 5 times:", time[-5:])
    print("last 5 values:", values[-5:])


    plt.show()


if __name__ == "__main__":

    steps = 1000000.0
    # steps=100
    cut_off= 10*60*60
    # cut_off=60

    delta_ts_ = [0.5*1e-2,1e-3,1e-4]
    tolerances_ = [1e-8,1e-10]
    iterations_ = [500,1000]
    Ds_ = [4,8,12,16]


    # prep_data(0.01,10000,1e-10,1000,4,50403)
    # show_e_c_test(0.01,10000,1e-10,1000,4,50403,4)
    # testdat2=load_trimmed_data(0.01,10000,1e-10,1000,4,50403)
    # unfold_data(testdat2,0.01,10000,1e-10,1000,4,50403,True)

    long_4=load_unfolded_data(0.01,10000,1e-10,1000,4,50403)

    # for D in [8,12,16]:
    #     print(D)
    #     generate_trimmed_and_unfolded(0.01,100000,1e-10,1000,D,50403,True)

    # sys.exit()    

    long_8=load_unfolded_data(0.01,100000,1e-10,1000,8,50403)
    long_12=load_unfolded_data(0.01,100000,1e-10,1000,12,50403)
    long_16=load_unfolded_data(0.01,100000,1e-10,1000,16,50403)

    # generate_trimmed_and_unfolded(0.005,1000,1e-10,3000,16,50404,True)
    fine_16_dt05_it3=load_unfolded_data(0.005,1000,1e-10,3000,16,50404)

    # generate_trimmed_and_unfolded(0.005,1000,1e-10,2000,16,50404,True)
    fine_16_dt05_it2=load_unfolded_data(0.005,1000,1e-10,2000,16,50404)

    # generate_trimmed_and_unfolded(0.001,1000,1e-11,1000,16,3604,True,timerange=0.1)
    fine_16_dt_001_tol=load_unfolded_data(0.001,1000,1e-11,1000,16,3604)


    # generate_trimmed_and_unfolded(0.005,200,1e-10,1000,4,50403,True)
    fine_16_dt_05_it1=load_unfolded_data(0.005,200,1e-10,1000,4,50403)



    # plot_unfolded_datasets(
    # (long_4, "D=4"),
    # (long_8, "D=8"),
    # fields=["cost", "renyi_entropy"]
    # )


    left="energy"
    right="cost"

    plot_unfolded_twofields(
    (long_4, "D=4"),
    (long_8, "D=8"),
    (long_12, "D=12"),
    (long_16, "D=16"),
    field_left=left,
    field_right=right,
    title=left+" and "+right
    )

    # plot_unfolded_twofields(
    # (long_16, "dt=0.01, iter=1000, tol=-10"),
    # (fine_16_dt_05_it1, "dt=0.005, iter=1000, tol=-10"),
    # (fine_16_dt05_it2, "dt=0.005, iter=2000, tol=-10"),
    # (fine_16_dt05_it3, "dt=0.005, iter=3000, tol=-10"),
    # (fine_16_dt_001_tol, "dt=0.001, iter=1000, tol=-11"),
    # field_left=left
    # )


    # plot_unfolded_twofields(
    # (fine_16_dt_001_tol, "dt=0.001, iter=1000, tol=-11"),
    # field_left=left,field_right=right
    # )

    left="cost"
    right="energy"

    plot_unfolded_twofields(
    (long_16, "dt=0.01, iter=1000, tol=-10"),
    (fine_16_dt05_it2, "dt=0.005, iter=2000, tol=-10"),
    (fine_16_dt05_it3, "dt=0.005, iter=3000, tol=-10"),
    (fine_16_dt_001_tol, "dt=0.001, iter=1000, tol=-11"),
    field_left=left,
    field_right=right,
    # title=left+" and "+right
    )


    sys.exit()
    for tolerance in tolerances_[1:]:
        for iterations in [1000]:
            for dt in delta_ts_[0:1]:
                for bondD in [16]:
                    print([tolerance,iterations,dt,bondD])
                
                    cutoff=50404
                    # show_entanglement_cost(dt, 10000, tolerance, iterations, 4, cutoff,4)
                    steps=1000
                    iterations=3000
                    # bondD=4
                    show_energy_cost(dt, steps, tolerance, iterations, bondD, cutoff)
                    show_cost_duration(dt, steps, tolerance, iterations, bondD, cutoff)
                    show_entanglement_cost(dt, steps, tolerance, iterations, bondD, cutoff,4)

                    

             