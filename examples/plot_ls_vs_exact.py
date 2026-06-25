# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # from qdmt.analysis.tools import compute_loschmidt
# # # from src.qdmt.uniform_mps import UniformMps
# # # from scipy.integrate import quad


# # # # import your A_from_snug_params from wherever this messy file lives
# # # from examples.find_parametrization import A_from_snug_params


# # # def load_param_reconstruction(filename):
# # #     data = np.load(filename, allow_pickle=True)
# # #     params_free = data["params_free"]
# # #     A_param = np.array([A_from_snug_params(p) for p in params_free])
# # #     return A_param, data


# # # def load_A_full_reference():
# # #     dat = np.load(
# # #         "/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM_debug/"
# # #         "benchmark_dt=0.2_steps=9_tol=1e-14_it=10000_D=2_cut=3600(_Trotter_order=-1).npz"
# # #     )

# # #     As = dat["state"]
# # #     times = dat["time"]

# # #     A0_tens = np.load("data/ground_state/tfim_AL_D2_g1.5.npz")["A"]

# # #     As_full = np.concatenate(([A0_tens], As))
# # #     times_full = np.concatenate(([0.0], times))

# # #     return times_full, As_full

# # # def load_mps_reference():
# # #     dat = np.load(
# # #         "/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM_debug/"
# # #         "benchmark_dt=0.2_steps=9_tol=1e-14_it=10000_D=2_cut=3600(_Trotter_order=-1).npz"
# # #     )

# # #     As = dat["state"]
# # #     times = dat["time"]

# # #     A0_tens = np.load("data/ground_state/tfim_AL_D2_g1.5.npz")["A"]

# # #     As_full = np.concatenate(([A0_tens], As))
# # #     times_full = np.concatenate(([0.0], times))

# # #     return times_full, As_full


# # # def f_complex(z, g0, g1):
# # #     def theta(k, g):
# # #         return np.arctan2(np.sin(k), g - np.cos(k)) / 2

# # #     def phi(k, g0, g1):
# # #         return theta(k, g0) - theta(k, g1)

# # #     def epsilon(k, g1):
# # #         return -2 * np.sqrt((g1 - np.cos(k))**2 + np.sin(k)**2)

# # #     def integrand(k):
# # #         ph = phi(k, g0, g1)
# # #         val = (
# # #             np.cos(ph)**2
# # #             + np.sin(ph)**2 * np.exp(-2 * z * epsilon(k, g1))
# # #         )
# # #         return -1 / (2 * np.pi) * np.log(val)

# # #     real = quad(lambda k: np.real(integrand(k)), 0, np.pi, limit=200)[0]
# # #     imag = quad(lambda k: np.imag(integrand(k)), 0, np.pi, limit=200)[0]
# # #     return real + 1j * imag


# # # def loschmidt_paper(t, g0=1.5, g1=0.2):
# # #     val = f_complex(1j * t, g0, g1) + f_complex(-1j * t, g0, g1)
# # #     return float(np.real_if_close(val))


# # # def main():
# # #     # ------------------------------------------------------------
# # #     # Load six-parameter snug fit
# # #     # ------------------------------------------------------------
# # #     param_file = "/Users/phys2259/Documents/qdmt/super_snug_results_warm_huge_sym.npz"
# # #     data = np.load(param_file, allow_pickle=True)

# # #     params_free = data["params_free"]
# # #     dists = data["dists"] if "dists" in data else None

# # #     # ------------------------------------------------------------
# # #     # Load full stored MPS / A_full reference
# # #     # ------------------------------------------------------------
# # #     t_A_full, A_full = load_A_full_reference()
# # #     ls_A_full = compute_loschmidt(A_full)    

# # #     print("Loaded:", param_file)
# # #     print("params_free shape:", params_free.shape)
# # #     if dists is not None:
# # #         print(
# # #             "fit dists min/median/max:",
# # #             float(np.min(dists)),
# # #             float(np.median(dists)),
# # #             float(np.max(dists)),
# # #         )

# # #     # ------------------------------------------------------------
# # #     # Reconstruct tensors from the 6 free parameters
# # #     # ------------------------------------------------------------
# # #     A_snug = np.array([A_from_snug_params(p) for p in params_free])

# # #     # Loschmidt from the reconstructed tensors
# # #     ls_snug = compute_loschmidt(A_snug)

# # #     # Match the snug time grid.
# # #     # Your saved huge symmetric run seems to be dt=0.2.
# # #     dt_snug = 0.2
# # #     t_snug = dt_snug * np.arange(len(ls_snug))

# # #     # ------------------------------------------------------------
# # #     # Analytic exact LS on same grid
# # #     # ------------------------------------------------------------
# # #     g0 = 1.5
# # #     g1 = 0.2

# # #     ls_exact_on_snug_grid = np.array([
# # #         loschmidt_paper(t, g0=g0, g1=g1)
# # #         for t in t_snug
# # #     ])

# # #     # Optional smooth exact curve
# # #     t_fine = np.linspace(0, t_snug[-1], 401)
# # #     ls_exact_fine = np.array([
# # #         loschmidt_paper(t, g0=g0, g1=g1)
# # #         for t in t_fine
# # #     ])

# # #     # ------------------------------------------------------------
# # #     # Plot main comparison
# # #     # ------------------------------------------------------------
# # #     plt.figure(figsize=(7, 4))
# # #     plt.plot(t_fine, ls_exact_fine, "-", label="analytic exact")
# # #     plt.plot(t_snug, ls_snug, "o--", label="6-param snug")

# # #     plt.figure(figsize=(7, 4))
# # #     plt.plot(t_fine, ls_exact_fine, "-", label="analytic exact")
# # #     plt.plot(t_A_full, ls_A_full, "s:", label="A_full / stored MPS")
# # #     plt.plot(t_snug, ls_snug, "o--", label="6-param snug")

# # #     plt.xlabel("time")
# # #     plt.ylabel("Loschmidt")
# # #     plt.title("Loschmidt: exact vs A_full vs 6-param snug")
# # #     plt.grid(True)
# # #     plt.legend()
# # #     plt.tight_layout()
# # #     plt.show()

# # #     plt.xlabel("time")
# # #     plt.ylabel("Loschmidt")
# # #     plt.title("Loschmidt: analytic exact vs 6-param snug")
# # #     plt.grid(True)
# # #     plt.legend()
# # #     plt.tight_layout()
# # #     plt.show()

# # #     # ------------------------------------------------------------
# # #     # Plot error
# # #     # ------------------------------------------------------------
# # #     err = np.abs(ls_snug - ls_exact_on_snug_grid)

# # #     plt.figure(figsize=(7, 4))
# # #     plt.plot(t_snug, err, "o-")
# # #     plt.xlabel("time")
# # #     plt.ylabel("|snug - exact|")
# # #     plt.yscale("log")
# # #     plt.title("Loschmidt error against analytic exact")
# # #     plt.grid(True, which="both")
# # #     plt.tight_layout()
# # #     plt.show()

# # #     print("t:", t_snug)
# # #     print("ls_snug:", ls_snug)
# # #     print("ls_exact:", ls_exact_on_snug_grid)
# # #     print("abs error:", err)



# # # if __name__ == "__main__":
# # #     main()


# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.integrate import quad

# # from qdmt.analysis.tools import compute_loschmidt


# # def f_complex(z, g0, g1):
# #     def theta(k, g):
# #         return np.arctan2(np.sin(k), g - np.cos(k)) / 2

# #     def phi(k, g0, g1):
# #         return theta(k, g0) - theta(k, g1)

# #     def epsilon(k, g1):
# #         return -2 * np.sqrt((g1 - np.cos(k))**2 + np.sin(k)**2)

# #     def integrand(k):
# #         ph = phi(k, g0, g1)
# #         val = (
# #             np.cos(ph)**2
# #             + np.sin(ph)**2 * np.exp(-2 * z * epsilon(k, g1))
# #         )
# #         return -1 / (2 * np.pi) * np.log(val)

# #     real = quad(lambda k: np.real(integrand(k)), 0, np.pi, limit=200)[0]
# #     imag = quad(lambda k: np.imag(integrand(k)), 0, np.pi, limit=200)[0]
# #     return real + 1j * imag


# # def loschmidt_paper(t, g0=1.5, g1=0.2):
# #     val = f_complex(1j * t, g0, g1) + f_complex(-1j * t, g0, g1)
# #     return float(np.real_if_close(val))


# # def load_A_full_reference():
# #     dat = np.load(
# #         "/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM_debug/"
# #         "benchmark_dt=0.2_steps=9_tol=1e-14_it=10000_D=2_cut=3600(_Trotter_order=-1).npz"
# #     )

# #     As = dat["state"]
# #     times = dat["time"]

# #     A0_tens = np.load("data/ground_state/tfim_AL_D2_g1.5.npz")["A"]

# #     A_full = np.concatenate(([A0_tens], As))
# #     t_full = np.concatenate(([0.0], times))

# #     return t_full, A_full


# # def main():
# #     g0 = 1.5
# #     g1 = 0.2
# #     dt_trotter = 0.2

# #     # ------------------------------------------------------------
# #     # D=2 classical / stored tensor trajectory
# #     # ------------------------------------------------------------
# #     t_full, A_full = load_A_full_reference()
# #     ls_A_full = compute_loschmidt(A_full)

# #     # Express x-axis in units of the Trotter step dt = 0.2
# #     n_full = t_full / dt_trotter

# #     # ------------------------------------------------------------
# #     # Analytic exact curve
# #     # ------------------------------------------------------------
# #     t_exact = np.linspace(0.0, t_full[-1], 500)
# #     n_exact = t_exact / dt_trotter

# #     ls_exact = np.array([
# #         loschmidt_paper(t, g0=g0, g1=g1)
# #         for t in t_exact
# #     ])

# #     # ------------------------------------------------------------
# #     # Plot
# #     # ------------------------------------------------------------
# #     plt.figure(figsize=(6.5, 4.0))

# #     plt.plot(
# #         n_exact,
# #         ls_exact,
# #         "-",
# #         linewidth=2.5,
# #         label="analytic exact",
# #     )

# #     plt.plot(
# #         n_full,
# #         ls_A_full,
# #         "o",
# #         markersize=5,
# #         label="D=2 classical",
# #     )

# #     plt.xlabel(r"time / $\Delta t$")
# #     plt.ylabel("Loschmidt rate")
# #     plt.title("Loschmidt echo")

# #     plt.grid(True, alpha=0.3)
# #     plt.legend(frameon=False)
# #     plt.tight_layout()

# #     plt.savefig("loschmidt_echo_D2_classical_vs_exact.pdf", bbox_inches="tight")
# #     plt.savefig("loschmidt_echo_D2_classical_vs_exact.png", dpi=300, bbox_inches="tight")

# #     plt.show()


# # if __name__ == "__main__":
# #     main()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import quad

# from qdmt.analysis.tools import compute_loschmidt


# def f_complex(z, g0, g1):
#     def theta(k, g):
#         return np.arctan2(np.sin(k), g - np.cos(k)) / 2

#     def phi(k, g0, g1):
#         return theta(k, g0) - theta(k, g1)

#     def epsilon(k, g1):
#         return -2 * np.sqrt((g1 - np.cos(k))**2 + np.sin(k)**2)

#     def integrand(k):
#         ph = phi(k, g0, g1)
#         val = (
#             np.cos(ph) ** 2
#             + np.sin(ph) ** 2 * np.exp(-2 * z * epsilon(k, g1))
#         )
#         return -1 / (2 * np.pi) * np.log(val)

#     real = quad(lambda k: np.real(integrand(k)), 0, np.pi, limit=200)[0]
#     imag = quad(lambda k: np.imag(integrand(k)), 0, np.pi, limit=200)[0]
#     return real + 1j * imag


# def loschmidt_paper(t, g0=1.5, g1=0.2):
#     val = f_complex(1j * t, g0, g1) + f_complex(-1j * t, g0, g1)
#     return float(np.real_if_close(val))


# def load_A_full_reference():
#     dat = np.load(
#         "/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM_debug/"
#         "benchmark_dt=0.2_steps=9_tol=1e-14_it=10000_D=2_cut=3600(_Trotter_order=-1).npz"
#     )

#     As = dat["state"]
#     times = dat["time"]

#     A0_tens = np.load("data/ground_state/tfim_AL_D2_g1.5.npz")["A"]

#     A_full = np.concatenate(([A0_tens], As))
#     t_full = np.concatenate(([0.0], times))

#     return t_full, A_full


# def main():
#     g0 = 1.5
#     g1 = 0.2
#     dt_trotter = 0.2

#     # ------------------------------------------------------------
#     # D=2 classical / stored tensor trajectory
#     # ------------------------------------------------------------
#     t_full, A_full = load_A_full_reference()
#     ls_A_full = compute_loschmidt(A_full)

#     # ------------------------------------------------------------
#     # Analytic exact curve
#     # ------------------------------------------------------------
#     t_exact = np.linspace(0.0, t_full[-1], 500)

#     ls_exact = np.array([
#         loschmidt_paper(t, g0=g0, g1=g1)
#         for t in t_exact
#     ])

#     # ------------------------------------------------------------
#     # Plot
#     # ------------------------------------------------------------
#     plt.figure(figsize=(6.5, 4.0))

#     plt.plot(
#         t_exact,
#         ls_exact,
#         "-",
#         color="black",
#         linewidth=2.5,
#         label="analytic exact",
#     )

#     plt.plot(
#         t_full,
#         ls_A_full,
#         "o-",
#         color="black",
#         linewidth=2.0,
#         linestyle="--",

#         markersize=4,
#         label="D=2 classical",
#     )

#     plt.xlabel("time")
#     plt.ylabel("Loschmidt rate")
#     plt.title("Loschmidt echo")

#     # ticks at 0.0, 0.2, 0.4, ...
#     xticks = np.arange(0.0, t_full[-1] + 0.5 * dt_trotter, dt_trotter)
#     plt.xticks(xticks)

#     plt.grid(True, alpha=0.3)
#     plt.legend(frameon=False)
#     plt.tight_layout()

#     plt.savefig("loschmidt_echo_D2_classical_vs_exact.pdf", bbox_inches="tight")
#     plt.savefig("loschmidt_echo_D2_classical_vs_exact.png", dpi=300, bbox_inches="tight")

#     plt.show()


# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from qdmt.analysis.tools import compute_loschmidt


# ============================================================
# Analytic exact Loschmidt rate
# ============================================================

def f_complex(z, g0, g1):
    def theta(k, g):
        return np.arctan2(np.sin(k), g - np.cos(k)) / 2

    def phi(k, g0, g1):
        return theta(k, g0) - theta(k, g1)

    def epsilon(k, g1):
        return -2 * np.sqrt((g1 - np.cos(k)) ** 2 + np.sin(k) ** 2)

    def integrand(k):
        ph = phi(k, g0, g1)
        val = (
            np.cos(ph) ** 2
            + np.sin(ph) ** 2 * np.exp(-2 * z * epsilon(k, g1))
        )
        return -1 / (2 * np.pi) * np.log(val)

    real = quad(lambda k: np.real(integrand(k)), 0, np.pi, limit=200)[0]
    imag = quad(lambda k: np.imag(integrand(k)), 0, np.pi, limit=200)[0]
    return real + 1j * imag


def loschmidt_paper(t, g0=1.5, g1=0.2):
    val = f_complex(1j * t, g0, g1) + f_complex(-1j * t, g0, g1)
    return float(np.real_if_close(val))


# ============================================================
# Load D=2 classical tensor trajectory
# ============================================================

def load_A_full_reference():
    dat = np.load(
        "/Users/phys2259/Documents/qdmt/results/integrable_D2_forQM_debug/"
        "benchmark_dt=0.05_steps=39_tol=1e-14_it=10000_D=2_cut=3600(_Trotter_order=-1).npz",
        allow_pickle=True,
    )

    As = dat["state"]
    times = dat["time"]

    A0_tens = np.load("data/ground_state/tfim_AL_D2_g1.5.npz")["A"]

    A_full = np.concatenate(([A0_tens], As))
    t_full = np.concatenate(([0.0], times))

    return t_full, A_full


# ============================================================
# Main plot
# ============================================================

def main():
    g0 = 1.5
    g1 = 0.2
    dt_trotter = 0.1

    # D=2 classical / stored tensor trajectory
    t_full, A_full = load_A_full_reference()
    ls_A_full = compute_loschmidt(A_full)

    print("Loaded D=2 classical data")
    print("t range:", t_full[0], "to", t_full[-1])
    print("number of points:", len(t_full))
    print("A_full shape:", A_full.shape)

    # Analytic exact curve
    t_exact = np.linspace(0.0, t_full[-1], 800)
    ls_exact = np.array([
        loschmidt_paper(t, g0=g0, g1=g1)
        for t in t_exact
    ])

    # Plot
    plt.figure(figsize=(6.5, 4.0))

    plt.plot(
        t_exact,
        ls_exact,
        color="black",
        linestyle="-",
        linewidth=2.5,
        label="analytic exact",
    )

    plt.plot(
        t_full,
        ls_A_full,
        color="black",
        linestyle="--",
        marker="o",
        markersize=4,
        linewidth=2.0,
        label="D=2 classical",
    )

    plt.xlabel("time")
    plt.ylabel("Loschmidt rate")
    plt.title("Loschmidt echo")

    # ticks at 0.0, 0.2, 0.4, ...
    tick_spacing = 0.2
    xticks = np.arange(0.0, t_full[-1] + 0.5 * tick_spacing, tick_spacing)
    plt.xticks(xticks)

    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig("loschmidt_echo_D2_classical_dt01_vs_exact.pdf", bbox_inches="tight")
    plt.savefig("loschmidt_echo_D2_classical_dt01_vs_exact.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()