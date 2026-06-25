# run_steps.py
import numpy as np

from qDMT.cost_functions import cost_func_clean
from qDMT.ansatz import QasmStateAnsatzXZGauge
from qDMT.evolution import timeEvolve
from qDMT.extrapolate import linFitGuess

from env_quantinuum import setup_qnexus, inject_into_cost_functions

def cost_func_avg_unfold_noiseless(θt, θtdt, g, dt, shots=1000,
                                  Ansatz=QasmStateAnsatzXZGauge,
                                  backend="emulator-nexus", savefile=None):
    return cost_func_clean(
        θt, θtdt, g, dt,
        shots=shots,
        Ansatz=Ansatz,
        backend=backend,
        savefile=savefile,
        trotter_type="avg",
        fold_type="unfold",
        noiselss_result_only=True,   # keeping your current spelling
        compare_noiseless=True       # required for your current logic
    )

def step_3site_avg_fold(i, dt, startparams):
    file = (
        f"quantinuum_test_results/a005c01/"
        f"opt_run{i}_1000_gauge_a_005_c_01_6_updates_nexus_3site_dt{dt}_avgfold"
    )

    res = timeEvolve(
        initParams=startparams,
        g=0.2,
        dt=dt,
        nSteps=1,
        shots=1000,
        Ansatz=QasmStateAnsatzXZGauge,
        costFunc=cost_func_avg_unfold_noiseless,  # noiseless debug
        backend="emulator-nexus",
        saveFile=file,
        nextGuess=linFitGuess,
        minimiser="SPSA",
        maxiter=None,
        tol=None,
        a=0.05,
        c=0.1,
        nIters=6,
        plot=True
    )

    params = np.load(file + "_params.npy", allow_pickle=True)
    return res, params

def main():
    # --- backend/session setup ---
    qnx, project, qn_config, compile_backend = setup_qnexus(
        project_name="QNexus Emulator Minimal",
        device_name="H2-Emulator",  # change here
        compile_device="H2-2",
        do_login=False
    )
    inject_into_cost_functions(qnx, project, qn_config, compile_backend)

    # --- load initial params ---
    init_gauge_params = np.array([
        [ 1.57079982, -1.57079737,  0.03284934, -0.30714525,  0.48471933,
          0.3381791 ,  1.1889755 ,  0.65126142, 0, 0],
        [ 2.44453459, -1.17595111,  0.02506549, -0.47125596,  0.50419693,
         -0.08200032,  1.18412256,  0.59756283, 0, 0],
        [ 2.59219334, -1.16739089,  0.01111036, -0.83728779,  0.41108322,
         -0.05090685,  1.28362378,  0.697738  , 0, 0]
    ])

    params_step = init_gauge_params

    for i in range(1, 9):
        print(f"\n=== avg fold | dt=0.2 | step {i} ===")
        res, params_step = step_3site_avg_fold(i, dt=0.2, startparams=params_step)
        np.save("latest_params_dt02_avgfold.npy", params_step, allow_pickle=True)

if __name__ == "__main__":
    main()
