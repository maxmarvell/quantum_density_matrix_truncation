# env_quantinuum.py
import time
import qnexus as qnx
from pytket.extensions.quantinuum import QuantinuumBackend

def wait_completed(job_ref, timeout_s=7200, poll_s=5):
    t0 = time.time()
    while True:
        df = job_ref.df()
        status = str(df.iloc[0]["last_status"])
        if "COMPLETED" in status:
            return
        if ("ERROR" in status) or ("CANCELLED" in status):
            raise RuntimeError(f"Job ended with status {status}: {df.iloc[0].to_dict()}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting; last_status={status}")
        time.sleep(poll_s)

def setup_qnexus(
    *,
    project_name="QNexus Emulator Minimal",
    device_name="H2-Emulator",     # or "H2-2E"
    compile_device="H2-2",         # compile-only backend
    do_login=False
):
    # qnx.login_with_credentials() if you want explicit login
    if do_login:
        qnx.login_with_credentials()

    project = qnx.projects.get_or_create(name=project_name)
    qnx.context.set_active_project(project)

    qn_config = qnx.QuantinuumConfig(device_name=device_name)
    compile_backend = QuantinuumBackend(compile_device)

    return qnx, project, qn_config, compile_backend

def inject_into_cost_functions(qnx, project, qn_config, compile_backend, wait_fn=wait_completed):
    # Keep your current pattern, but do it once, centrally.
    import importlib
    import qDMT.cost_functions as cf
    importlib.reload(cf)

    cf.qnx = qnx
    cf.project = project
    cf.qn_config = qn_config
    cf.compile_backend = compile_backend
    cf.wait_completed = wait_fn

    return cf  # handy for debugging
