from tenpy.models import spin_chain
from tenpy.networks.mps import MPS
from tenpy.algorithms.vumps import run_VUMPS

model_params = {
    'S': 0.5,
    'Jz': 1.0,           # Ising interaction
    'hx': 3.0,           # Pre-quench transverse field
    'bc_MPS': 'infinite',  # Use infinite MPS
    'conserve': None       # Don't conserve quantum numbers
}

# Step 2: Build the spin chain model
model = SpinChain(model_params)

# Step 3: Initialize an MPS (product state: all spins up or down)
product_state = ['up', 'up']
psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc='infinite')

# Step 4: Run VUMPS to variationally optimize the ground state
E, psi = run_VUMPS(psi, model)