using MPSKit, MPSKitModels, TensorKit
using ProgressMeter, Plots # for demonstration purposes

# D = 4 # bonddimension
# init_state = InfiniteMPS(ℂ^2, ℂ^D)

# g_values = 0.1:0.1:2

# M = @showprogress map(g_values) do g
#     H = transverse_field_ising(; g=g)
#     groundstate, environment, δ = find_groundstate(init_state, H, VUMPS(; verbosity=0))
#     return abs(expectation_value(groundstate, 1 => σᶻ()))
# end

# scatter(g_values, M, xlabel="g", ylabel="M", label="D=$D", title="Magnetization")


# using MPSKit
# using MPSKitModels
# using TensorKit
using NPZ

# Parameters
D = 4
g = 1.5            # transverse field = 1.5 (paramagnet point)
d = 2              # physical dimension

println("Building transverse-field Ising model...")
H = transverse_field_ising(; g=g)

println("Initializing Infinite MPS with D = $D")
mps0 = InfiniteMPS(ℂ^d, ℂ^D)

println("Running VUMPS...")
groundstate, environment, δ = find_groundstate(mps0, H, VUMPS(; tol=1.0e-15, maxiter=10000, verbosity=1))
println("Converged with δ = $δ")


# ---- Extract left-canonical unit-cell tensor ----
# groundstate.AL is a vector (unit cell), you want the first (and only) site

AL1 = groundstate.AL[1]        # TensorMap: (ℂ^D ⊗ ℂ^d) ← ℂ^D

# print(AL1.data)

A = reshape(AL1.data, D, d, D)
println(size(A))   # must be (6, 2, 6)

outpath = "/Users/phys2259/Documents/qdmt/data/ground_state/tfim_AL_D$D" * "_g1.5.npz"
npzwrite(outpath, Dict("A" => A))
println("Saved to: ", outpath)

print(abs(expectation_value(groundstate, 1 => σᶻ())))

# npzwrite("tfim_AL_D6_g1.0.npz", Dict("A" => A))


# # Convert to plain array: this will be a (D*d, D) matrix
# V = Array(AL1)
# println("size(V) = ", size(V))  # should be (D * 2, D) e.g. (12, 6)

# # ---- Save for Python ----
# npzwrite("tfim_V_D$(D)_g$(g).npz", Dict("V" => V))
# println("Saved V to tfim_V_D$(D)_g$(g).npz")
