using Pkg
Pkg.activate("power_grids")

using PowerDynamics
using AmbientForcing
using OrdinaryDiffEq
using Random
using ProgressBars
using JLD2

include("../random_force.jl")

N = 1000
t0 = 0.0
t1 = 10.0
dt = 0.1
tspan = (t0, t1)
saveat = t0:dt:t1
abstol = reltol = 1e-9
a = 0.15
b = 1.0
# index_to_perturb = 1
τ = 0.2

grid = read_powergrid("power_grids/ieee14bus/ieee14bus.json", Json)
op = find_operationpoint(grid)
rhs_grid = rhs(grid)

rng = Random.default_rng()
Random.seed!(rng, 1)

trajectories = []
for i = ProgressBar(1:N)
    # Perturb using ambient forcing
    index_to_perturb = Random.rand(setdiff(1:14, 2))
    Frand = random_force_uniform_circle(a, b, 14) #, index_to_perturb)
    afoprob = ambient_forcing_problem(rhs_grid, op.vec, τ, Frand, method = :ForwardDiff)
    u0 = ambient_forcing(afoprob, op.vec, τ, Frand)  # Our new valid initial condition

    prob = ODEProblem(rhs_grid, u0, tspan, nothing)
    sol = solve(prob, Rodas4(); saveat, reltol, abstol)
    traj = stack(sol.u)
    traj = vcat(traj[1:2:end-1, :], traj[2:2:end, :])  # We want [all_real_parts, all_imag_parts]
    push!(trajectories, traj)
end

data = stack(trajectories)
filename = replace("data_tau_$(τ)_T_$(Int(t1))_dt_$(dt)_all_nodes", '.'=>"")
JLD2.save_object("power_grids/ieee14bus/$(filename).jld2", data)
