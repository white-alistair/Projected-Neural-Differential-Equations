# Run all evaluation scripts and generate figures in evaluation/figures/
#
# Usage:
#   LD_LIBRARY_PATH="/path/to/magma-install/lib:$LD_LIBRARY_PATH" julia --project=. evaluation/run_all.jl

mkpath(joinpath(@__DIR__, "figures"))
mkpath(joinpath(@__DIR__, "data"))

println("\n" * "="^80)
println("1/7  Quartic Potential 2D")
println("="^80)
include(joinpath(@__DIR__, "quartic_potential_2d.jl"))
main(save_data=true, save_figures=true)

println("\n" * "="^80)
println("2/7  Lotka-Volterra")
println("="^80)
include(joinpath(@__DIR__, "lotka_volterra.jl"))
main(save_data=true)

println("\n" * "="^80)
println("3/7  N-Pendulum")
println("="^80)
include(joinpath(@__DIR__, "n_pendulum.jl"))
main()

println("\n" * "="^80)
println("4/7  IEEE 14-Bus")
println("="^80)
include(joinpath(@__DIR__, "ieee14bus.jl"))
main()

println("\n" * "="^80)
println("5/7  Phase plots (QP, LV)")
println("="^80)
include(joinpath(@__DIR__, "quartic_potential_phase.jl"))
main()
include(joinpath(@__DIR__, "lotka_volterra_phase.jl"))
main()

println("\n" * "="^80)
println("6/7  N-Pendulum phase plots")
println("="^80)
include(joinpath(@__DIR__, "n_pendulum_phase.jl"))
main()

println("\n" * "="^80)
println("7/7  IEEE 14-Bus long rollout")
println("="^80)
include(joinpath(@__DIR__, "ieee14bus_long_rollout_figures.jl"))
main()

println("\n" * "="^80)
println("Done! Figures saved to evaluation/figures/")
println("="^80)
