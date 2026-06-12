using Test
using ProjectedNDEs

@testset "ProjectedNDEs Tests" begin
    include("solvers.jl")
    include("n_pendulum.jl")
    include("power_grids.jl")
end