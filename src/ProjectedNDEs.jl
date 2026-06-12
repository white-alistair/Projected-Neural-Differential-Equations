module ProjectedNDEs

using Optimisers,
    OrdinaryDiffEq,
    SciMLBase,
    SciMLSensitivity,
    Lux,
    LuxCUDA,
    ComponentArrays,
    Zygote,
    ForwardDiff,
    Parameters,
    ParameterSchedulers,
    ArgParse,
    LinearAlgebra,
    StatsBase,
    Statistics,
    Random,
    DelimitedFiles,
    TOML,
    JLD2,
    Printf,
    CUDA,
    ChainRulesCore,
    MLUtils,
    NNlib,
    FastClosures,
    Adapt,
    TensorBoardLogger,
    Colors,
    Dates

abstract type AbstractDynamicalSystem{T} end

include("experiments/n_pendulum.jl")
include("experiments/power_grids.jl")
include("experiments/quartic_potential_2d.jl")
include("experiments/lotka_volterra.jl")

include("neural_nets.jl")
include("models/models.jl")
include("models/neural_odes.jl")
include("models/constrained_neural_odes.jl")
include("models/stabilized_neural_odes.jl")

include("command_line.jl")
include("gpu.jl")
include("magma.jl")
include("cusolver.jl")
include("solvers.jl")
include("projection.jl")
include("time_series.jl")
include("normalization.jl")
include("data.jl")
include("multiple_shooting.jl")
include("evaluate.jl")
include("losses.jl")
include("scheduler.jl")
include("optimiser.jl")
include("adjoints.jl")
include("train.jl")
include("checkpoints.jl")
include("io.jl")
include("relative_error.jl")

function __init__()
    if isfile(MAGMA_LIB)
        try
            magma_init()
        catch e
            @warn "Failed to initialize MAGMA: $e"
        end
    end
end

end
