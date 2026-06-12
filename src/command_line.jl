function parse_command_line(; log = false)
    settings = ArgParseSettings(autofix_names = true)

    @add_arg_table settings begin
        #! format: off
        "--experiment"
            help = "The name of the experiment to run"
            arg_type = Symbol
            required = true
        "--experiment-version", "--version"
            help = "If multiple versions of an experiment exist, specify which version to run"
            arg_type = Int
            default = 1
        "--experiment-name", "--exp-name"
            arg_type = String
        "--device"
            help = "CPU or CUDA device"
            arg_type = Symbol
        "--job-id"
            help = "Job ID used for serialization of experiment results"
            arg_type = String
            default = haskey(ENV, "SLURM_ARRAY_JOB_ID") ? ENV["SLURM_ARRAY_JOB_ID"] * "_" * ENV["SLURM_ARRAY_TASK_ID"] : ENV["SLURM_JOB_ID"]
        "--checkpoint"
            help = "ID of run from which to start training"
            arg_type = Int
        "--rng-seed", "--seed"
            help = "RNG seed"
            arg_type = Int
            default = 1
        "--NF", "--precision"   
            help = "The number format to use"
            arg_type = DataType
            default = Float64

        # Data generation args
        "--dataset-name"
            help = "Name of data file for power grid experiments"
            arg_type = String
        "--t0"
            help = "Start time of each training trajectory"
            arg_type = Float64
            default = 0.0
        "--t1"
            help = "End time of each training trajectory"
            arg_type = Float64
        "--dt"
            help = "Timestep for training trajectories"
            arg_type = Float64
        "--data-solver"
            help = "Name of solver from OrdinaryDiffEq.jl for generating training data"
            arg_type = SciMLBase.AbstractDEAlgorithm
            default = Vern9()
        "--data-reltol"
            help = "Solver relative tolerance for generating training data"
            arg_type = Float64
            default = 1e-12
        "--data-abstol"
            help = "Solver absolute tolerance for generating training data"
            arg_type = Float64
            default = 1e-12
        "--normalize-data"
            help = "Whether to normalize the data"
            action = :store_true
        "--filter-threshold"
            help = "Filter out chunks where κ(GGᵀ) exceeds this threshold (for PNDE stability)"
            arg_type = Float64

        # Data split args
        "--n-train"
            help = "Number of training trajectories"
            arg_type = Int
        "--n-valid"
            help = "Number of validation trajectories"
            arg_type = Int
            default = 0
        "--n-test"
            help = "Number of test trajectories"
            arg_type = Int
            default = 0
        "--chunk-size"
            help = "The number of timesteps per chunk of a trajectory"
            arg_type = Int
            required = true
        "--batch-size"
            help = "The number of chunks per minibatch"
            arg_type = Int
            required = true

        # N-pendulum args
        "--N"
            help = "Number of pendulum bobs"
            arg_type = Int
        "--friction"
            help = "Damping coefficient"
            arg_type = Float64
            default = 0.0

        # Neural net args
        "--hidden-layers", "--layers"
            help = "Number of hidden layers"
            arg_type = Int
        "--hidden-width", "--width"
            help = "Width of hidden layers"
            arg_type = Int
        "--activation"
            help = "Activation function"
            arg_type = Function
            default = relu
        "--use-skip"
            help = "Whether to use skip connections between hidden layers"
            action = :store_true

        # Stabilization
        "--stabilization-param"
            arg_type = Float64
            default = 0.0

        # Two-stage training
        "--switch-epoch"
            help = "Epoch at which to switch to switch-version (for two-stage training)"
            arg_type = Int
            default = nothing
        "--switch-version"
            help = "Experiment version to switch to at switch-epoch"
            arg_type = Int
            default = nothing
        "--switch-lr"
            help = "Learning rate to use after switching (optional, continues schedule if not set)"
            arg_type = Float64
            default = nothing
        "--switch-optimiser"
            help = "Re-initialize optimiser when switching models"
            action = :store_true

        # Training args
        "--loss"
            help = "Loss Function"
            arg_type = Symbol
            default = :MSE
        "--epochs"
            help = "Total number of epochs"
            arg_type = Int
            required = true
        "--schedule-file", "--schedule"
            help = "Path to learning rate schedule config file"
            arg_type = String
            required = true
        "--optimiser-rule", "--opt"
            help = "Choice of optimiser from Optimisers.jl"
            arg_type = Symbol
            default = :Adam
        "--beta1"
            help = "Adam β1"
            arg_type = Float32
            default = 0.9f0
        "--beta2"
            help = "Adam β2"
            arg_type = Float32
            default = 0.999f0
        "--lambda"
            help = "Weight decay"
            arg_type = Float32
            default = 0.f0
        "--clip-grad"
            help = "Gradient clipping"
            arg_type = Float32
        "--time-limit", "--time"
            help = "Time limit for the training loop"
            arg_type = Float64
            default = Inf64
        "--manual-gc"
            help = "Whether to manually perform garbage collection after every epoch"
            action = :store_true

        # Solver args
        "--reltol"
            help = "Solver relative tolerance"
            arg_type = Float32
            default = 1f-4
        "--abstol"
            help = "Solver absolute tolerance"
            arg_type = Float32
            default = 1f-4
        "--solver"
            help = "Name of solver from OrdinaryDiffEq.jl"
            arg_type = SciMLBase.AbstractDEAlgorithm
            default = Tsit5()
        "--sensealg"
            help = "Name of sensitivity algorithm from SciMLSensitivity.jl"
            arg_type = Symbol
            default = :BacksolveAdjoint
        "--vjp"
            help = "Choice of AD for computing the vector-Jacobian product"
            arg_type = Symbol
            default = :ZygoteVJP
        "--preconditioner"
            help = "Preconditioner for projection linear solves (none, diagonal)"
            arg_type = Symbol
            default = :none

        # I/0
        "--verbose"
            help = "Whether to print loss for every training iteration"
            action = :store_true
        "--results-file"
            help = "Where to store description of results"
            arg_type = String
            default = "results.csv"
        #! format: on
    end

    args = parse_args(settings)
    if log
        log_args(args)
    end

    return args
end

function log_args(args)
    ordered_args = sort(collect(args); by = x -> x[1])
    for (arg_name, arg_value) in ordered_args
        @info "$arg_name = $arg_value"
    end
end

# Various functions for parsing custom types
# https://argparsejl.readthedocs.io/en/latest/argparse.html#parsing-to-custom-types
function eval_string(s)
    return eval(Meta.parse(s))
end

function ArgParse.parse_item(::Type{DataType}, type_name::AbstractString)
    return eval_string(type_name)
end

function ArgParse.parse_item(::Type{Function}, function_name::AbstractString)
    return eval_string(function_name)
end

function ArgParse.parse_item(
    ::Type{SciMLBase.AbstractDEAlgorithm},
    solver_name::AbstractString,
)
    return eval_string(solver_name * "()")
end

function ArgParse.parse_item(::Type{NamedTuple}, arg_string::AbstractString)
    return eval_string(arg_string)
end
