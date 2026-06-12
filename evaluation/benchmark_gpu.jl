# ==============================================================================
# GPU Benchmark Script for Neural ODE Models
# ==============================================================================
#
# This script benchmarks GPU performance for all neural ODE models from the
# evaluation scripts, measuring:
#   1. Single forward pass of the full RHS (including constraints) - batch size 1 and 1024
#   2. Full ODE integration - batch size 1 and 1024
#
# Usage:
#   julia --project=. evaluation/benchmark_gpu.jl
#
# Or interactively:
#   include("evaluation/benchmark_gpu.jl")
#   results = main()

using ProjectedNDEs:
    ProjectedNDEs,
    NPendulum,
    LotkaVolterra,
    QuarticPotential2D,
    IEEE14Bus,
    initial_conditions,
    constraints,
    constraints_jacobian,
    cartesian,
    load_checkpoint,
    get_model,
    normalize,
    denormalize,
    min_norm_solve,
    project_to_nullspace,
    MAGMACholesky,
    cuSOLVERCholesky
using OrdinaryDiffEq
using SciMLBase
using Lux
using Random
using Statistics
using Dates
using BenchmarkTools
using CUDA
using JLD2

# ==============================================================================
# Configuration
# ==============================================================================

const PROJECT_ROOT = dirname(@__DIR__)
const CHECKPOINT_DIR = joinpath(PROJECT_ROOT, "checkpoints")
const OUTPUT_DIR = joinpath(@__DIR__, "benchmark_results")

# Benchmark settings
const BENCHMARK_EVALS_FORWARD = 5   # Multiple evals for fast forward passes
const BENCHMARK_EVALS_ODE = 1       # Single eval for longer ODE integrations
const BENCHMARK_SAMPLES = 10        # Fixed number of samples per benchmark
const BENCHMARK_SECONDS = 120
const WARMUP_ITERATIONS = 5         # Number of warmup iterations before benchmarking

# Solver backends to benchmark (for SNDE/PNDE models that use a linear solver)
const BACKENDS = [
    MAGMACholesky() => "magma_cholesky",
    cuSOLVERCholesky() => "cusolver",
    # MAGMAQR() => "magma_qr",
    # cuSOLVERQR() => "cusolver_qr",
]

# ==============================================================================
# Checkpoint Configurations
# ==============================================================================

# Each entry: (system_name, model_name, checkpoint_id, model_variant, architecture, time_config, extra_kwargs)
const BENCHMARK_CONFIGS = [
    # N-Pendulum N=4 (10 layers, 1024 width, use_skip=true) — npendulum27 checkpoints
    (
        system = :npendulum,
        N = 4,
        model = :nde_angular,
        checkpoint = "6775410_1",
        variant = Val(1),
        hidden_layers = 10,
        hidden_width = 1024,
        t0 = 0.0f0,
        t1 = 5.0f0,
        dt = 0.01f0,
        kwargs = (use_skip = true,),
    ),
    (
        system = :npendulum,
        N = 4,
        model = :snde_20,
        checkpoint = "6783810_1",
        variant = Val(3),
        hidden_layers = 10,
        hidden_width = 1024,
        t0 = 0.0f0,
        t1 = 5.0f0,
        dt = 0.01f0,
        kwargs = (use_skip = true, γ = 20.0f0),
    ),
    (
        system = :npendulum,
        N = 4,
        model = :snde_200,
        checkpoint = "6830472_1",
        variant = Val(3),
        hidden_layers = 10,
        hidden_width = 1024,
        t0 = 0.0f0,
        t1 = 5.0f0,
        dt = 0.01f0,
        kwargs = (use_skip = true, γ = 200.0f0),
    ),
    (
        system = :npendulum,
        N = 4,
        model = :pnde,
        checkpoint = "6775410_4",
        variant = Val(4),
        hidden_layers = 10,
        hidden_width = 1024,
        t0 = 0.0f0,
        t1 = 5.0f0,
        dt = 0.01f0,
        kwargs = (use_skip = true,),
    ),

    # N-Pendulum N=8 (10 layers, 1024 width, use_skip=true) — npendulum27 checkpoints
    (
        system = :npendulum,
        N = 8,
        model = :nde_angular,
        checkpoint = "6775414_1",
        variant = Val(1),
        hidden_layers = 10,
        hidden_width = 1024,
        t0 = 0.0f0,
        t1 = 5.0f0,
        dt = 0.01f0,
        kwargs = (use_skip = true,),
    ),
    (
        system = :npendulum,
        N = 8,
        model = :snde_20,
        checkpoint = "6783810_2",
        variant = Val(3),
        hidden_layers = 10,
        hidden_width = 1024,
        t0 = 0.0f0,
        t1 = 5.0f0,
        dt = 0.01f0,
        kwargs = (use_skip = true, γ = 20.0f0),
    ),
    (
        system = :npendulum,
        N = 8,
        model = :snde_200,
        checkpoint = "6830472_2",
        variant = Val(3),
        hidden_layers = 10,
        hidden_width = 1024,
        t0 = 0.0f0,
        t1 = 5.0f0,
        dt = 0.01f0,
        kwargs = (use_skip = true, γ = 200.0f0),
    ),
    (
        system = :npendulum,
        N = 8,
        model = :pnde,
        checkpoint = "6775414_4",
        variant = Val(4),
        hidden_layers = 10,
        hidden_width = 1024,
        t0 = 0.0f0,
        t1 = 5.0f0,
        dt = 0.01f0,
        kwargs = (use_skip = true,),
    ),

    # Lotka-Volterra (6 layers, 512 width)
    (
        system = :lotka_volterra,
        N = nothing,
        model = :nde,
        checkpoint = 6230034,
        variant = Val(1),
        hidden_layers = 6,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 100.0f0,
        dt = 0.01f0,
        kwargs = (),
    ),
    (
        system = :lotka_volterra,
        N = nothing,
        model = :pnde,
        checkpoint = 6244648,
        variant = Val(3),
        hidden_layers = 6,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 100.0f0,
        dt = 0.01f0,
        kwargs = (),
    ),

    # Quartic Potential 2D (4 layers, 512 width) — quartic8 checkpoints
    (
        system = :quartic_potential,
        N = nothing,
        model = :nde,
        checkpoint = 6504429,
        variant = Val(1),
        hidden_layers = 4,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 100.0f0,
        dt = 0.01f0,
        kwargs = (),
    ),
    (
        system = :quartic_potential,
        N = nothing,
        model = :snde_0p1,
        checkpoint = 6504431,
        variant = Val(2),
        hidden_layers = 4,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 100.0f0,
        dt = 0.01f0,
        kwargs = (γ = 0.1f0,),
    ),
    (
        system = :quartic_potential,
        N = nothing,
        model = :snde_1,
        checkpoint = 6504432,
        variant = Val(2),
        hidden_layers = 4,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 100.0f0,
        dt = 0.01f0,
        kwargs = (γ = 1.0f0,),
    ),
    (
        system = :quartic_potential,
        N = nothing,
        model = :snde_10,
        checkpoint = 6504433,
        variant = Val(2),
        hidden_layers = 4,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 100.0f0,
        dt = 0.01f0,
        kwargs = (γ = 10.0f0,),
    ),
    (
        system = :quartic_potential,
        N = nothing,
        model = :pnde,
        checkpoint = 6504430,
        variant = Val(3),
        hidden_layers = 4,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 100.0f0,
        dt = 0.01f0,
        kwargs = (),
    ),

    # IEEE 14 Bus (10 layers, 512 width) — ieee14bus6 checkpoints
    (
        system = :ieee14bus,
        N = nothing,
        model = :nde,
        checkpoint = "7787801_6",
        variant = Val(1),
        hidden_layers = 10,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 10.0f0,
        dt = 0.1f0,
        kwargs = (),
    ),
    (
        system = :ieee14bus,
        N = nothing,
        model = :snde_0p1,
        checkpoint = "7787814_0",
        variant = Val(2),
        hidden_layers = 10,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 10.0f0,
        dt = 0.1f0,
        kwargs = (γ = 0.1f0,),
    ),
    (
        system = :ieee14bus,
        N = nothing,
        model = :snde_1,
        checkpoint = "7787814_1",
        variant = Val(2),
        hidden_layers = 10,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 10.0f0,
        dt = 0.1f0,
        kwargs = (γ = 1.0f0,),
    ),
    (
        system = :ieee14bus,
        N = nothing,
        model = :snde_10,
        checkpoint = "7787814_2",
        variant = Val(2),
        hidden_layers = 10,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 10.0f0,
        dt = 0.1f0,
        kwargs = (γ = 10.0f0,),
    ),
    (
        system = :ieee14bus,
        N = nothing,
        model = :pnde,
        checkpoint = "7787814_3",
        variant = Val(3),
        hidden_layers = 10,
        hidden_width = 512,
        t0 = 0.0f0,
        t1 = 10.0f0,
        dt = 0.1f0,
        kwargs = (),
    ),
]

# ==============================================================================
# Helper Functions
# ==============================================================================

"""
Create a dynamical system instance for the given configuration.
"""
function create_system(config; device)
    T = Float32
    if config.system == :npendulum
        return NPendulum{T,config.N}(0.1f0)  # b=0.1 friction
    elseif config.system == :lotka_volterra
        return LotkaVolterra{T}()
    elseif config.system == :quartic_potential
        return QuarticPotential2D{T}()
    elseif config.system == :ieee14bus
        return IEEE14Bus{T}(device)
    else
        error("Unknown system: $(config.system)")
    end
end

"""
Check if a benchmark config uses a solver backend (SNDE or PNDE models).
"""
function uses_backend(config)
    model_str = string(config.model)
    return contains(model_str, "snde") || model_str == "pnde"
end

"""
Get a human-readable label for a solver backend.
"""
backend_label(::MAGMACholesky) = "magma_cholesky"
backend_label(::cuSOLVERCholesky) = "cusolver_cholesky"

"""
Generate initial conditions for benchmarking.
For N-Pendulum, optionally converts to Cartesian coordinates.
"""
function generate_initial_conditions(
    system::NPendulum{T,N},
    batch_size::Int;
    device,
    rng = Random.default_rng(),
    use_cartesian::Bool = false,
) where {T,N}
    ics_angular = [T.(initial_conditions(system, rng)) for _ = 1:batch_size]
    u0_angular = stack(ics_angular, dims = 2)

    if use_cartesian
        # Convert to Cartesian coordinates
        u0 = T.(cartesian(u0_angular, system))
    else
        u0 = u0_angular
    end

    return device(u0)
end

# IEEE 14 Bus: load from data file (no initial_conditions method available)
const IEEE14BUS_DATA_PATH =
    joinpath(PROJECT_ROOT, "power_grids/ieee14bus/data_tau_02_T_10_dt_01_all_nodes.jld2")
const IEEE14BUS_DATA_CACHE = Ref{Union{Nothing,Array}}(nothing)

function generate_initial_conditions(
    system::IEEE14Bus,
    batch_size::Int;
    device,
    rng = Random.default_rng(),
    kwargs...,
)
    T = Float32
    # Load and cache the data file
    if IEEE14BUS_DATA_CACHE[] === nothing
        IEEE14BUS_DATA_CACHE[] = JLD2.load_object(IEEE14BUS_DATA_PATH)
    end
    raw_data = IEEE14BUS_DATA_CACHE[]

    # Use test set trajectories (indices 251:350 in evaluate.jl)
    # Select random trajectories based on batch_size
    n_available = size(raw_data, 3)
    if batch_size <= n_available
        # Randomly select batch_size trajectories
        indices = randperm(rng, n_available)[1:batch_size]
    else
        # If we need more than available, sample with replacement
        indices = rand(rng, 1:n_available, batch_size)
    end

    # Extract initial conditions (first timestep)
    u0 = T.(raw_data[:, 1, indices])
    return device(u0)
end

function generate_initial_conditions(
    system,
    batch_size::Int;
    device,
    rng = Random.default_rng(),
    kwargs...,
)
    T = Float32
    ics = [T.(initial_conditions(system, rng)) for _ = 1:batch_size]
    u0 = stack(ics, dims = 2)
    return device(u0)
end

"""
Get the RHS function for a model (includes constraints for SNDE/PNDE).
Dispatches based on model type.
"""
function get_rhs_function(model, params, g0)
    (; lux_model, state) = model

    # Check if it's a stabilized model (has γ field)
    if hasproperty(model, :γ)
        # Stabilized Neural ODE (SNDE)
        (; γ, system, backend) = model
        return (u, θ, t) -> begin
            du_dt = lux_model(u, θ, state)[1]
            G = constraints_jacobian(u, t, system)
            g = constraints(u, t, system) .- g0
            return du_dt .- γ * min_norm_solve(G, g, backend)
        end
    elseif hasproperty(model, :backend) && hasproperty(model, :system)
        # Projected Neural ODE (PNDE) - has backend and system but no γ
        (; system, backend) = model
        return (u, θ, t) -> begin
            du_dt = lux_model(u, θ, state)[1]
            G = constraints_jacobian(u, t, system)
            return project_to_nullspace(G, du_dt, backend)
        end
    else
        # Standard Neural ODE (NDE)
        return (u, θ, t) -> lux_model(u, θ, state)[1]
    end
end

"""
Benchmark a single forward pass of the RHS function.
"""
function benchmark_forward_pass(rhs, u, params, t = 0.0f0)
    # Extended warmup to stabilize GPU clocks and JIT
    for _ in 1:WARMUP_ITERATIONS
        rhs(u, params, t)
        CUDA.synchronize()
    end

    # Clear memory state before measurement
    GC.gc(false)
    CUDA.reclaim()
    CUDA.synchronize()

    trial = @benchmark begin
        $rhs($u, $params, $t)
        CUDA.synchronize()
    end evals = BENCHMARK_EVALS_FORWARD samples = BENCHMARK_SAMPLES seconds = BENCHMARK_SECONDS

    return trial
end

"""
Count the number of RHS evaluations during one ODE integration by wrapping
the model's RHS with a counter.
"""
function count_rhs_evaluations(model, u0, g0, times, params; abstol = 1e-4, reltol = 1e-4)
    (; lux_model, state) = model
    counter = Ref(0)

    if hasproperty(model, :γ)
        (; γ, system, backend) = model
        rhs = (u, θ, t) -> begin
            counter[] += 1
            du_dt = lux_model(u, θ, state)[1]
            G = constraints_jacobian(u, t, system)
            g = constraints(u, t, system) .- g0
            return du_dt .- γ * min_norm_solve(G, g, backend)
        end
    elseif hasproperty(model, :backend) && hasproperty(model, :system)
        (; system, backend) = model
        rhs = (u, θ, t) -> begin
            counter[] += 1
            du_dt = lux_model(u, θ, state)[1]
            G = constraints_jacobian(u, t, system)
            return project_to_nullspace(G, du_dt, backend)
        end
    else
        rhs = (u, θ, t) -> begin
            counter[] += 1
            lux_model(u, θ, state)[1]
        end
    end

    normalization = hasproperty(model, :normalization) ? model.normalization : nothing
    u0_denorm = denormalize(u0, normalization)
    t = times isa AbstractMatrix ? times[:, 1] : times
    tspan = (t[1], t[end])

    prob = ODEProblem{false,SciMLBase.FullSpecialize}(rhs, u0_denorm, tspan)
    solve(prob, Tsit5(); p = params, saveat = t, reltol, abstol)
    CUDA.synchronize()
    return counter[]
end

"""
Benchmark ODE integration. Returns (trial, n_steps) where n_steps is the number
of RHS evaluations taken during integration.
"""
function benchmark_ode_integration(
    model,
    u0,
    g0,
    times,
    params;
    abstol = 1e-4,
    reltol = 1e-4,
)
    # Extended warmup to stabilize GPU clocks and JIT
    for _ in 1:WARMUP_ITERATIONS
        model(u0, g0, times, params; abstol, reltol)
        CUDA.synchronize()
    end

    # Count RHS evaluations from one representative call
    n_steps = try
        count_rhs_evaluations(model, u0, g0, times, params; abstol, reltol)
    catch
        missing
    end

    # Clear memory state before measurement
    GC.gc(false)
    CUDA.reclaim()
    CUDA.synchronize()

    trial = @benchmark begin
        $model($u0, $g0, $times, $params; abstol = $abstol, reltol = $reltol)
        CUDA.synchronize()
    end evals = BENCHMARK_EVALS_ODE samples = BENCHMARK_SAMPLES seconds = BENCHMARK_SECONDS

    return trial, n_steps
end

"""
Format benchmark result as a string with min/mean/median/std.
"""
function format_benchmark(trial)
    min_val = minimum(trial.times) / 1e6  # Convert to ms
    mean_val = mean(trial.times) / 1e6
    med = median(trial.times) / 1e6
    std_val = std(trial.times) / 1e6
    return "min=$(round(min_val, digits=3)) mean=$(round(mean_val, digits=3)) med=$(round(med, digits=3)) ± $(round(std_val, digits=3)) ms"
end

# ==============================================================================
# Main Benchmark Function
# ==============================================================================

function benchmark_config(config; device, backend = nothing)
    be_str = backend !== nothing ? " [$(backend_label(backend))]" : ""
    println("\n" * "="^60)
    println(
        "Benchmarking: $(config.system) $(config.N !== nothing ? "N=$(config.N) " : "")$(config.model)$(be_str)",
    )
    println("  Checkpoint: $(config.checkpoint)")
    println("="^60)

    T = Float32
    activation = gelu

    # Create system
    system = create_system(config; device)

    # Load checkpoint
    println("Loading checkpoint...")
    checkpoint_data =
        load_checkpoint(config.checkpoint; dir = CHECKPOINT_DIR, adapt_to = device)
    params = checkpoint_data[1]
    normalization = length(checkpoint_data) >= 4 ? checkpoint_data[4] : nothing

    # Create model
    println("Creating model...")
    # Merge backend into kwargs if provided (forwarded to get_model → model constructor)
    model_kwargs = if backend !== nothing
        (; config.kwargs..., backend)
    else
        config.kwargs
    end

    model = get_model(
        system,
        config.variant,
        device,
        Random.default_rng(),
        activation,
        config.hidden_layers,
        config.hidden_width,
        normalization;
        model_kwargs...,
    )

    # Generate initial conditions
    println("Generating initial conditions...")
    rng = Random.MersenneTwister(42)  # Fixed seed for reproducibility

    # For N-Pendulum: Val{1} uses angular coords, Val{2,3,4} use Cartesian
    use_cartesian = (config.system == :npendulum && config.variant != Val(1))

    u0_bs1 = generate_initial_conditions(system, 1; device, rng, use_cartesian)
    rng = Random.MersenneTwister(42)
    u0_bs1024 = generate_initial_conditions(system, config.system == :npendulum ? 2048 : 1024; device, rng, use_cartesian)

    # Compute initial constraints
    # Note: For N-Pendulum Angular NDE (Val{1}), constraints aren't used but the function
    # expects Cartesian inputs. Generate Cartesian ICs just for constraint computation.
    if config.system == :npendulum && config.variant == Val(1)
        # For angular NDE, generate Cartesian ICs just for constraint computation
        rng_cart = Random.MersenneTwister(42)
        u0_cart_bs1 = generate_initial_conditions(
            system,
            1;
            device,
            rng = rng_cart,
            use_cartesian = true,
        )
        rng_cart = Random.MersenneTwister(42)
        u0_cart_bs1024 = generate_initial_conditions(
            system,
            1024;
            device,
            rng = rng_cart,
            use_cartesian = true,
        )
        g0_bs1 = constraints(u0_cart_bs1, nothing, system)
        g0_bs1024 = constraints(u0_cart_bs1024, nothing, system)
    else
        g0_bs1 = constraints(u0_bs1, nothing, system)
        g0_bs1024 = constraints(u0_bs1024, nothing, system)
    end

    # Normalize if needed
    if normalization !== nothing
        u0_bs1_norm = normalize(u0_bs1, normalization)
        u0_bs1024_norm = normalize(u0_bs1024, normalization)
    else
        u0_bs1_norm = u0_bs1
        u0_bs1024_norm = u0_bs1024
    end

    # Time configuration
    times = config.t0:config.dt:config.t1

    results = Dict{Symbol,Any}()
    results[:system] = string(config.system)
    results[:N] = config.N
    results[:model] = string(config.model)
    results[:checkpoint] = config.checkpoint
    results[:backend] = backend !== nothing ? backend_label(backend) : "n/a"

    # Get RHS function
    rhs = get_rhs_function(model, params, g0_bs1)
    rhs_1024 = get_rhs_function(model, params, g0_bs1024)

    # Benchmark forward pass (batch size 1)
    println("\nBenchmarking forward pass (batch size 1)...")
    trial_fwd_1 = benchmark_forward_pass(rhs, u0_bs1_norm, params)
    results[:forward_bs1_min_ms] = minimum(trial_fwd_1.times) / 1e6
    results[:forward_bs1_mean_ms] = mean(trial_fwd_1.times) / 1e6
    results[:forward_bs1_median_ms] = median(trial_fwd_1.times) / 1e6
    results[:forward_bs1_std_ms] = std(trial_fwd_1.times) / 1e6
    println("  Forward pass (bs=1): $(format_benchmark(trial_fwd_1))")

    # Benchmark forward pass (batch size 1024)
    println("Benchmarking forward pass (batch size 1024)...")
    trial_fwd_1024 = benchmark_forward_pass(rhs_1024, u0_bs1024_norm, params)
    results[:forward_bs1024_min_ms] = minimum(trial_fwd_1024.times) / 1e6
    results[:forward_bs1024_mean_ms] = mean(trial_fwd_1024.times) / 1e6
    results[:forward_bs1024_median_ms] = median(trial_fwd_1024.times) / 1e6
    results[:forward_bs1024_std_ms] = std(trial_fwd_1024.times) / 1e6
    println("  Forward pass (bs=1024): $(format_benchmark(trial_fwd_1024))")

    # Benchmark ODE integration (batch size 1)
    println("Benchmarking ODE integration (batch size 1)...")
    trial_ode_1, n_steps_bs1 = benchmark_ode_integration(model, u0_bs1_norm, g0_bs1, times, params)
    results[:ode_bs1_min_ms] = minimum(trial_ode_1.times) / 1e6
    results[:ode_bs1_mean_ms] = mean(trial_ode_1.times) / 1e6
    results[:ode_bs1_median_ms] = median(trial_ode_1.times) / 1e6
    results[:ode_bs1_std_ms] = std(trial_ode_1.times) / 1e6
    results[:ode_bs1_nf] = n_steps_bs1
    println("  ODE integration (bs=1): $(format_benchmark(trial_ode_1)) [nf=$(n_steps_bs1)]")

    # Benchmark ODE integration (batch size 1024)
    println("Benchmarking ODE integration (batch size 1024)...")
    trial_ode_1024, n_steps_bs1024 =
        benchmark_ode_integration(model, u0_bs1024_norm, g0_bs1024, times, params)
    results[:ode_bs1024_min_ms] = minimum(trial_ode_1024.times) / 1e6
    results[:ode_bs1024_mean_ms] = mean(trial_ode_1024.times) / 1e6
    results[:ode_bs1024_median_ms] = median(trial_ode_1024.times) / 1e6
    results[:ode_bs1024_std_ms] = std(trial_ode_1024.times) / 1e6
    results[:ode_bs1024_nf] = n_steps_bs1024
    println("  ODE integration (bs=1024): $(format_benchmark(trial_ode_1024)) [nf=$(n_steps_bs1024)]")

    return results
end

"""
Collect GPU metadata for reproducibility.
"""
function collect_gpu_metadata()
    meta = Dict{String,String}()
    meta["gpu_name"] = CUDA.name(CUDA.device())
    meta["cuda_runtime_version"] = string(CUDA.runtime_version())
    meta["cuda_driver_version"] = string(CUDA.driver_version())
    meta["gpu_memory_total_gb"] = string(round(CUDA.totalmem(CUDA.device()) / 1024^3, digits=2))
    # Attempt to get additional info via nvidia-smi
    try
        clocks = read(`nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader,nounits`, String)
        parts = strip.(split(strip(clocks), ","))
        if length(parts) >= 4
            meta["gpu_clock_sm_mhz"] = parts[1]
            meta["gpu_clock_mem_mhz"] = parts[2]
            meta["gpu_temperature_c"] = parts[3]
            meta["gpu_power_draw_w"] = parts[4]
        end
    catch
        # nvidia-smi not available or failed
    end
    return meta
end

function main(; save_results::Bool = true)
    device = Lux.gpu_device()
    println("Using device: $device")

    # Record GPU metadata
    gpu_meta = collect_gpu_metadata()
    println("\nGPU Metadata:")
    for (k, v) in sort(collect(gpu_meta))
        println("  $k: $v")
    end

    # Collect all results
    all_results = Vector{Dict{Symbol,Any}}()

    for config in BENCHMARK_CONFIGS
        # For SNDE/PNDE models, benchmark with each solver backend.
        # For NDE models (no backend), benchmark once.
        backends_to_test = if uses_backend(config)
            BACKENDS
        else
            [nothing => "n/a"]
        end

        for (be, be_name) in backends_to_test
            try
                results = benchmark_config(config; device, backend = be)
                push!(all_results, results)
            catch e
                println(
                    "ERROR benchmarking $(config.system) $(config.model) ($(be_name)): $e",
                )
                println(sprint(showerror, e, catch_backtrace()))
                # Add NaN/missing values for failed benchmarks
                push!(
                    all_results,
                    Dict{Symbol,Any}(
                        :system => string(config.system),
                        :N => config.N,
                        :model => string(config.model),
                        :checkpoint => config.checkpoint,
                        :backend => be_name,
                        :forward_bs1_min_ms => NaN,
                        :forward_bs1_mean_ms => NaN,
                        :forward_bs1_median_ms => NaN,
                        :forward_bs1_std_ms => NaN,
                        :forward_bs1024_min_ms => NaN,
                        :forward_bs1024_mean_ms => NaN,
                        :forward_bs1024_median_ms => NaN,
                        :forward_bs1024_std_ms => NaN,
                        :ode_bs1_min_ms => NaN,
                        :ode_bs1_mean_ms => NaN,
                        :ode_bs1_median_ms => NaN,
                        :ode_bs1_std_ms => NaN,
                        :ode_bs1_nf => missing,
                        :ode_bs1024_min_ms => NaN,
                        :ode_bs1024_mean_ms => NaN,
                        :ode_bs1024_median_ms => NaN,
                        :ode_bs1024_std_ms => NaN,
                        :ode_bs1024_nf => missing,
                    ),
                )
            end
        end
    end

    # Print summary
    println("\n" * "="^80)
    println("BENCHMARK RESULTS SUMMARY")
    println("="^80)

    # Print header
    header = "system,N,model,backend,fwd_bs1_med_ms,fwd_bs1024_med_ms,ode_bs1_med_ms,ode_bs1024_med_ms,ode_bs1_nf,ode_bs1024_nf"
    println(header)

    for r in all_results
        n_str = r[:N] === nothing ? "" : string(r[:N])
        line =
            "$(r[:system]),$(n_str),$(r[:model]),$(r[:backend])," *
            "$(round(r[:forward_bs1_median_ms], digits=3))," *
            "$(round(r[:forward_bs1024_median_ms], digits=3))," *
            "$(round(r[:ode_bs1_median_ms], digits=3))," *
            "$(round(r[:ode_bs1024_median_ms], digits=3))," *
            "$(r[:ode_bs1_nf]),$(r[:ode_bs1024_nf])"
        println(line)
    end

    if save_results
        mkpath(OUTPUT_DIR)
        output_file = joinpath(OUTPUT_DIR, "gpu_benchmark_results.csv")

        # Write CSV
        open(output_file, "w") do io
            println(
                io,
                "system,N,model,checkpoint,backend," *
                "forward_bs1_min_ms,forward_bs1_mean_ms,forward_bs1_median_ms,forward_bs1_std_ms," *
                "forward_bs1024_min_ms,forward_bs1024_mean_ms,forward_bs1024_median_ms,forward_bs1024_std_ms," *
                "ode_bs1_min_ms,ode_bs1_mean_ms,ode_bs1_median_ms,ode_bs1_std_ms,ode_bs1_nf," *
                "ode_bs1024_min_ms,ode_bs1024_mean_ms,ode_bs1024_median_ms,ode_bs1024_std_ms,ode_bs1024_nf",
            )
            for r in all_results
                n_str = r[:N] === nothing ? "" : string(r[:N])
                println(
                    io,
                    "$(r[:system]),$(n_str),$(r[:model]),$(r[:checkpoint]),$(r[:backend])," *
                    "$(r[:forward_bs1_min_ms]),$(r[:forward_bs1_mean_ms]),$(r[:forward_bs1_median_ms]),$(r[:forward_bs1_std_ms])," *
                    "$(r[:forward_bs1024_min_ms]),$(r[:forward_bs1024_mean_ms]),$(r[:forward_bs1024_median_ms]),$(r[:forward_bs1024_std_ms])," *
                    "$(r[:ode_bs1_min_ms]),$(r[:ode_bs1_mean_ms]),$(r[:ode_bs1_median_ms]),$(r[:ode_bs1_std_ms]),$(r[:ode_bs1_nf])," *
                    "$(r[:ode_bs1024_min_ms]),$(r[:ode_bs1024_mean_ms]),$(r[:ode_bs1024_median_ms]),$(r[:ode_bs1024_std_ms]),$(r[:ode_bs1024_nf])",
                )
            end
        end
        println("\nResults saved to: $output_file")

        # Save GPU metadata alongside results
        meta_file = joinpath(OUTPUT_DIR, "gpu_metadata.txt")
        open(meta_file, "w") do io
            println(io, "GPU Benchmark Metadata")
            println(io, "=" ^ 40)
            println(io, "Timestamp: $(Dates.now())")
            for (k, v) in sort(collect(gpu_meta))
                println(io, "$k: $v")
            end
            println(io, "\nBenchmark settings:")
            println(io, "  BENCHMARK_EVALS_FORWARD: $BENCHMARK_EVALS_FORWARD")
            println(io, "  BENCHMARK_EVALS_ODE: $BENCHMARK_EVALS_ODE")
            println(io, "  BENCHMARK_SAMPLES: $BENCHMARK_SAMPLES")
            println(io, "  WARMUP_ITERATIONS: $WARMUP_ITERATIONS")
        end
        println("Metadata saved to: $meta_file")
    end

    return all_results
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
