# ==============================================================================
# N-Pendulum Experiment Evaluation
# ==============================================================================
#
# Evaluates NDE, SNDE, and PNDE models on N-pendulum systems for N = 4, 8
# using checkpoints from npendulum27 (400 train, 200 valid trajectories).
#
# Usage:
#   include("n_pendulum.jl")
#   main()

using ProjectedNDEs:
    ProjectedNDEs,
    NPendulum,
    initial_conditions,
    cartesian,
    constraints,
    load_checkpoint,
    get_model,
    cpu,
    normalize,
    denormalize,
    get_mean_relative_state_error,
    get_mean_relative_constraint_error,
    MAGMACholesky
using OrdinaryDiffEq
using Lux
using Random
using Statistics
using JLD2
using CairoMakie
using Colors
using ProgressBars

# ==============================================================================
# Configuration
# ==============================================================================

# Project root directory (for finding checkpoints)
const PROJECT_ROOT = dirname(@__DIR__)
const CHECKPOINT_DIR = joinpath(PROJECT_ROOT, "checkpoints")

# npendulum27 checkpoints (400 train, 200 valid trajectories)
# Models:
# - NDE Angular: Val{1} - angular coordinates [θ, ω]
# - SNDE: Val{3} - Cartesian with stabilization (multiple γ)
# - PNDE: Val{4} - Cartesian with projection
const CONFIGS = Dict(
    4 => (
        checkpoint_nde_angular = "6775410_1",
        checkpoint_snde = Dict{Float32,String}(
            20.0f0 => "6783810_1",
            200.0f0 => "6830472_1",
        ),
        checkpoint_pnde = "6775410_4",
        hidden_layers = 10,
        hidden_width = 1024,
    ),
    8 => (
        checkpoint_nde_angular = "6775414_1",
        checkpoint_snde = Dict{Float32,String}(
            20.0f0 => "6783810_2",
            200.0f0 => "6830472_2",
        ),
        checkpoint_pnde = "6775414_4",
        hidden_layers = 10,
        hidden_width = 1024,
    ),
)

const EVAL_CONFIG = (
    activation = gelu,
    use_skip = true,
    t0 = 0.0f0,
    t1 = 5.0f0,
    dt = 0.01f0,
    n_trajectories = 100,
    seed = 123,
    generation_tol = 1e-9,
    evaluation_tol = 1e-6,
    b = 0.1f0,
)

include(joinpath(@__DIR__, "plot_colors.jl"))

# ==============================================================================
# Data Generation
# ==============================================================================

"""
Generate ground truth trajectories for an N-pendulum system.
Returns both angular [θ, ω] and Cartesian [x, y, dx, dy] representations.
"""
function generate_trajectories(system::NPendulum{T,N}, config; device) where {T,N}
    rng = Random.default_rng()
    Random.seed!(rng, config.seed)

    tspan = (Float64(config.t0), Float64(config.t1))
    saveat = config.t0:config.dt:config.t1

    trajectories_angular = Vector{Matrix{T}}()

    for _ in ProgressBar(1:config.n_trajectories)
        u0 = initial_conditions(system, rng)
        prob = ODEProblem(system, u0, tspan)
        sol = solve(
            prob,
            Tsit5();
            saveat,
            reltol = config.generation_tol,
            abstol = config.generation_tol,
        )
        push!(trajectories_angular, T.(stack(sol.u)))
    end

    trajectories_angular = stack(trajectories_angular)
    trajectories_cartesian = T.(cartesian(trajectories_angular, system))

    return device(trajectories_angular), device(trajectories_cartesian)
end

# ==============================================================================
# Model Evaluation
# ==============================================================================

function evaluate_models(
    system::NPendulum{T,N},
    target_angular,
    target_cartesian,
    config;
    device,
) where {T,N}
    times = config.t0:config.dt:config.t1
    model_config = CONFIGS[N]

    u0_angular = target_angular[:, 1, :]
    u0_cartesian = target_cartesian[:, 1, :]
    g0 = constraints(u0_cartesian, nothing, system)

    results = Dict{Symbol,Any}()

    # NDE Angular (Val{1})
    println("Evaluating NDE Angular (N=$N)...")
    checkpoint_data = load_checkpoint(
        model_config.checkpoint_nde_angular;
        dir = CHECKPOINT_DIR,
        adapt_to = device,
    )
    normalization = length(checkpoint_data) >= 4 ? checkpoint_data[4] : nothing
    model = get_model(
        system,
        Val(1),
        device,
        Random.default_rng(),
        config.activation,
        model_config.hidden_layers,
        model_config.hidden_width,
        normalization;
        use_skip = config.use_skip,
    )

    u0_norm = normalization !== nothing ? normalize(u0_angular, normalization) : u0_angular
    @time pred_angular = model(
        u0_norm,
        g0,
        times,
        checkpoint_data[1];
        abstol = config.evaluation_tol,
        reltol = config.evaluation_tol,
    )
    pred_angular =
        normalization !== nothing ? denormalize(pred_angular, normalization) : pred_angular
    results[:nde_angular] = T.(cartesian(pred_angular, system))

    # SNDE variants (Val{3}) - one per gamma value
    for (γ, checkpoint_id) in sort(collect(model_config.checkpoint_snde); by = first)
        println("Evaluating SNDE γ=$γ (N=$N)...")
        checkpoint_data = load_checkpoint(
            checkpoint_id;
            dir = CHECKPOINT_DIR,
            adapt_to = device,
        )
        normalization = length(checkpoint_data) >= 4 ? checkpoint_data[4] : nothing
        model = get_model(
            system,
            Val(3),
            device,
            Random.default_rng(),
            config.activation,
            model_config.hidden_layers,
            model_config.hidden_width,
            normalization;
            γ = γ,
            use_skip = config.use_skip,
            backend = MAGMACholesky(),
        )

        u0_norm = normalization !== nothing ? normalize(u0_cartesian, normalization) : u0_cartesian
        @time pred = model(
            u0_norm,
            g0,
            times,
            checkpoint_data[1];
            abstol = config.evaluation_tol,
            reltol = config.evaluation_tol,
        )
        pred = normalization !== nothing ? denormalize(pred, normalization) : pred
        key = Symbol("snde_$(Int(γ))")
        results[key] = pred
    end

    # PNDE (Val{4})
    println("Evaluating PNDE (N=$N)...")
    checkpoint_data = load_checkpoint(
        model_config.checkpoint_pnde;
        dir = CHECKPOINT_DIR,
        adapt_to = device,
    )
    normalization = length(checkpoint_data) >= 4 ? checkpoint_data[4] : nothing
    model = get_model(
        system,
        Val(4),
        device,
        Random.default_rng(),
        config.activation,
        model_config.hidden_layers,
        model_config.hidden_width,
        normalization;
        use_skip = config.use_skip,
        backend = MAGMACholesky(),
    )

    u0_norm = normalization !== nothing ? normalize(u0_cartesian, normalization) : u0_cartesian
    @time pred = model(
        u0_norm,
        g0,
        times,
        checkpoint_data[1];
        abstol = config.evaluation_tol,
        reltol = config.evaluation_tol,
    )
    results[:pnde] = normalization !== nothing ? denormalize(pred, normalization) : pred

    return times, results
end

# ==============================================================================
# Metrics
# ==============================================================================

function print_metrics(N::Int, system, target, predictions)
    println("\n" * "="^60)
    println("Results Summary (N=$N)")
    println("="^60)

    for (name, pred) in predictions
        pred_cpu = cpu(pred)
        target_cpu = cpu(target)

        # State MSE
        errors_squared = (pred_cpu .- target_cpu) .^ 2
        state_mse = mean(errors_squared)
        state_ci = 1.96 * std(errors_squared) / sqrt(size(pred_cpu, 3))

        # Constraint MSE
        g_pred = constraints(pred_cpu, nothing, system)
        g_true = constraints(target_cpu, nothing, system)
        constraint_errors = g_pred .- g_true
        constraint_mse = mean(abs2, constraint_errors)
        constraint_ci = 1.96 * std(constraint_errors) / sqrt(size(constraint_errors, 3))

        println("$name:")
        println("  State MSE: $state_mse ± $state_ci")
        println("  Constraint Error: $constraint_mse ± $constraint_ci")
    end
end

# ==============================================================================
# Plotting
# ==============================================================================

"""
Build model display order from predictions, placing SNDE variants (sorted by γ)
between :nde_angular and :pnde.
"""
function build_model_order(predictions::Dict{Symbol,Any})
    order = Symbol[]
    haskey(predictions, :nde_angular) && push!(order, :nde_angular)
    snde_keys = sort(filter(k -> startswith(string(k), "snde_"), collect(keys(predictions)));
        by = k -> parse(Int, replace(string(k), "snde_" => "")))
    append!(order, snde_keys)
    haskey(predictions, :pnde) && push!(order, :pnde)
    return order
end


function create_figure(
    systems::Dict{Int,NPendulum},
    times,
    targets::Dict{Int,Any},
    all_predictions::Dict{Int,Dict{Symbol,Any}},
)
    N_values = sort(collect(keys(systems)))
    n_rows = length(N_values)

    # Build a unified model order from all N values
    model_order = Symbol[]
    for N in N_values
        for m in build_model_order(all_predictions[N])
            m in model_order || push!(model_order, m)
        end
    end

    fig = Figure(size = (245, 110 * n_rows), fontsize = 7, figure_padding = (1, 3, 2, 0))

    for (row, N) in enumerate(N_values)
        system = systems[N]
        target_cpu = cpu(targets[N])
        predictions = all_predictions[N]
        ordered_models = filter(k -> k in keys(predictions), model_order)

        ax_state = Axis(
            fig[row, 1],
            aspect = 1,
            xlabel = row == n_rows ? "Time [s]" : "",
            ylabel = "Relative State Error",
            xticks = 0:1:5,
            ylabelpadding = 2,
            yticklabelpad = 2,
            xlabelpadding = 1,
            xticklabelpad = 2,
        )

        for model in ordered_models
            pred_cpu = cpu(predictions[model])
            rel_err = get_mean_relative_state_error(pred_cpu, target_cpu)
            lines!(ax_state, collect(times), rel_err, color = get_color(model), linewidth = 1)
        end

        ax_constraint = Axis(
            fig[row, 2],
            aspect = 1,
            xlabel = row == n_rows ? "Time [s]" : "",
            ylabel = "Constraint Error",
            yscale = log10,
            xticks = 0:1:5,
            yticks = [1e-6, 1e-4, 1e-2, 1e0, 1e2],
            ytickformat = values -> ["10$(Makie.UnicodeFun.to_superscript(round(Int, log10(v))))" for v in values],
            ylabelpadding = 2,
            yticklabelpad = 2,
            xlabelpadding = 1,
            xticklabelpad = 2,
        )

        for model in ordered_models
            pred_cpu = cpu(predictions[model])
            g_pred = constraints(pred_cpu, nothing, system)
            constraint_err = vec(mean(sqrt.(sum(abs2, g_pred, dims=1)), dims=3))
            lines!(ax_constraint, collect(times), constraint_err, color = get_color(model), linewidth = 1)
        end

        Label(fig[row, 0], "N = $N", font = :bold, fontsize = 8, rotation = π / 2, tellheight = false)

        if row < n_rows
            hidexdecorations!(ax_state, grid = false, ticks = false)
            hidexdecorations!(ax_constraint, grid = false, ticks = false)
        end
    end

    linewidth = 2
    elements = [LineElement(; color = get_color(m), linewidth) for m in model_order]
    labels = [get_label(m) for m in model_order]

    nbanks = max(1, cld(length(model_order), 3))
    Legend(
        fig[n_rows + 1, 0:2],
        elements,
        labels,
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        nbanks = nbanks,
        rowgap = 2,
        colgap = 8,
        patchsize = (15, 8),
        padding = (4, 4, 4, 4),
    )

    colsize!(fig.layout, 0, Auto())
    colgap!(fig.layout, 1, 2)
    colgap!(fig.layout, 2, 6)
    rowgap!(fig.layout, 3)
    rowgap!(fig.layout, n_rows, 6)

    return fig
end

# ==============================================================================
# Main Entry Point
# ==============================================================================

function main(; save_figures::Bool = true, device = nothing)
    device = device === nothing ? MLDataDevices.CUDADevice() : device
    T = Float32
    N_values = sort(collect(keys(CONFIGS)))

    println("="^60)
    println("N-Pendulum Experiment Evaluation")
    println("="^60)
    println("  Friction: b=$(EVAL_CONFIG.b)")
    println("  Time: $(EVAL_CONFIG.t0) to $(EVAL_CONFIG.t1), dt=$(EVAL_CONFIG.dt)")
    println("  Trajectories: $(EVAL_CONFIG.n_trajectories)")
    println("  N values: $N_values")
    println("="^60)

    systems = Dict{Int,NPendulum}()
    targets_angular = Dict{Int,Any}()
    targets_cartesian = Dict{Int,Any}()
    all_predictions = Dict{Int,Dict{Symbol,Any}}()
    all_times = nothing

    for N in N_values
        println("\n" * "="^60)
        println("Processing N = $N")
        println("="^60)

        system = NPendulum{T,N}(EVAL_CONFIG.b)
        systems[N] = system

        println("\n[1/2] Generating test trajectories...")
        target_angular, target_cartesian =
            generate_trajectories(system, EVAL_CONFIG; device)
        targets_angular[N] = target_angular
        targets_cartesian[N] = target_cartesian

        println("\n[2/2] Evaluating models...")
        times, predictions =
            evaluate_models(system, target_angular, target_cartesian, EVAL_CONFIG; device)
        all_times = times
        all_predictions[N] = predictions

        print_metrics(N, system, target_cartesian, predictions)
    end

    println("\n" * "="^60)
    println("Creating figure...")
    println("="^60)
    fig = create_figure(systems, all_times, targets_cartesian, all_predictions)

    if save_figures
        output_path = joinpath(@__DIR__, "figures", "n_pendulum.pdf")
        CairoMakie.save(output_path, fig)
        println("Saved figure to $output_path")
    end

    return (; systems, times = all_times, targets_cartesian, all_predictions, fig)
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
