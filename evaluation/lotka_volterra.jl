# ==============================================================================
# Lotka-Volterra Experiment Evaluation
# ==============================================================================
#
# This script evaluates NDE and PNDE models on the Lotka-Volterra
# predator-prey system.
#
# Usage:
#   include("lotka_volterra.jl")
#   main()                    # Run full evaluation and generate error figure
#   main(save_data=true)      # Also save results to JLD2 (for phase plots)
#
# Phase plots:
#   Run with save_data=true, then use lotka_volterra_phase.jl

using ProjectedNDEs:
    ProjectedNDEs,
    LotkaVolterra,
    initial_conditions,
    constraints,
    load_checkpoint,
    get_model,
    cpu,
    normalize,
    denormalize,
    get_mean_relative_state_error,
    get_mean_relative_constraint_error
using OrdinaryDiffEq
using Lux
using Random
using Statistics
using JLD2
using CairoMakie
using ProgressBars

# ==============================================================================
# Configuration
# ==============================================================================

# Project root directory (for finding checkpoints)
const PROJECT_ROOT = dirname(@__DIR__)
const CHECKPOINT_DIR = joinpath(PROJECT_ROOT, "checkpoints")

const CONFIG = (
    # Checkpoints (from lotka_volterra3 experiments - H-filtered ICs, t1=5.0)
    checkpoint_nde = 6230034,     # Best NDE: 6 layers, 512 width, val=8.02e-07, test=2.59e-06
    checkpoint_pnde = 6244648,    # PNDE: 6 layers, 512 width, val=6.29e-07, test=2.33e-06

    # Model architecture (must match trained models)
    hidden_layers = 6,
    hidden_width = 512,
    activation = gelu,

    # Eval data generation
    t0 = 0.0f0,
    t1 = 100.0f0,   # Long-horizon extrapolation test (trained on t1=5.0)
    dt = 0.01f0,
    n_trajectories = 100,
    seed = 123,
    generation_tol = 1e-6,
    evaluation_tol = 1e-6,
)

# ==============================================================================
# Data Generation
# ==============================================================================

function generate_trajectories(system, config; device)
    T = Float32
    rng = Random.default_rng()
    Random.seed!(rng, config.seed)

    tspan = (Float64(config.t0), Float64(config.t1))
    saveat = config.t0:config.dt:config.t1

    trajectories = Vector{Matrix{T}}()

    for _ in ProgressBar(1:config.n_trajectories)
        u0 = initial_conditions(system, rng)
        prob = ODEProblem(system, u0, tspan)
        sol = solve(
            prob,
            Vern9();
            saveat,
            reltol = config.generation_tol,
            abstol = config.generation_tol,
        )
        push!(trajectories, stack(sol.u))
    end

    result = T.(stack(trajectories))
    return device(result)
end

# ==============================================================================
# Model Evaluation (Batch)
# ==============================================================================

function evaluate_models(system, target_trajectories, config; device)
    times = config.t0:config.dt:config.t1

    # Get initial conditions and constraints
    g = constraints(target_trajectories, nothing, system)
    u0 = target_trajectories[:, 1, :]
    g0 = g[:, 1, :]

    results = Dict{Symbol,Any}()

    # NDE
    println("Evaluating NDE...")
    checkpoint_data =
        load_checkpoint(config.checkpoint_nde; dir = CHECKPOINT_DIR, adapt_to = device)
    normalization = checkpoint_data[4]
    model = get_model(
        system,
        Val(1),
        device,
        Random.default_rng(),
        config.activation,
        config.hidden_layers,
        config.hidden_width,
        normalization,
    )
    u0_norm = normalization !== nothing ? normalize(u0, normalization) : u0
    @time pred = model(
        u0_norm,
        g0,
        times,
        checkpoint_data[1];
        abstol = config.evaluation_tol,
        reltol = config.evaluation_tol,
    )
    results[:nde] = normalization !== nothing ? denormalize(pred, normalization) : pred

    # PNDE
    println("Evaluating PNDE...")
    checkpoint_data =
        load_checkpoint(config.checkpoint_pnde; dir = CHECKPOINT_DIR, adapt_to = device)
    normalization = length(checkpoint_data) >= 4 ? checkpoint_data[4] : nothing
    model = get_model(
        system,
        Val(3),
        device,
        Random.default_rng(),
        config.activation,
        config.hidden_layers,
        config.hidden_width,
        normalization,
    )
    u0_norm = normalization !== nothing ? normalize(u0, normalization) : u0
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

function print_metrics(system, target, predictions)
    println("\n" * "="^60)
    println("Results Summary")
    println("="^60)

    for (name, pred) in predictions
        pred_cpu = cpu(pred)
        target_cpu = cpu(target)

        # State MSE
        errors_squared = (pred_cpu .- target_cpu) .^ 2
        state_mse = mean(errors_squared)
        state_ci = 1.96 * std(errors_squared) / sqrt(size(pred_cpu, 3))

        # Check for physical constraint violations (negative populations)
        min_prey = minimum(pred_cpu[1, :, :])
        min_predator = minimum(pred_cpu[2, :, :])
        n_negative = sum(pred_cpu .< 0)

        println("$name:")
        println("  State MSE: $state_mse ± $state_ci")
        println("  Min prey: $min_prey, Min predator: $min_predator")

        if min_prey > 0 && min_predator > 0
            # Constraint MSE (conserved quantity H) - only if all values positive
            g_pred = constraints(pred_cpu, nothing, system)
            g_true = constraints(target_cpu, nothing, system)
            constraint_errors = g_pred .- g_true
            constraint_mse = mean(abs2, constraint_errors)
            constraint_ci = 1.96 * std(constraint_errors) / sqrt(size(constraint_errors, 3))
            println("  Constraint Error: $constraint_mse ± $constraint_ci")
        else
            println(
                "  Constraint Error: N/A (negative values detected, $n_negative violations)",
            )
        end
    end
end

# ==============================================================================
# Plotting
# ==============================================================================

include(joinpath(@__DIR__, "plot_colors.jl"))

"""
Create the main comparison figure with two panels:
(a) Relative state error over time
(b) Conserved quantity error over time
"""
function create_figure(system, times, target, predictions)
    # Single column width for Chaos/AIP two-column format (~3.4 inches = 245 pt)
    fig = Figure(size = (245, 155), fontsize = 7, figure_padding = (1, 3, 2, 0))

    # Define model order: NDE, PNDE
    model_order = [:nde, :pnde]
    ordered_models = filter(k -> k in keys(predictions), model_order)

    target_cpu = cpu(target)

    # (a) Relative state error (square aspect ratio)
    ax1 = Axis(
        fig[1, 1],
        xlabel = "Time",
        ylabel = "Relative State Error",
        aspect = 1,
        title = "(a)",
        titlefont = :bold,
        titlegap = 2,
    )

    for model in ordered_models
        pred_cpu = cpu(predictions[model])
        rel_err = get_mean_relative_state_error(pred_cpu, target_cpu)
        lines!(ax1, times, rel_err, color = get_color(model), linewidth = 1)
    end

    # (b) Constraint error (log scale, square aspect ratio)
    ax2 = Axis(
        fig[1, 2],
        xlabel = "Time",
        ylabel = "Constraint Error",
        yscale = log10,
        aspect = 1,
        title = "(b)",
        titlefont = :bold,
        titlegap = 2,
    )

    for model in ordered_models
        pred_cpu = cpu(predictions[model])
        # Check for negative values (physical constraint violation)
        if any(pred_cpu .< 0)
            # Find last valid timestep (before any trajectory goes negative)
            last_valid = 1
            for t_idx = 1:size(pred_cpu, 2)
                if all(pred_cpu[:, t_idx, :] .> 0)
                    last_valid = t_idx
                else
                    break
                end
            end
            if last_valid > 1
                constraint_err = get_mean_relative_constraint_error(
                    system,
                    pred_cpu[:, 1:last_valid, :],
                    target_cpu[:, 1:last_valid, :],
                )
                lines!(
                    ax2,
                    times[1:last_valid],
                    constraint_err,
                    color = get_color(model),
                    linewidth = 1,
                )
            end
            # Note: model went unstable, constraint error not shown for full trajectory
        else
            constraint_err =
                get_mean_relative_constraint_error(system, pred_cpu, target_cpu)
            lines!(ax2, times, constraint_err, color = get_color(model), linewidth = 1)
        end
    end

    # Legend
    linewidth = 2
    elements = [LineElement(; color = get_color(m), linewidth) for m in ordered_models]
    labels = [get_label(m) for m in ordered_models]

    Legend(
        fig[2, 1:2],
        elements,
        labels,
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        nbanks = 1,
        rowgap = 2,
        colgap = 8,
        patchsize = (15, 8),
        padding = (4, 4, 4, 4),
    )

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 3)

    return fig
end

# ==============================================================================
# Main Entry Point
# ==============================================================================

function main(; save_data::Bool = false, device = nothing)
    device = device === nothing ? MLDataDevices.CUDADevice() : device
    T = Float32
    system = LotkaVolterra{T}()

    println("="^60)
    println("Lotka-Volterra Experiment")
    println("="^60)
    println("  System: α=$(system.α), β=$(system.β), γ=$(system.γ), δ=$(system.δ)")
    println("  Time: $(CONFIG.t0) to $(CONFIG.t1), dt=$(CONFIG.dt)")
    println("  Trajectories: $(CONFIG.n_trajectories)")
    println("  Architecture: $(CONFIG.hidden_layers) layers, $(CONFIG.hidden_width) width")
    println("="^60)

    # 1. Generate test data
    println("\n[1/3] Generating test trajectories...")
    target_trajectories = generate_trajectories(system, CONFIG; device)

    # 2. Evaluate models
    println("\n[2/3] Evaluating models...")
    times, predictions = evaluate_models(system, target_trajectories, CONFIG; device)

    # Print metrics
    print_metrics(system, target_trajectories, predictions)

    # 3. Create error comparison figure
    println("\n[3/3] Creating error comparison figure...")
    fig = create_figure(system, times, target_trajectories, predictions)

    # Save data if requested (required for phase plots and combined figure)
    if save_data
        output_path = joinpath(@__DIR__, "data", "lotka_volterra.jld2")
        mkpath(dirname(output_path))
        jldsave(
            output_path;
            times = cpu(times),
            target_trajectories = cpu(target_trajectories),
            (Symbol("predictions_$k") => cpu(v) for (k, v) in predictions)...,
        )
        println("Saved data to $output_path")
        println("Run lotka_volterra_phase.jl to generate phase plots.")
    end

    return (; times, target_trajectories, predictions, fig)
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
