# ==============================================================================
# 2D Quartic Potential Experiment Evaluation
# ==============================================================================
#
# This script evaluates NDE, SNDE, and PNDE models on the 2D quartic potential
# system.
#
# Usage:
#   include("quartic_potential_2d.jl")
#   main()                    # Run full evaluation and generate figures
#   main(save_data=true)      # Also save results to JLD2

using ProjectedNDEs:
    ProjectedNDEs,
    QuarticPotential2D,
    initial_conditions,
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
using ProgressBars

# ==============================================================================
# Configuration
# ==============================================================================

# Project root directory (for finding checkpoints)
const PROJECT_ROOT = dirname(@__DIR__)
const CHECKPOINT_DIR = joinpath(PROJECT_ROOT, "checkpoints")

const CONFIG = (
    # Checkpoints (from quartic8 experiment - 2x training data)
    checkpoint_nde = 6504429,
    checkpoint_pnde = 6504430,
    checkpoint_snde = Dict{Float32,Int}(
        0.1f0 => 6504431,
        1.0f0 => 6504432,
        10.0f0 => 6504433,
    ),  # γ => checkpoint_id

    # Model architecture
    hidden_layers = 4,
    hidden_width = 512,
    activation = gelu,

    # Eval data generation
    t0 = 0.0f0,
    t1 = 100.0f0,
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
            Tsit5();
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
# Model Evaluation
# ==============================================================================

function evaluate_models(system, target_trajectories, config; device)
    times = config.t0:config.dt:config.t1

    # Get initial conditions
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

    # SNDE variants
    for (γ, checkpoint) in config.checkpoint_snde
        γ_str = replace(string(γ), "." => "p", "f0" => "")
        println("Evaluating SNDE γ=$γ...")
        checkpoint_data =
            load_checkpoint(checkpoint; dir = CHECKPOINT_DIR, adapt_to = device)
        normalization = length(checkpoint_data) >= 4 ? checkpoint_data[4] : nothing
        model = get_model(
            system,
            Val(2),
            device,
            Random.default_rng(),
            config.activation,
            config.hidden_layers,
            config.hidden_width,
            normalization;
            γ,
            backend = MAGMACholesky(),
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
        key = Symbol("snde_$(γ_str)")
        results[key] = normalization !== nothing ? denormalize(pred, normalization) : pred
    end

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
        normalization;
        backend = MAGMACholesky(),
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

        # Constraint MSE (energy conservation)
        g_pred = constraints(pred_cpu, nothing, system)
        g_true = constraints(target_cpu, nothing, system)
        constraint_errors = g_pred .- g_true
        constraint_mse = mean(abs2, constraint_errors)
        constraint_ci = 1.96 * std(constraint_errors) / sqrt(size(constraint_errors, 3))

        println("$name:")
        println("  State MSE: $state_mse ± $state_ci")
        println("  Energy Error: $constraint_mse ± $constraint_ci")
    end
end

# ==============================================================================
# Plotting
# ==============================================================================

include(joinpath(@__DIR__, "plot_colors.jl"))

"""
Create the main comparison figure with two panels:
(a) Relative state error over time
(b) Energy conservation error over time
"""
function create_figure(system, times, target, predictions)
    target_cpu = cpu(target)

    # Single column width for Chaos/AIP two-column format (~3.4 inches = 245 pt)
    fig = Figure(size = (245, 155), fontsize = 7, figure_padding = (1, 3, 2, 0))

    # Define model order: NDE, SNDE by increasing γ, PNDE
    model_order = [:nde, :snde_0p1, :snde_1p0, :snde_10p0, :pnde]
    ordered_models = filter(k -> k in keys(predictions), model_order)

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

    # (b) Energy error (log scale, square aspect ratio)
    ax2 = Axis(
        fig[1, 2],
        xlabel = "Time",
        ylabel = "Energy Error",
        yscale = log10,
        aspect = 1,
        title = "(b)",
        titlefont = :bold,
        titlegap = 2,
    )

    for model in ordered_models
        pred_cpu = cpu(predictions[model])
        energy_err = get_mean_relative_constraint_error(system, pred_cpu, target_cpu)
        lines!(ax2, times, energy_err, color = get_color(model), linewidth = 1)
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
        nbanks = 2,
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

function main(; save_data::Bool = false, save_figures::Bool = true, device = nothing)
    device = device === nothing ? MLDataDevices.CUDADevice() : device
    T = Float32
    system = QuarticPotential2D{T}()

    println("="^60)
    println("2D Quartic Potential Experiment")
    println("="^60)
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

    # 3. Create error figures
    println("\n[3/3] Creating error figures...")
    fig = create_figure(system, times, target_trajectories, predictions)
    display(fig)

    if save_figures
        output_path = joinpath(@__DIR__, "figures", "quartic_potential_2d.pdf")
        mkpath(dirname(output_path))
        CairoMakie.save(output_path, fig)
        println("Saved figure to $output_path")
    end

    # Save data if requested
    if save_data
        output_path = joinpath(@__DIR__, "data", "quartic_potential_2d.jld2")
        mkpath(dirname(output_path))
        jldsave(
            output_path;
            times = cpu(times),
            target_trajectories = cpu(target_trajectories),
            (Symbol("predictions_$k") => cpu(v) for (k, v) in predictions)...,
        )
        println("Saved data to $output_path")
    end

    return (; times, target_trajectories, predictions, fig)
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
