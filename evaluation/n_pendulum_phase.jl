# ==============================================================================
# N-Pendulum Phase Plots
# ==============================================================================
#
# This script creates a phase plot figure for N-pendulum systems (N=4, 8),
# showing the position of the endpoint (x_N, y_N) in Cartesian coordinates.
# Each row corresponds to a different N value, each column to a different initial condition.
#
# Usage:
#   include("pendulum_phase.jl")
#   main()                    # Run and save figures
#   main(save_figure=false)   # Run without saving
#
# Prerequisites:
#   Requires checkpoints for N-pendulum from npendulum experiments.

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
    MAGMACholesky
using OrdinaryDiffEq
using Lux
using Random
using JLD2
using CairoMakie
using Colors

# ==============================================================================
# Configuration
# ==============================================================================

# Project root directory (for finding checkpoints)
const PROJECT_ROOT = dirname(@__DIR__)
const CHECKPOINT_DIR = joinpath(PROJECT_ROOT, "checkpoints")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

# N values to evaluate
const N_VALUES = [4, 8]

# Model configurations for each N (from npendulum14)
const MODEL_CONFIGS = Dict(
    4 => (
        checkpoint_nde_angular = 6508776,
        checkpoint_pnde = 6508772,
        hidden_layers = 10,
        hidden_width = 512,
    ),
    8 => (
        checkpoint_nde_angular = 6508779,
        checkpoint_pnde = 6508774,
        hidden_layers = 10,
        hidden_width = 512,
    ),
)

const EVAL_CONFIG = (
    activation = gelu,
    use_skip = true,
    t0 = 0.0f0,
    t1 = 3.0f0,
    dt = 0.01f0,
    n_trajectories = 2,  # Generate 2 trajectories for 3x2 plot
    seed = 123,          # Same seed as n_pendulum.jl for reproducibility
    generation_tol = 1e-9,
    evaluation_tol = 1e-6,
    b = 0.1f0,  # Friction coefficient
)

# ==============================================================================
# Plotting Configuration
# ==============================================================================

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

    for _ = 1:config.n_trajectories
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

"""
Evaluate NDE Angular and PNDE models for an N-pendulum system.
Returns predictions in Cartesian coordinates.
"""
function evaluate_models(
    system::NPendulum{T,N},
    target_angular,
    target_cartesian,
    config;
    device,
) where {T,N}
    times = config.t0:config.dt:config.t1
    model_config = MODEL_CONFIGS[N]

    # Get initial conditions
    u0_angular = target_angular[:, 1, :]
    u0_cartesian = target_cartesian[:, 1, :]

    # Constraints are computed in Cartesian space
    g0 = constraints(u0_cartesian, nothing, system)

    results = Dict{Symbol,Any}()

    # NDE Angular (Val{1}) - operates in angular coordinates
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
    # Convert to Cartesian
    results[:nde] = T.(cartesian(cpu(pred_angular), system))

    # PNDE (Val{4}) - Cartesian with projection
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
    @time pred_cartesian = model(
        u0_norm,
        g0,
        times,
        checkpoint_data[1];
        abstol = config.evaluation_tol,
        reltol = config.evaluation_tol,
    )
    pred_cartesian_cpu = cpu(pred_cartesian)
    if normalization !== nothing
        # Move normalization to CPU for denormalization
        norm_cpu = ProjectedNDEs.Normalization(cpu(normalization.μ), cpu(normalization.σ), normalization.ε)
        results[:pnde] = denormalize(pred_cartesian_cpu, norm_cpu)
    else
        results[:pnde] = pred_cartesian_cpu
    end

    return times, results
end

# ==============================================================================
# Figure Creation
# ==============================================================================

"""
Create a 3x2 phase plot figure showing endpoint position (x_N, y_N).
Each row is a different N value, each column is a different initial condition.
"""
function create_figure(n_values, all_targets, all_predictions)
    # Single column width for Chaos/AIP two-column format (~3.4 inches = 245 pt)
    # Height adjusted for 3x2 grid + legend
    fig = Figure(size = (245, 360), fontsize = 7, figure_padding = (1, 3, 2, 0))

    panel_labels = ["IC 1", "IC 2"]
    linewidth = 1.0

    for (row_idx, N) in enumerate(n_values)
        target_cartesian = all_targets[N]
        predictions = all_predictions[N]

        for col_idx = 1:2
            ax = Axis(
                fig[row_idx, col_idx],
                xlabel = L"x_{%$N}",
                ylabel = col_idx == 1 ? L"y_{%$N}" : "",
                xlabelsize = 10,
                ylabelsize = 10,
                xlabelpadding = 0,
                ylabelpadding = 2,
                aspect = 1,
                title = row_idx == 1 ? panel_labels[col_idx] : "",
                titlefont = :bold,
                titlegap = 2,
            )

            # Ground truth - endpoint position (x_N, y_N)
            # Cartesian state: [x_1, ..., x_N, y_1, ..., y_N, dx_1, ..., dx_N, dy_1, ..., dy_N]
            x_true = target_cartesian[N, :, col_idx]      # x_N
            y_true = target_cartesian[2N, :, col_idx]     # y_N
            lines!(ax, x_true, y_true, color = :black, linewidth = linewidth)

            # NDE
            if haskey(predictions, :nde)
                pred_nde = predictions[:nde]
                x_nde = pred_nde[N, :, col_idx]
                y_nde = pred_nde[2N, :, col_idx]
                lines!(ax, x_nde, y_nde, color = get_color(:nde), linewidth = linewidth)
            end

            # PNDE
            if haskey(predictions, :pnde)
                pred_pnde = predictions[:pnde]
                x_pnde = pred_pnde[N, :, col_idx]
                y_pnde = pred_pnde[2N, :, col_idx]
                lines!(ax, x_pnde, y_pnde, color = get_color(:pnde), linewidth = linewidth)
            end

            # Initial condition marker (brick red)
            scatter!(ax, [target_cartesian[N, 1, col_idx]], [target_cartesian[2N, 1, col_idx]],
                     color = IC_COLOR, markersize = 6)
        end

        # Add N label on the left
        Label(fig[row_idx, 0], "N = $N", font = :bold, rotation = π / 2, tellheight = false)
    end

    # Legend (initial condition first)
    legend_linewidth = 2
    elements = [
        MarkerElement(color = IC_COLOR, marker = :circle, markersize = 8),
        LineElement(color = :black, linewidth = legend_linewidth),
        LineElement(color = get_color(:nde), linewidth = legend_linewidth),
        LineElement(color = get_color(:pnde), linewidth = legend_linewidth),
    ]
    labels = [L"\mathrm{Initial\;Condition}", get_label(:ground_truth), get_label(:nde), get_label(:pnde)]

    Legend(
        fig[4, 1:2],
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

    colsize!(fig.layout, 0, Fixed(4))
    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 3)

    return fig
end

# ==============================================================================
# Main Entry Point
# ==============================================================================

function main(; save_figure::Bool = true, device = nothing)
    device = device === nothing ? Lux.MLDataDevices.CUDADevice() : device
    T = Float32

    println("="^60)
    println("N-Pendulum Phase Plots")
    println("="^60)
    println("  N values: $N_VALUES")
    println("  Friction: b=$(EVAL_CONFIG.b)")
    println("  Time: $(EVAL_CONFIG.t0) to $(EVAL_CONFIG.t1), dt=$(EVAL_CONFIG.dt)")
    println("  Seed: $(EVAL_CONFIG.seed)")
    println("  Trajectories: $(EVAL_CONFIG.n_trajectories)")
    println("="^60)

    # Storage for all data
    all_targets = Dict{Int,Any}()
    all_predictions = Dict{Int,Any}()

    for N in N_VALUES
        println("\n" * "="^60)
        println("Processing N = $N")
        println("="^60)

        # Create system with friction
        system = NPendulum{T,N}(EVAL_CONFIG.b)

        # Generate test trajectories
        println("\n[1/3] Generating test trajectories...")
        target_angular, target_cartesian = generate_trajectories(system, EVAL_CONFIG; device)

        # Evaluate models
        println("\n[2/3] Evaluating models...")
        times, predictions = evaluate_models(system, target_angular, target_cartesian, EVAL_CONFIG; device)

        # Store results
        all_targets[N] = cpu(target_cartesian)
        all_predictions[N] = predictions
    end

    # Create combined figure
    println("\n[3/3] Creating combined figure...")
    fig = create_figure(N_VALUES, all_targets, all_predictions)

    if save_figure
        output_path = joinpath(FIGURES_DIR, "pendulum_phase_combined.pdf")
        CairoMakie.save(output_path, fig)
        println("Saved figure to $output_path")
    end

    println("\nDone!")
    return fig
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
