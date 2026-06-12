# IEEE 14 Bus Experiment 6 - Evaluation and Figure Generation
#
# Usage:
#   julia --project=. evaluation/ieee14bus.jl
#
# This script:
#   1. Loads trained models from checkpoints
#   2. Generates predictions on test data
#   3. Creates comparison figures

using ProjectedNDEs:
    ProjectedNDEs,
    IEEE14Bus,
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
using Colors

# ==============================================================================
# Configuration
# ==============================================================================

const PROJECT_ROOT = dirname(@__DIR__)
const CHECKPOINT_DIR = joinpath(PROJECT_ROOT, "checkpoints")

# Model checkpoints from ieee14bus6/results.csv
const CHECKPOINTS = Dict(
    :nde => "7787801_6",          # NDE (10 layers, 512 width) - best NDE
    :snde_0p1 => "7787814_0",    # SNDE γ=0.1
    :snde_1p0 => "7787814_1",    # SNDE γ=1.0
    :snde_10p0 => "7787814_2",   # SNDE γ=10.0
    :pnde => "7787814_3",        # PNDE - best overall
)

include(joinpath(@__DIR__, "plot_colors.jl"))

# ==============================================================================
# Prediction Generation
# ==============================================================================

function generate_predictions(; device=Lux.gpu_device(), save_data::Bool=true)
    println("="^60)
    println("IEEE 14 Bus Experiment 6 - Generating Predictions")
    println("="^60)

    T = Float32
    system = IEEE14Bus{T}(device)
    hidden_layers = 10
    hidden_width = 512
    activation = gelu
    abstol = reltol = 1e-4

    println("\nLoading test data...")
    ids = 101:200  # Test set (n_train=50, n_valid=50, then 100 test)
    raw_data = JLD2.load_object("power_grids/ieee14bus/data_tau_02_T_10_dt_01_all_nodes.jld2")
    target_trajectories = device(T.(raw_data[:, :, ids]))
    g = ProjectedNDEs.constraints(target_trajectories, nothing, system)
    times = collect(0.0f0:0.1f0:10.0f0)
    u0 = target_trajectories[:, 1, :]
    g0 = g[:, 1, :]

    predictions = Dict{Symbol, Any}()
    predictions[:times] = times
    predictions[:target] = cpu(target_trajectories)

    # NDE
    println("\nLoading NDE checkpoint $(CHECKPOINTS[:nde])...")
    checkpoint_params, _, _, normalization = load_checkpoint(CHECKPOINTS[:nde], adapt_to=device)
    model = get_model(
        system, Val(1), device, Random.default_rng(), activation,
        hidden_layers, hidden_width, normalization
    )
    u0_normalized = normalize(u0, normalization)
    println("Generating NDE predictions...")
    @time pred = model(u0_normalized, g0, times, checkpoint_params; abstol, reltol)
    predictions[:nde] = cpu(denormalize(pred, normalization))

    # SNDE γ=0.1
    println("\nLoading SNDE γ=0.1 checkpoint $(CHECKPOINTS[:snde_0p1])...")
    checkpoint_params, _, _, normalization = load_checkpoint(CHECKPOINTS[:snde_0p1], adapt_to=device)
    model = get_model(
        system, Val(2), device, Random.default_rng(), activation,
        hidden_layers, hidden_width, normalization; γ=0.1f0
    )
    u0_normalized = normalize(u0, normalization)
    println("Generating SNDE γ=0.1 predictions...")
    @time pred = model(u0_normalized, g0, times, checkpoint_params; abstol, reltol)
    predictions[:snde_0p1] = cpu(denormalize(pred, normalization))

    # SNDE γ=1.0
    println("\nLoading SNDE γ=1.0 checkpoint $(CHECKPOINTS[:snde_1p0])...")
    checkpoint_params, _, _, normalization = load_checkpoint(CHECKPOINTS[:snde_1p0], adapt_to=device)
    model = get_model(
        system, Val(2), device, Random.default_rng(), activation,
        hidden_layers, hidden_width, normalization; γ=1.0f0
    )
    u0_normalized = normalize(u0, normalization)
    println("Generating SNDE γ=1.0 predictions...")
    @time pred = model(u0_normalized, g0, times, checkpoint_params; abstol, reltol)
    predictions[:snde_1p0] = cpu(denormalize(pred, normalization))

    # SNDE γ=10.0
    println("\nLoading SNDE γ=10.0 checkpoint $(CHECKPOINTS[:snde_10p0])...")
    checkpoint_params, _, _, normalization = load_checkpoint(CHECKPOINTS[:snde_10p0], adapt_to=device)
    model = get_model(
        system, Val(2), device, Random.default_rng(), activation,
        hidden_layers, hidden_width, normalization; γ=10.0f0
    )
    u0_normalized = normalize(u0, normalization)
    println("Generating SNDE γ=10.0 predictions...")
    @time pred = model(u0_normalized, g0, times, checkpoint_params; abstol, reltol)
    predictions[:snde_10p0] = cpu(denormalize(pred, normalization))

    # PNDE
    println("\nLoading PNDE checkpoint $(CHECKPOINTS[:pnde])...")
    checkpoint_params, _, _, normalization = load_checkpoint(CHECKPOINTS[:pnde], adapt_to=device)
    model = get_model(
        system, Val(3), device, Random.default_rng(), activation,
        hidden_layers, hidden_width, normalization
    )
    u0_normalized = normalize(u0, normalization)
    println("Generating PNDE predictions...")
    @time pred = model(u0_normalized, g0, times, checkpoint_params; abstol, reltol)
    predictions[:pnde] = cpu(denormalize(pred, normalization))

    if save_data
        output_file = joinpath(@__DIR__, "data", "ieee14bus.jld2")
        jldsave(output_file; predictions...)
        println("\nSaved predictions to $output_file")
    end

    return predictions
end

# ==============================================================================
# Figure Creation
# ==============================================================================

function create_ieee14bus_figure(predictions::Dict; save_figure::Bool=true)
    println("\nCreating comparison figure...")

    # Handle both Symbol and String keys (JLD2 loads as strings)
    get_key = k -> haskey(predictions, k) ? predictions[k] : predictions[string(k)]
    times = get_key(:times)
    target = get_key(:target)
    system = IEEE14Bus{Float32}(Lux.cpu_device())

    # Figure dimensions consistent with combined_qp_lv.jl (245pt width, square plots)
    fig = Figure(size=(245, 155), fontsize=7, figure_padding=(1, 3, 2, 0))

    model_order = [:nde, :snde_0p1, :snde_1p0, :pnde]
    available_models = filter(k -> haskey(predictions, k) || haskey(predictions, string(k)), model_order)

    # (a) Relative State Error (log scale)
    ax1 = Axis(
        fig[1, 1],
        xlabel="Time [s]",
        ylabel="Relative State Error",
        yscale=log10,
        aspect=1,
        title="(a)",
        titlefont=:bold,
        titlegap=2,
        xlabelpadding=1,
        ylabelpadding=1,
        xticklabelpad=2,
        yticklabelpad=2,
    )

    for model in available_models
        pred = get_key(model)
        rel_err = get_mean_relative_state_error(pred, target)
        lines!(ax1, times, rel_err, color=MODEL_COLORS[model], linewidth=1)
    end
    ylims!(ax1, (1e-4, nothing))

    # (b) Constraint Error (log scale)
    ax2 = Axis(
        fig[1, 2],
        xlabel="Time [s]",
        ylabel="Constraint Error",
        yscale=log10,
        aspect=1,
        title="(b)",
        titlefont=:bold,
        titlegap=2,
        xlabelpadding=1,
        ylabelpadding=1,
        xticklabelpad=2,
        yticklabelpad=2,
    )

    for model in available_models
        pred = get_key(model)
        constraint_err = get_mean_relative_constraint_error(system, pred, target)
        lines!(ax2, times, constraint_err, color=MODEL_COLORS[model], linewidth=1)
    end

    # Legend
    linewidth = 2
    elements = [LineElement(; color=MODEL_COLORS[m], linewidth) for m in available_models]
    labels = [MODEL_LABELS[m] for m in available_models]

    Legend(
        fig[2, 1:2],
        elements,
        labels,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
        nbanks=2,
        rowgap=2,
        colgap=8,
        patchsize=(15, 8),
        padding=(4, 4, 4, 4),
    )

    colgap!(fig.layout, 1, 6)
    rowgap!(fig.layout, 1, 3)

    if save_figure
        output_path = joinpath(@__DIR__, "figures", "ieee14bus.pdf")
        CairoMakie.save(output_path, fig)
        println("Saved figure to $output_path")
    end

    return fig
end

# ==============================================================================
# Main Entry Point
# ==============================================================================

function main(; generate::Bool=true, save_data::Bool=true, save_figure::Bool=true)
    println("="^60)
    println("IEEE 14 Bus Experiment 6 - Evaluation")
    println("="^60)

    predictions_file = joinpath(@__DIR__, "data", "ieee14bus.jld2")

    if generate || !isfile(predictions_file)
        predictions = generate_predictions(; save_data)
    else
        println("Loading cached predictions from $predictions_file")
        predictions = load(predictions_file)
    end

    fig = create_ieee14bus_figure(predictions; save_figure)
    display(fig)

    return fig
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
