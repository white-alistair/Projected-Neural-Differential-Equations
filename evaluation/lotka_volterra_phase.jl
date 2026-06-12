# ==============================================================================
# Lotka-Volterra Phase Plots
# ==============================================================================
#
# This script creates a 2x2 phase plot figure for the Lotka-Volterra system,
# comparing ground truth trajectories with NDE and PNDE predictions.
#
# Usage:
#   include("lotka_volterra_phase.jl")
#   main()                    # Run and save figure
#   main(save_figure=false)   # Run without saving
#
# Prerequisites:
#   Run lotka_volterra.jl with save_data=true first.

using ProjectedNDEs: ProjectedNDEs, LotkaVolterra
using JLD2
using CairoMakie
using Colors

# ==============================================================================
# Configuration
# ==============================================================================

const DATA_DIR = joinpath(@__DIR__, "data")
const FIGURES_DIR = joinpath(@__DIR__, "figures")

# Data file
const LV_DATA = joinpath(DATA_DIR, "lotka_volterra.jld2")

# ==============================================================================
# Plotting Configuration
# ==============================================================================

include(joinpath(@__DIR__, "plot_colors.jl"))

# ==============================================================================
# Data Loading
# ==============================================================================

function load_predictions(data::Dict, prefix::String = "predictions_")
    predictions = Dict{Symbol,Any}()
    for (key, value) in data
        key_str = string(key)
        if startswith(key_str, prefix)
            model_name = Symbol(replace(key_str, prefix => ""))
            predictions[model_name] = value
        end
    end
    return predictions
end

# ==============================================================================
# Phase Plot Figure
# ==============================================================================

"""
Create a 2x2 phase plot figure showing ground truth, NDE, and PNDE for the first 4 trajectories.

Styling is consistent with the error comparison figure in lotka_volterra.jl.
"""
function create_figure(target, predictions)
    # Single column width for Chaos/AIP two-column format (~3.4 inches = 245 pt)
    # Height adjusted for 2x2 grid + legend
    fig = Figure(size = (245, 270), fontsize = 7, figure_padding = (1, 3, 2, 0))

    panel_labels = ["IC 1", "IC 2", "IC 3", "IC 4"]
    linewidth = 1.0  # Same thickness for all

    for traj_idx = 1:4
        row = (traj_idx - 1) ÷ 2 + 1
        col = (traj_idx - 1) % 2 + 1

        ax = Axis(
            fig[row, col],
            xlabel = row == 2 ? L"x" : "",
            ylabel = col == 1 ? L"y" : "",
            xlabelsize = 10,
            ylabelsize = 10,
            xlabelpadding = 0,
            ylabelpadding = 2,
            aspect = 1,
            title = panel_labels[traj_idx],
            titlefont = :bold,
            titlegap = 2,
        )

        # Ground truth
        x_true = target[1, :, traj_idx]
        y_true = target[2, :, traj_idx]
        lines!(ax, x_true, y_true, color = :black, linewidth = linewidth)

        # NDE
        if haskey(predictions, :nde)
            pred_nde = predictions[:nde]
            x_nde = pred_nde[1, :, traj_idx]
            y_nde = pred_nde[2, :, traj_idx]
            lines!(ax, x_nde, y_nde, color = get_color(:nde), linewidth = linewidth)
        end

        # PNDE
        if haskey(predictions, :pnde)
            pred_pnde = predictions[:pnde]
            x_pnde = pred_pnde[1, :, traj_idx]
            y_pnde = pred_pnde[2, :, traj_idx]
            lines!(ax, x_pnde, y_pnde, color = get_color(:pnde), linewidth = linewidth)
        end

        # Initial condition marker (brick red)
        scatter!(ax, [target[1, 1, traj_idx]], [target[2, 1, traj_idx]],
                 color = IC_COLOR, markersize = 6)
    end

    # Legend (initial condition first)
    linewidth = 2
    elements = [
        MarkerElement(color = IC_COLOR, marker = :circle, markersize = 8),
        LineElement(color = :black, linewidth = linewidth),
        LineElement(color = get_color(:nde), linewidth = linewidth),
        LineElement(color = get_color(:pnde), linewidth = linewidth),
    ]
    labels = [L"\mathrm{Initial\;Condition}", get_label(:ground_truth), get_label(:nde), get_label(:pnde)]

    Legend(
        fig[3, 1:2],
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

function main(; save_figure::Bool = true)
    println("="^60)
    println("Lotka-Volterra Phase Plots")
    println("="^60)

    # Check that data file exists
    if !isfile(LV_DATA)
        error(
            "Lotka-Volterra data not found at $LV_DATA\n" *
            "Run lotka_volterra.jl with save_data=true first.",
        )
    end

    # Load data
    println("Loading Lotka-Volterra data from $LV_DATA...")
    lv_data = load(LV_DATA)
    target = lv_data["target_trajectories"]
    predictions = load_predictions(lv_data)

    # Create 2x2 phase figure
    println("\nCreating 2x2 phase figure...")
    fig = create_figure(target, predictions)

    if save_figure
        output_path = joinpath(FIGURES_DIR, "lotka_volterra_phase.pdf")
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
