# IEEE 14 Bus — Publication Figures for Long Rollout Analysis
#
# Creates:
#   1. Main 2x2 figure: P injection (PQ buses) + |V| (PV buses)
#   2. Supplementary 3x3: P injection at all 9 PQ buses
#   3. Supplementary 2x2: |V| at all 4 PV buses
#
# Usage:
#   julia --project=. evaluation/ieee14bus_long_rollout_figures.jl

using ProjectedNDEs:
    ProjectedNDEs,
    IEEE14Bus,
    load_checkpoint,
    get_model,
    cpu,
    normalize,
    denormalize,
    operating_point
using OrdinaryDiffEq
using Lux
using Random
using LinearAlgebra: norm
using Statistics
using JLD2
using CairoMakie
using Colors

const CHECKPOINTS = Dict(
    :nde  => "7787801_6",
    :pnde => "7787814_3",
)

include(joinpath(@__DIR__, "plot_colors.jl"))

const NDE_LABEL  = L"\mathrm{NDE}"
const PNDE_LABEL = L"\mathrm{PNDE}"

const PQ_NODES = [4, 5, 7, 9, 10, 11, 12, 13, 14]
const PV_NODES = [1, 3, 6, 8]  # non-PQ, non-slack

function generate_data(;
    device = Lux.gpu_device(),
    t_end = 500.0f0,
    dt = 0.1f0,
    n_traj = 50,
)
    println("="^60)
    println("Generating long rollout data (T = $(t_end)s)")
    println("="^60)

    T = Float32
    system = IEEE14Bus{T}(device)
    hidden_layers = 10
    hidden_width = 512
    activation = gelu
    abstol = reltol = 1e-6

    raw_data = JLD2.load_object("power_grids/ieee14bus/data_tau_02_T_10_dt_01_all_nodes.jld2")
    op = operating_point(system)

    ids = 101:(100 + n_traj)
    test_data = device(T.(raw_data[:, :, ids]))
    g = ProjectedNDEs.constraints(test_data, nothing, system)
    u0 = test_data[:, 1, :]
    g0 = g[:, 1, :]
    times = collect(0.0f0:dt:t_end)

    results = Dict{Symbol,Any}()
    results[:times] = times
    results[:g0] = cpu(g0)
    results[:operating_point] = Float32.(op)

    for (model_key, experiment_version) in [(:nde, Val(1)), (:pnde, Val(3))]
        ckpt_id = CHECKPOINTS[model_key]
        println("  Loading $(model_key) ($(ckpt_id))...")
        checkpoint_params, _, _, normalization = load_checkpoint(ckpt_id, adapt_to=device)
        model = get_model(
            system, experiment_version, device, Random.default_rng(), activation,
            hidden_layers, hidden_width, normalization
        )
        u0_norm = normalize(u0, normalization)
        @time pred = model(u0_norm, g0, times, checkpoint_params; abstol, reltol)
        results[model_key] = cpu(denormalize(pred, normalization))
    end

    return results
end

function compute_constraint_values(pred)
    cpu_system = IEEE14Bus{Float32}(Lux.cpu_device())
    n_times = size(pred, 2)
    n_traj = size(pred, 3)
    n_pq = length(cpu_system.PQ_nodes)
    g_all = zeros(Float32, 2 * n_pq, n_times, n_traj)
    for j in 1:n_traj, i in 1:n_times
        g_all[:, i, j] = ProjectedNDEs.constraints(pred[:, i, j], nothing, cpu_system)
    end
    return g_all
end

# ==============================================================================
# Main 2x2 Figure
# ==============================================================================

function create_main_figure(results, nde_g, pnde_g; save_figure = true)
    times = results[:times]
    op = results[:operating_point]
    n_buses = 14
    cpu_system = IEEE14Bus{Float32}(Lux.cpu_device())
    pq_nodes = cpu_system.PQ_nodes

    nde_pred = results[:nde]
    pnde_pred = results[:pnde]
    n_traj = size(nde_pred, 3)

    nde_vmag = sqrt.(nde_pred[1:n_buses, :, :].^2 .+ nde_pred[n_buses+1:2*n_buses, :, :].^2)
    pnde_vmag = sqrt.(pnde_pred[1:n_buses, :, :].^2 .+ pnde_pred[n_buses+1:2*n_buses, :, :].^2)
    op_vmag = sqrt.(op[1:n_buses].^2 .+ op[n_buses+1:2*n_buses].^2)

    fig = Figure(size = (245, 270), fontsize = 7, figure_padding = (1, 3, 2, 0))

    xtickvals = [0, 250, 500]

    pq_bus_picks = [4, 10]
    pq_idxs = [findfirst(==(b), pq_nodes) for b in pq_bus_picks]

    pv_bus_picks = [1, 3]

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    panel_idx = 1

    # ---- Top row: Active power injection at PQ buses ----
    for (col, (bus_num, pq_i)) in enumerate(zip(pq_bus_picks, pq_idxs))
        ax = Axis(
            fig[1, col],
            ylabel = L"P_{%$bus_num}  \;\mathrm{[p.u.]}",
            aspect = 1,
            title = panel_labels[panel_idx],
            titlefont = :bold,
            titlegap = 2,
            xticklabelsvisible = false,
            ylabelpadding = 1,
            yticklabelpad = 2,
            xticks = xtickvals,
        )
        panel_idx += 1

        hlines!(ax, [0.0], color = (:gray70, 0.5), linewidth = 0.5)

        for j in 1:n_traj
            lines!(ax, times, nde_g[pq_i, :, j],
                color = (NDE_COLOR, 0.2), linewidth = 0.5)
        end
        for j in 1:n_traj
            lines!(ax, times, pnde_g[pq_i, :, j],
                color = (PNDE_COLOR, 0.2), linewidth = 0.5)
        end
    end

    # ---- Bottom row: Voltage magnitude at PV buses ----
    for (col, bus) in enumerate(pv_bus_picks)
        ax = Axis(
            fig[2, col],
            ylabel = L"|V_{%$bus}|  \;\mathrm{[p.u.]}",
            xlabel = "Time [s]",
            aspect = 1,
            title = panel_labels[panel_idx],
            titlefont = :bold,
            titlegap = 2,
            xlabelpadding = 1,
            xticklabelpad = 2,
            ylabelpadding = 1,
            yticklabelpad = 2,
            xticks = xtickvals,
        )
        panel_idx += 1

        hlines!(ax, [op_vmag[bus]], color = OP_COLOR, linewidth = 0.8, linestyle = :dash)

        for j in 1:n_traj
            lines!(ax, times, nde_vmag[bus, :, j],
                color = (NDE_COLOR, 0.2), linewidth = 0.5)
        end
        for j in 1:n_traj
            lines!(ax, times, pnde_vmag[bus, :, j],
                color = (PNDE_COLOR, 0.2), linewidth = 0.5)
        end
    end

    # Legend
    linewidth = 2
    elements = [
        LineElement(color = NDE_COLOR, linewidth = linewidth),
        LineElement(color = PNDE_COLOR, linewidth = linewidth),
        LineElement(color = OP_COLOR, linewidth = linewidth, linestyle = :dash),
    ]
    labels = [NDE_LABEL, PNDE_LABEL, L"\mathrm{Op.\;point}"]

    Legend(
        fig[3, 1:2],
        elements, labels,
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        rowgap = 2,
        colgap = 8,
        patchsize = (15, 8),
        padding = (4, 4, 4, 4),
    )

    colgap!(fig.layout, 6)
    rowgap!(fig.layout, 3)

    if save_figure
        path = joinpath(@__DIR__, "figures", "ieee14bus_long_rollout_main.pdf")
        CairoMakie.save(path, fig)
        println("Saved main figure to $path")
    end

    return fig
end

# ==============================================================================
# Supplementary: 3x3 Active Power at All PQ Buses
# ==============================================================================

function create_supp_pq_figure(results, nde_g, pnde_g; save_figure = true)
    times = results[:times]
    cpu_system = IEEE14Bus{Float32}(Lux.cpu_device())
    pq_nodes = cpu_system.PQ_nodes
    n_traj = size(results[:nde], 3)

    fig = Figure(size = (245, 270), fontsize = 6, figure_padding = (1, 2, 1, 0))
    xtickvals = [0, 250, 500]

    for (i, bus_num) in enumerate(pq_nodes)
        pq_i = i
        row = (i - 1) ÷ 3 + 1
        col = (i - 1) % 3 + 1
        is_last_row = row == 3

        ax = Axis(
            fig[row, col],
            ylabel = L"P_{%$bus_num}",
            xlabel = is_last_row ? "Time [s]" : "",
            aspect = 1,
            title = "Bus $bus_num",
            titlefont = :bold,
            titlegap = 1,
            xlabelpadding = 1,
            ylabelpadding = 1,
            xticklabelpad = 1,
            yticklabelpad = 1,
            xticks = xtickvals,
            xticklabelsvisible = is_last_row,
        )

        hlines!(ax, [0.0], color = (:gray70, 0.5), linewidth = 0.5)

        for j in 1:n_traj
            lines!(ax, times, nde_g[pq_i, :, j],
                color = (NDE_COLOR, 0.15), linewidth = 0.4)
        end
        for j in 1:n_traj
            lines!(ax, times, pnde_g[pq_i, :, j],
                color = (PNDE_COLOR, 0.15), linewidth = 0.4)
        end
    end

    linewidth = 2
    elements = [
        LineElement(color = NDE_COLOR, linewidth = linewidth),
        LineElement(color = PNDE_COLOR, linewidth = linewidth),
    ]
    labels = [NDE_LABEL, PNDE_LABEL]

    Legend(
        fig[4, 1:3],
        elements, labels,
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        colgap = 8,
        patchsize = (15, 8),
        padding = (4, 4, 4, 4),
    )

    colgap!(fig.layout, 3)
    rowgap!(fig.layout, 3)

    if save_figure
        path = joinpath(@__DIR__, "figures", "ieee14bus_long_rollout_supp_pq.pdf")
        CairoMakie.save(path, fig)
        println("Saved supplementary PQ figure to $path")
    end

    return fig
end

# ==============================================================================
# Supplementary: 2x2 Voltage Magnitude at All PV Buses
# ==============================================================================

function create_supp_pv_figure(results; save_figure = true)
    times = results[:times]
    op = results[:operating_point]
    n_buses = 14

    nde_pred = results[:nde]
    pnde_pred = results[:pnde]
    n_traj = size(nde_pred, 3)

    nde_vmag = sqrt.(nde_pred[1:n_buses, :, :].^2 .+ nde_pred[n_buses+1:2*n_buses, :, :].^2)
    pnde_vmag = sqrt.(pnde_pred[1:n_buses, :, :].^2 .+ pnde_pred[n_buses+1:2*n_buses, :, :].^2)
    op_vmag = sqrt.(op[1:n_buses].^2 .+ op[n_buses+1:2*n_buses].^2)

    fig = Figure(size = (245, 270), fontsize = 7, figure_padding = (1, 3, 2, 0))
    xtickvals = [0, 250, 500]

    for (i, bus) in enumerate(PV_NODES)
        row = (i - 1) ÷ 2 + 1
        col = (i - 1) % 2 + 1
        is_last_row = row == 2

        ax = Axis(
            fig[row, col],
            ylabel = L"|V_{%$bus}|  \;\mathrm{[p.u.]}",
            xlabel = is_last_row ? "Time [s]" : "",
            aspect = 1,
            title = "Bus $bus",
            titlefont = :bold,
            titlegap = 2,
            xlabelpadding = 1,
            xticklabelpad = 2,
            ylabelpadding = 1,
            yticklabelpad = 2,
            xticks = xtickvals,
            xticklabelsvisible = is_last_row,
        )

        hlines!(ax, [op_vmag[bus]], color = OP_COLOR, linewidth = 0.8, linestyle = :dash)

        for j in 1:n_traj
            lines!(ax, times, nde_vmag[bus, :, j],
                color = (NDE_COLOR, 0.2), linewidth = 0.5)
        end
        for j in 1:n_traj
            lines!(ax, times, pnde_vmag[bus, :, j],
                color = (PNDE_COLOR, 0.2), linewidth = 0.5)
        end
    end

    linewidth = 2
    elements = [
        LineElement(color = NDE_COLOR, linewidth = linewidth),
        LineElement(color = PNDE_COLOR, linewidth = linewidth),
        LineElement(color = OP_COLOR, linewidth = linewidth, linestyle = :dash),
    ]
    labels = [NDE_LABEL, PNDE_LABEL, L"\mathrm{Op.\;point}"]

    Legend(
        fig[3, 1:2],
        elements, labels,
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        colgap = 8,
        patchsize = (15, 8),
        padding = (4, 4, 4, 4),
    )

    colgap!(fig.layout, 6)
    rowgap!(fig.layout, 3)

    if save_figure
        path = joinpath(@__DIR__, "figures", "ieee14bus_long_rollout_supp_pv.pdf")
        CairoMakie.save(path, fig)
        println("Saved supplementary PV figure to $path")
    end

    return fig
end

# ==============================================================================
# Main Entry Point
# ==============================================================================

function main(; t_end = 500.0f0, n_traj = 50)
    results = generate_data(; t_end, n_traj)

    println("\nComputing constraint values...")
    nde_g = compute_constraint_values(results[:nde])
    pnde_g = compute_constraint_values(results[:pnde])

    println("\nCreating main figure...")
    create_main_figure(results, nde_g, pnde_g)

    println("Creating supplementary PQ figure...")
    create_supp_pq_figure(results, nde_g, pnde_g)

    println("Creating supplementary PV figure...")
    create_supp_pv_figure(results)

    println("\nDone!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
