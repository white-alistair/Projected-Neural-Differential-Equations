# Shared colour definitions for evaluation figures.
#
# Colour-blind friendly palette based on Wong (2011), Nature Methods 8:441.
# NDE and PNDE colours are anchored to the IEEE 14-bus long-rollout figures.
#
# Usage:  include(joinpath(@__DIR__, "plot_colors.jl"))
#
# Assumes `using Colors` and `using CairoMakie` have already been called.

# Primary model colours
const NDE_COLOR  = colorant"#E69F00"   # Orange
const PNDE_COLOR = colorant"#0072B2"   # Dark blue

# SNDE variant colours (assigned by ordinal position within each experiment)
const SNDE_COLOR_1 = colorant"#56B4E9"  # Sky blue   (lowest γ)
const SNDE_COLOR_2 = colorant"#009E73"  # Bluish green
const SNDE_COLOR_3 = colorant"#CC79A7"  # Reddish purple (highest γ)

# Auxiliary colours
const GROUND_TRUTH_COLOR = colorant"#000000"  # Black
const OP_COLOR           = colorant"#D55E00"  # Vermillion
const IC_COLOR           = colorant"#B22222"  # Brick red (initial condition markers)

# Lookup dictionary: model key → colour
const MODEL_COLORS = Dict{Symbol,Any}(
    :nde          => NDE_COLOR,
    :nde_angular  => NDE_COLOR,
    :snde_0p1     => SNDE_COLOR_1,
    :snde_1p0     => SNDE_COLOR_2,
    :snde_10p0    => SNDE_COLOR_3,
    :snde_20      => SNDE_COLOR_1,
    :snde_200     => SNDE_COLOR_2,
    :pnde         => PNDE_COLOR,
    :ground_truth => GROUND_TRUTH_COLOR,
)

# Lookup dictionary: model key → LaTeX label
const MODEL_LABELS = Dict{Symbol,Any}(
    :nde          => L"\mathrm{NDE}",
    :nde_angular  => L"\mathrm{NDE\;(Angular)}",
    :snde_0p1     => L"\mathrm{SNDE}\;(\gamma=0.1)",
    :snde_1p0     => L"\mathrm{SNDE}\;(\gamma=1)",
    :snde_10p0    => L"\mathrm{SNDE}\;(\gamma=10)",
    :snde_20      => L"\mathrm{SNDE}\;(\gamma=20)",
    :snde_200     => L"\mathrm{SNDE}\;(\gamma=200)",
    :pnde         => L"\mathrm{PNDE}",
    :ground_truth => L"\mathrm{Ground\;Truth}",
)

get_color(model::Symbol) = get(MODEL_COLORS, model, Makie.wong_colors()[1])
get_label(model::Symbol) = get(MODEL_LABELS, model, string(model))
