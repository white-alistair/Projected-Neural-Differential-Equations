struct LotkaVolterra{T} <: AbstractDynamicalSystem{T}
    α::T  # prey growth rate
    β::T  # predation rate
    γ::T  # predator death rate
    δ::T  # predator reproduction rate
end

LotkaVolterra{T}() where {T} = LotkaVolterra{T}(1.5, 1.0, 3.0, 1.0)  # Classic params

# Ground truth equations of motion
# dx/dt = αx - βxy (prey growth minus predation)
# dy/dt = δxy - γy (predator reproduction minus death)
function (system::LotkaVolterra)(du, u, p, t)
    (; α, β, γ, δ) = system
    x, y = u

    du[1] = α * x - β * x * y
    du[2] = δ * x * y - γ * y

    return nothing
end

# IC regimes for exploring different regions of phase space
# With default params (α=1.5, β=1.0, γ=3.0, δ=1.0), equilibrium is at (x*, y*) = (3.0, 1.5)
const IC_REGIMES = Dict(
    :balanced => (x_range = (0.8, 3.0), y_range = (0.5, 2.5)),           # Around equilibrium
    :low_prey_high_predator => (x_range = (0.3, 1.0), y_range = (2.0, 4.0)),  # Stiff dynamics
    :high_prey_low_predator => (x_range = (3.0, 6.0), y_range = (0.3, 1.0)),  # Prey boom
    :high_prey_high_predator => (x_range = (3.0, 6.0), y_range = (2.0, 4.0)), # High populations
)

# Hamiltonian (conserved quantity) for a single state vector
# Larger H → larger orbit amplitude → more likely to pass through stiff low-prey region
# At equilibrium (3.0, 1.5) with default params: H ≈ 0.6
# H_max ≈ 4.0 keeps orbits from dipping below x ≈ 0.3
function hamiltonian(u, system::LotkaVolterra)
    (; α, β, γ, δ) = system
    x, y = u[1], u[2]
    return δ * x - γ * log(x) + β * y - α * log(y)
end

# Default H_max = 4.0 is conservative: rejects large-amplitude orbits that pass through
# the stiff low-prey region (x < 0.3) where predator dynamics become very fast
const DEFAULT_H_MAX = 4.0

function initial_conditions(
    system::LotkaVolterra,
    rng::Random.AbstractRNG;
    regime::Symbol = :balanced,
    H_max::Real = DEFAULT_H_MAX,
    max_attempts::Int = 100,
)
    ranges = IC_REGIMES[regime]
    x_lo, x_hi = ranges.x_range
    y_lo, y_hi = ranges.y_range

    for _ = 1:max_attempts
        x = x_lo + (x_hi - x_lo) * rand(rng)
        y = y_lo + (y_hi - y_lo) * rand(rng)
        u0 = [x, y]
        if hamiltonian(u0, system) ≤ H_max
            return u0
        end
    end

    error(
        "Failed to sample IC with H ≤ $H_max after $max_attempts attempts. " *
        "Try increasing H_max or using a different regime.",
    )
end

# Conserved quantity: H = δx - γ*log(x) + βy - α*log(y)
function constraints(u::AbstractMatrix{T}, t, system::LotkaVolterra{T}) where {T}
    (; α, β, γ, δ) = system
    x, y = u[1:1, :], u[2:2, :]
    H = δ .* x .- γ .* log.(x) .+ β .* y .- α .* log.(y)
    return H
end

# 3D version for batched trajectories: (state_dim, time, n_trajectories)
function constraints(u::AbstractArray{T,3}, t, system::LotkaVolterra{T}) where {T}
    return stack([constraints(u[:, :, i], t, system) for i in axes(u, 3)], dims = 3)
end

# Jacobian of constraint H w.r.t. [x, y]
# ∂H/∂x = δ - γ/x
# ∂H/∂y = β - α/y
function constraints_jacobian(u::AbstractMatrix, t, system::LotkaVolterra)
    (; α, β, γ, δ) = system
    x, y = u[1, :], u[2, :]

    # Build (1, 2, N) Jacobian tensor
    return stack([
        stack([δ .- γ ./ x, β .- α ./ y], dims = 1),  # dH row: (2, N)
    ], dims = 1)
end

function transform(trajectory, ::LotkaVolterra, experiment_version)
    return trajectory
end

# EXPERIMENT 1: First-order Neural ODE
function get_model(
    ::LotkaVolterra{T},
    ::Val{1},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    kwargs...,
) where {T}
    return FirstOrderAutonomousNeuralODE(
        2,
        hidden_layers,
        hidden_width,
        activation,
        rng,
        T,
        device;
        preprocess_mlp_inputs = u -> normalize(u, normalization),
        normalization,
    )
end

# EXPERIMENT 2: First-order SNDE (Stabilized Neural ODE)
function get_model(
    system::LotkaVolterra{T},
    experiment_version::Val{2},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    γ,
    backend = MAGMACholesky(),
    kwargs...,
) where {T}
    return FirstOrderAutonomousStabilizedNeuralODE(
        2,
        hidden_layers,
        hidden_width,
        activation,
        rng,
        T,
        device,
        system,
        γ;
        preprocess_mlp_inputs = u -> normalize(u, normalization),
        normalization,
        backend,
    )
end

# EXPERIMENT 3: First-order PNDE (Projected Neural ODE)
function get_model(
    system::LotkaVolterra{T},
    experiment_version::Val{3},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    backend = MAGMACholesky(),
    kwargs...,
) where {T}
    return FirstOrderAutonomousProjectedNeuralODE(
        2,
        hidden_layers,
        hidden_width,
        activation,
        rng,
        T,
        device,
        system;
        preprocess_mlp_inputs = u -> normalize(u, normalization),
        normalization,
        backend,
    )
end
