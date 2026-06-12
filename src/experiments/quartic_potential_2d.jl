struct QuarticPotential2D{T} <: AbstractDynamicalSystem{T} end

# Ground truth equations of motion
function (system::QuarticPotential2D)(du, u, p, t)
    x, y, vx, vy = u
    r_sq = x^2 + y^2

    du[1] = vx
    du[2] = vy
    du[3] = -x * r_sq
    du[4] = -y * r_sq

    return nothing
end

function initial_conditions(::QuarticPotential2D, rng::Random.AbstractRNG)
    x = -1 + 2 * rand(rng)
    y = -1 + 2 * rand(rng)
    vx = -1 + 2 * rand(rng)
    vy = -1 + 2 * rand(rng)
    return [x, y, vx, vy]
end

# Constraints: Energy E = (1/2)(vx² + vy²) + (1/4)(x² + y²)²
#              Angular momentum L = x*vy - y*vx
function constraints(u::AbstractMatrix{T}, t, ::QuarticPotential2D{T}) where {T}
    x, y, vx, vy = u[1:1, :], u[2:2, :], u[3:3, :], u[4:4, :]
    r_sq = x .^ 2 .+ y .^ 2
    E = T(0.5) .* (vx .^ 2 .+ vy .^ 2) .+ T(0.25) .* r_sq .^ 2
    # L = x .* vy .- y .* vx
    return E #vcat(E, L)
end

# 3D version for batched trajectories: (state_dim, time, n_trajectories)
function constraints(u::AbstractArray{T,3}, t, system::QuarticPotential2D{T}) where {T}
    return stack([constraints(u[:, :, i], t, system) for i in axes(u, 3)], dims = 3)
end

# Jacobian of constraints
# dE/d[x,y,vx,vy] = [x*(x²+y²), y*(x²+y²), vx, vy]
# dL/d[x,y,vx,vy] = [vy, -vx, -y, x]
function constraints_jacobian(u::AbstractMatrix, t, ::QuarticPotential2D)
    x, y, vx, vy = u[1, :], u[2, :], u[3, :], u[4, :]
    r_sq = x .^ 2 .+ y .^ 2

    # Build (2, 4, N) Jacobian tensor
    return stack(
        [
            stack([x .* r_sq, y .* r_sq, vx, vy], dims = 1),  # dE row: (4, N)
            # stack([vy, -vx, -y, x], dims = 1),                # dL row: (4, N)
        ],
        dims = 1,
    )
end

function transform(trajectory, ::QuarticPotential2D, experiment_version)
    return trajectory
end

# EXPERIMENT 1: Neural ODE
function get_model(
    ::QuarticPotential2D{T},
    ::Val{1},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    kwargs...,
) where {T}
    return SecondOrderAutonomousNeuralODE(
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

# EXPERIMENT 2: SNDE
function get_model(
    system::QuarticPotential2D{T},
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
    return SecondOrderAutonomousStabilizedNeuralODE(
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

# EXPERIMENT 3: PNDE
function get_model(
    system::QuarticPotential2D{T},
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
    return SecondOrderAutonomousProjectedNeuralODE(
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
