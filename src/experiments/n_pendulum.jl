struct NPendulum{T,N} <: AbstractDynamicalSystem{T}
    b::Float64  # Coefficient of friction
    g::Float64  # Acceleration due to gravity
end

NPendulum{T,N}(b) where {T,N} = NPendulum{T,N}(b, 9.81)

# Ground truth equations of motion
# Adapted from https://travisdoesmath.github.io/pendulum-explainer/
function (system::NPendulum{T,N})(u, p, t) where {T,N}
    (; b, g) = system

    θ = u[1:N]
    ω = u[N+1:end]

    c(i, j) = N - max(i, j) + 1

    A = [c(i, j) * cos(θ[i] - θ[j]) for i = 1:N, j = 1:N]
    y = [
        -sum([c(i, j) * ω[j]^2 * sin(θ[i] - θ[j]) for j = 1:N]) - g * (N - i + 1) * sin(θ[i])
        for i = 1:N
    ]  # This vector is called "b" in the source but renamed here
    dω = A \ y

    # We add friction proportional to the relative angular velocity at the joint
    dω .-= b * vcat(ω[1], [ω[i] - ω[i-1] for i = 2:N])

    return vcat(ω, dω)
end

function initial_conditions(::NPendulum{T,N}, rng::Random.AbstractRNG) where {T,N}
    θ = 2π .* rand(rng, N)
    ω = zeros(N)
    return vcat(θ, ω)  # These should be Float64 regardless of T
end

# Vectorized cartesian transformation - GPU and AD compatible
function cartesian(u::AbstractArray{T1,1}, ::NPendulum{T2,N}) where {T1,T2,N}
    θ = u[1:N]
    ω = u[N+1:2N]

    x = cumsum(sin.(θ))
    y = cumsum(-cos.(θ))
    dx = cumsum(cos.(θ) .* ω)
    dy = cumsum(sin.(θ) .* ω)

    return vcat(x, y, dx, dy)
end

function cartesian(trajectory::AbstractArray{T,2}, ::NPendulum{T2,N}) where {T,T2,N}
    θ = trajectory[1:N, :]
    ω = trajectory[N+1:2N, :]

    x = cumsum(sin.(θ); dims=1)
    y = cumsum(-cos.(θ); dims=1)
    dx = cumsum(cos.(θ) .* ω; dims=1)
    dy = cumsum(sin.(θ) .* ω; dims=1)

    return vcat(x, y, dx, dy)
end

function cartesian(batched_trajectory::AbstractArray{T,3}, ::NPendulum{T2,N}) where {T,T2,N}
    θ = batched_trajectory[1:N, :, :]
    ω = batched_trajectory[N+1:2N, :, :]

    x = cumsum(sin.(θ); dims=1)
    y = cumsum(-cos.(θ); dims=1)
    dx = cumsum(cos.(θ) .* ω; dims=1)
    dy = cumsum(sin.(θ) .* ω; dims=1)

    return vcat(x, y, dx, dy)
end

function constraints(u::AbstractMatrix, t, ::NPendulum{T,N}) where {T,N}  # no type param on u so the ForwardDiff tests work
    x = u[1:N, :]
    y = u[N+1:2N, :]
    dx = u[2N+1:3N, :]
    dy = u[3N+1:4N, :]

    batch_size = size(u)[2]
    x = vcat(zeros(T, 1, batch_size), x)
    y = vcat(zeros(T, 1, batch_size), y)
    dx = vcat(zeros(T, 1, batch_size), dx)
    dy = vcat(zeros(T, 1, batch_size), dy)

    position_constraints =
        @. (x[2:end, :] - x[1:end-1, :])^2 + (y[2:end, :] - y[1:end-1, :])^2 - 1
    velocity_constraints =
        @. (x[2:end, :] - x[1:end-1, :]) * (dx[2:end, :] - dx[1:end-1, :]) +
           (y[2:end, :] - y[1:end-1, :]) * (dy[2:end, :] - dy[1:end-1, :])

    return vcat(position_constraints, velocity_constraints)
end

function constraints(u::AbstractArray{T,3}, t, system::NPendulum{T,N}) where {T,N}
    return mapslices(u -> constraints(u, t, system), u, dims = (1, 2))
end

function constraints(u::CuMatrix, t, ::NPendulum{T,N}) where {T,N}  # no type param on u so the ForwardDiff tests work
    x = u[1:N, :]
    y = u[N+1:2N, :]
    dx = u[2N+1:3N, :]
    dy = u[3N+1:4N, :]

    batch_size = size(u)[2]
    x = vcat(CUDA.zeros(T, 1, batch_size), x)
    y = vcat(CUDA.zeros(T, 1, batch_size), y)
    dx = vcat(CUDA.zeros(T, 1, batch_size), dx)
    dy = vcat(CUDA.zeros(T, 1, batch_size), dy)

    position_constraints =
        @. (x[2:end, :] - x[1:end-1, :])^2 + (y[2:end, :] - y[1:end-1, :])^2 - 1
    velocity_constraints =
        @. (x[2:end, :] - x[1:end-1, :]) * (dx[2:end, :] - dx[1:end-1, :]) +
           (y[2:end, :] - y[1:end-1, :]) * (dy[2:end, :] - dy[1:end-1, :])

    return vcat(position_constraints, velocity_constraints)
end

function constraints_jacobian(u::AbstractMatrix, t, ::NPendulum{T,N}) where {T,N}
    device = get_device(u)
    batch_size = size(u, 2)

    # Extract state components
    x = view(u, 1:N, :)
    y = view(u, (N+1):(2*N), :)
    dx = view(u, (2*N+1):(3*N), :)
    dy = view(u, (3*N+1):(4*N), :)

    # Add fixed point at origin
    zeros_batch = device(zeros(T, 1, batch_size))
    x = vcat(zeros_batch, x)
    y = vcat(zeros_batch, y)
    dx = vcat(zeros_batch, dx)
    dy = vcat(zeros_batch, dy)

    # Calculate relative differences
    x_diff = x[2:end, :] .- x[1:end-1, :]
    y_diff = y[2:end, :] .- y[1:end-1, :]
    dx_diff = dx[2:end, :] .- dx[1:end-1, :]
    dy_diff = dy[2:end, :] .- dy[1:end-1, :]

    # Create pattern matrices for diagonal and subdiagonal elements
    diag_pattern = device(Matrix(I, N, N))
    
    # Subdiagonal matrix using diagm
    subdiag_pattern = device(diagm(N, N, -1 => ones(T, N-1)))

    # Add batch dimension
    diag_pattern = reshape(diag_pattern, N, N, 1)
    subdiag_pattern = reshape(subdiag_pattern, N, N, 1)
    x_diff = reshape(x_diff, N, 1, batch_size)
    y_diff = reshape(y_diff, N, 1, batch_size)
    dx_diff = reshape(dx_diff, N, 1, batch_size)
    dy_diff = reshape(dy_diff, N, 1, batch_size)

    # Position constraint blocks
    block_pos_x = 2f0 .* x_diff .* diag_pattern .- 2f0 .* x_diff .* subdiag_pattern
    block_pos_y = 2f0 .* y_diff .* diag_pattern .- 2f0 .* y_diff .* subdiag_pattern

    # Velocity constraint blocks
    block_vel_x = dx_diff .* diag_pattern .- dx_diff .* subdiag_pattern
    block_vel_y = dy_diff .* diag_pattern .- dy_diff .* subdiag_pattern
    block_vel_dx = x_diff .* diag_pattern .- x_diff .* subdiag_pattern
    block_vel_dy = y_diff .* diag_pattern .- y_diff .* subdiag_pattern

    # Create zero blocks
    zeros_block = device(zeros(T, N, N, batch_size))

    # Assemble the Jacobian using concatenation
    # Row 1: Position constraints
    row1 = hcat(block_pos_x, block_pos_y, zeros_block, zeros_block)
    
    # Row 2: Velocity constraints
    row2 = hcat(block_vel_x, block_vel_y, block_vel_dx, block_vel_dy)
    
    # Stack rows vertically
    J = vcat(row1, row2)

    return J
end

function ChainRulesCore.rrule(::typeof(constraints_jacobian), u::AbstractMatrix, t, system::NPendulum{T,N}) where {T,N}
    J = constraints_jacobian(u, t, system)
    
    function constraints_jacobian_pullback(Δ_J)
        device = get_device(u)
        batch_size = size(u, 2)
        Δ_J = ChainRulesCore.unthunk(Δ_J)
        
        # Pattern matrices (same as in forward pass)
        diag_pattern = device(Matrix(I, N, N))
        subdiag_pattern = device(diagm(N, N, -1 => ones(T, N-1)))
        diff_pattern = diag_pattern .- subdiag_pattern  # (N, N)
        
        # Split the cotangent
        Δ_pos = Δ_J[1:N, :, :]        # (N, 4N, batch_size) 
        Δ_vel = Δ_J[(N+1):2*N, :, :]  # (N, 4N, batch_size)
        
        # Extract cotangents for each variable block
        Δ_pos_x = Δ_pos[:, 1:N, :]           # (N, N, batch_size)
        Δ_pos_y = Δ_pos[:, (N+1):2*N, :]     # (N, N, batch_size)
        
        Δ_vel_x = Δ_vel[:, 1:N, :]           # (N, N, batch_size)
        Δ_vel_y = Δ_vel[:, (N+1):2*N, :]     # (N, N, batch_size)
        Δ_vel_dx = Δ_vel[:, (2*N+1):3*N, :]  # (N, N, batch_size)
        Δ_vel_dy = Δ_vel[:, (3*N+1):4*N, :]  # (N, N, batch_size)
        
        # Vectorized gradient computation w.r.t. diff variables
        # For each diff variable, sum over the pattern coefficients
        diff_pattern_expanded = reshape(diff_pattern, N, N, 1)  # (N, N, 1) for broadcasting
        
        # Gradients w.r.t. x_diff: from position (factor 2) and velocity constraints
        ∂x_diff = sum(Δ_pos_x .* (2 * diff_pattern_expanded), dims=2)[:, 1, :] .+ 
                  sum(Δ_vel_dx .* diff_pattern_expanded, dims=2)[:, 1, :]  # (N, batch_size)
        
        # Gradients w.r.t. y_diff: from position (factor 2) and velocity constraints  
        ∂y_diff = sum(Δ_pos_y .* (2 * diff_pattern_expanded), dims=2)[:, 1, :] .+
                  sum(Δ_vel_dy .* diff_pattern_expanded, dims=2)[:, 1, :]  # (N, batch_size)
        
        # Gradients w.r.t. dx_diff and dy_diff: only from velocity constraints
        ∂dx_diff = sum(Δ_vel_x .* diff_pattern_expanded, dims=2)[:, 1, :]  # (N, batch_size)
        ∂dy_diff = sum(Δ_vel_y .* diff_pattern_expanded, dims=2)[:, 1, :]  # (N, batch_size)
        
        # Vectorized propagation from diff variables to original state variables
        # Initialize gradient w.r.t. u
        ∂u = device(zeros(T, 4*N, batch_size))
        
        # For x: x_diff[i] = x[i] - x[i-1] where x[0] = 0
        # So ∂x_diff[i]/∂x[j] = δ_{i,j} - δ_{i,j+1}
        ∂u[1:N, :] .= ∂x_diff  # ∂/∂x[i] gets +∂x_diff[i]
        ∂u[1:N-1, :] .-= ∂x_diff[2:N, :]  # ∂/∂x[i-1] gets -∂x_diff[i]
        
        # For y: y_diff[i] = y[i] - y[i-1] where y[0] = 0
        ∂u[(N+1):2*N, :] .= ∂y_diff  # ∂/∂y[i] gets +∂y_diff[i]
        ∂u[(N+1):(2*N-1), :] .-= ∂y_diff[2:N, :]  # ∂/∂y[i-1] gets -∂y_diff[i]
        
        # For dx: dx_diff[i] = dx[i] - dx[i-1] where dx[0] = 0
        ∂u[(2*N+1):3*N, :] .= ∂dx_diff  # ∂/∂dx[i] gets +∂dx_diff[i]
        ∂u[(2*N+1):(3*N-1), :] .-= ∂dx_diff[2:N, :]  # ∂/∂dx[i-1] gets -∂dx_diff[i]
        
        # For dy: dy_diff[i] = dy[i] - dy[i-1] where dy[0] = 0
        ∂u[(3*N+1):4*N, :] .= ∂dy_diff  # ∂/∂dy[i] gets +∂dy_diff[i]
        ∂u[(3*N+1):(4*N-1), :] .-= ∂dy_diff[2:N, :]  # ∂/∂dy[i-1] gets -∂dy_diff[i]
        
        return NoTangent(), ∂u, NoTangent(), NoTangent()
    end
    
    return J, constraints_jacobian_pullback
end

# EXPERIMENT 1: Second order neural ODE in angular coordinates
function get_model(
    ::NPendulum{T,N},
    ::Val{1},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    use_skip = false,
    kwargs...,
) where {T,N}
    # ODE is solved in physical space [θ, ω].
    # MLP sees [sin(θ), cos(θ), normalized_ω] where:
    # - sin/cos are naturally bounded in [-1, 1]
    # - velocities are normalized
    ω_μ = normalization.μ[(N+1):2N, :]
    ω_σ = normalization.σ[(N+1):2N, :]
    ε = normalization.ε

    preprocess_mlp_inputs = @closure x -> begin
        θ = x[1:N, :]
        ω = x[(N+1):2N, :]
        normalized_ω = (ω .- ω_μ) ./ (ω_σ .+ ε)
        return vcat(sin.(θ), cos.(θ), normalized_ω)
    end

    return SecondOrderAutonomousNeuralODE(
        N,
        hidden_layers,
        hidden_width,
        activation,
        rng,
        T,
        device;
        mlp_input_dim = 3N,
        preprocess_mlp_inputs,
        normalization,
        use_skip,
    )
end

function get_constraints(
    ::NPendulum{T,N},
    ::Val{1},
    trajectory::AbstractArray{T},
    times::AbstractVector{T};
    device,
) where {T,N}
    return device(zero(trajectory))  # Placeholder, not used
end

# EXPERIMENT 2: Second order neural ODE in Cartesian coordinates
function get_model(
    ::NPendulum{T,N},
    ::Val{2},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    use_skip = false,
    kwargs...,
) where {T,N}
    return SecondOrderAutonomousNeuralODE(
        2N,
        hidden_layers,
        hidden_width,
        activation,
        rng,
        T,
        device;
        preprocess_mlp_inputs = u -> normalize(u, normalization),
        normalization,
        use_skip,
    )
end

function transform(trajectory, system::NPendulum, ::Val{2})
    return cartesian(trajectory, system)
end

# EXPERIMENT 3: Second order neural ODE in Cartesian coordinates with stabilization
function get_model(
    system::NPendulum{T,N},
    ::Val{3},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    γ,
    backend = MAGMACholesky(),
    use_skip = false,
    kwargs...,
) where {T,N}
    return SecondOrderAutonomousStabilizedNeuralODE(
        2N,
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
        use_skip,
    )
end

function transform(trajectory, system::NPendulum, ::Val{3})
    return cartesian(trajectory, system)
end

# EXPERIMENT 4: Second order neural ODE in Cartesian coordinates with projection
function get_model(
    system::NPendulum{T,N},
    ::Val{4},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    backend = MAGMACholesky(),
    use_skip = false,
    kwargs...,
) where {T,N}
    return SecondOrderAutonomousProjectedNeuralODE(
        2N,
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
        use_skip,
    )
end

function transform(trajectory, system::NPendulum, ::Val{4})
    return cartesian(trajectory, system)
end
