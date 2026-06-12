abstract type AbstractPowerGrid{T} <: AbstractDynamicalSystem{T} end

# IEEE14Bus
struct IEEE14Bus{T<:AbstractFloat,M<:AbstractMatrix{Complex{T}}} <: AbstractPowerGrid{T}
    PQ_nodes::Vector{Int}  # Indexes of PQ nodes
    LY::M                  # Nodal admittance matrix
end

# Constructor with defaults
IEEE14Bus{T}(device::MLDataDevices.AbstractDevice) where {T} = IEEE14Bus(
    # PQ Nodes
    [4, 5, 7, 9, 10, 11, 12, 13, 14],
    # Nodal admittance matrix
    device(get_nodal_admittance_matrix_ieee14bus(T)),
)

# Pretty printing
Base.show(io::IO, ::IEEE14Bus{T,M}) where {T,M} = print(io, "IEEE14Bus{$T,$M}")

function get_nodal_admittance_matrix_ieee14bus(T::DataType)
    y = zeros(Complex{T}, 14, 14)  # Admittances (not yet the nodal admittance matrix)

    y[1, 2] = 4.999131600798035 - 15.263086523179553im
    y[1, 5] = 1.025897454970189 - 4.234983682334831im
    y[2, 3] = 1.1350191923073958 - 4.781863151757718im
    y[2, 4] = 1.686033150614943 - 5.115838325872083im
    y[2, 5] = 1.7011396670944048 - 5.193927397969713im
    y[3, 4] = 1.9859757099255606 - 5.0688169775939205im
    y[4, 5] = 6.840980661495672 - 21.578553981691588im
    y[4, 7] = 0.0 - 4.781943381790359im
    y[4, 9] = 0.0 - 1.7979790715236075im
    y[5, 6] = 0.0 - 3.967939052456154im
    y[6, 11] = 1.9550285631772604 - 4.0940743442404415im
    y[6, 12] = 1.525967440450974 - 3.1759639650294003im
    y[6, 13] = 3.0989274038379877 - 6.102755448193116im
    y[7, 8] = 0.0 - 5.676979846721544im
    y[7, 9] = 0.0 - 9.09008271975275im
    y[9, 10] = 3.902049552447428 - 10.365394127060915im
    y[9, 14] = 1.4240054870199312 - 3.0290504569306034im
    y[10, 11] = 1.8808847537003996 - 4.402943749460521im
    y[12, 13] = 2.4890245868219187 - 2.251974626172212im
    y[13, 14] = 1.1369941578063267 - 2.314963475105352im

    # Make the admittances symmetric
    for i = 1:14
        for j = i:14
            y[j, i] = y[i, j]
        end
    end

    # Build the nodal admittance matrix
    return [i == j ? sum(y[i, :]) : -y[i, j] for i = 1:14, j = 1:14]
end

# Slack bus indices for each power grid system
# For a system with N buses, state vector is [v_re_1, ..., v_re_N, v_im_1, ..., v_im_N]
slack_indices(::IEEE14Bus) = [2, 16]   # Bus 2: real part at index 2, imaginary at index 14+2=16

# Operating points from PowerDynamics.find_operationpoint, converted to
# dataset format [v_re_1, ..., v_re_N, v_im_1, ..., v_im_N].
#! format: off
function operating_point(::IEEE14Bus{T}) where {T}
    T.([
        # v_re_1 .. v_re_14
         0.9862314332465445,  1.0,                  0.965000014918108,
         0.8885264133000343,  0.9144234351165191,    0.915240264285257,
         0.6601087940750454,  0.3798331435405208,    0.7149860831441548,
         0.739185323633917,   0.8209356677192995,    0.8797108436208047,
         0.8614299662953769 ,  0.7497368896100742,
        # v_im_1 .. v_im_14
         0.10221517439575852, -6.408919707129654e-21, -0.15316891595232124,
        -0.07444880794445695, -0.06005303850578621,  -0.2236007589470778,
        -0.10440934099852094, -0.06007777422420501,  -0.14785657861781923,
        -0.16413907838488195, -0.19336527294356398,  -0.2334235977478931,
        -0.22438293141604176, -0.1981795896643274,
    ])
end
#! format: on

"""
    _slack_derivative_mask(system::AbstractPowerGrid{T}, device) where {T}

Create a binary mask that zeros out the time derivative at slack bus indices.
The slack bus has constant voltage, so its derivative is identically zero.
"""
function _slack_derivative_mask(system::AbstractPowerGrid{T}, device) where {T}
    N = size(system.LY, 1)
    mask = ones(T, 2N)
    mask[slack_indices(system)] .= zero(T)
    return device(mask)
end

# Generic constraint functions and jacobians
function constraints(u::AbstractVector, t, system::AbstractPowerGrid)
    (; PQ_nodes, LY) = system
    N = size(LY, 1)
    v = u[1:N] .+ im .* u[N+1:end]
    i = LY * v
    s = v .* conj.(i)
    s_PQ = s[PQ_nodes]
    return vcat(real(s_PQ), imag(s_PQ))
end

function constraints(u::AbstractMatrix, t, system::AbstractPowerGrid)
    (; PQ_nodes, LY) = system
    N = size(LY, 1)
    v = u[1:N, :] .+ im .* u[N+1:end, :]
    i = NNlib.batched_mul(LY, v)
    s = v .* conj.(i)
    s_PQ = s[PQ_nodes, :]
    return vcat(real(s_PQ), imag(s_PQ))
end

function constraints(u::AbstractArray{T,3}, t, system::AbstractPowerGrid{T}) where {T}
    return mapslices(u -> constraints(u, t, system), u, dims = (1, 2))
end

function constraints_jacobian(u::AbstractVector, t, system::AbstractPowerGrid)
    (; PQ_nodes, LY) = system
    device = get_device(u)

    # Get dimensions
    N = size(LY, 1)

    # Separate real and imaginary parts of voltage
    v_re = u[1:N]
    v_im = u[N+1:2*N]

    # Complex voltage and current
    v = v_re .+ im .* v_im
    i = LY * v

    # Extract real and imaginary components
    i_real = real.(i)
    i_imag = imag.(i)
    LY_re = real.(LY)
    LY_im = imag.(LY)

    # Create identity matrix on the correct device
    I_eye = device(LinearAlgebra.I(N))

    # Reshape vectors for broadcasting
    v_re_expanded = reshape(v_re, N, 1)
    v_im_expanded = reshape(v_im, N, 1)
    i_real_expanded = reshape(i_real, N, 1)
    i_imag_expanded = reshape(i_imag, N, 1)

    # Compute the blocks of the Jacobian
    # diag(i_real) + diag(v_re) * LY_re + diag(v_im) * LY_im
    upper_left = i_real_expanded .* I_eye .+ v_re_expanded .* LY_re .+ v_im_expanded .* LY_im
    
    # diag(i_imag) + diag(v_re) * (-LY_im) + diag(v_im) * LY_re  
    upper_right = i_imag_expanded .* I_eye .- v_re_expanded .* LY_im .+ v_im_expanded .* LY_re
    
    # -diag(i_imag) + diag(v_re) * (-LY_im) + diag(v_im) * LY_re
    lower_left = .-i_imag_expanded .* I_eye .- v_re_expanded .* LY_im .+ v_im_expanded .* LY_re
    
    # diag(i_real) - diag(v_re) * LY_re - diag(v_im) * LY_im
    lower_right = i_real_expanded .* I_eye .- v_re_expanded .* LY_re .- v_im_expanded .* LY_im

    # Assemble final Jacobian
    J = [
        upper_left[PQ_nodes, :] upper_right[PQ_nodes, :]
        lower_left[PQ_nodes, :] lower_right[PQ_nodes, :]
    ]

    return J
end

function constraints_jacobian(u::AbstractMatrix, t, system::AbstractPowerGrid)
    (; LY, PQ_nodes) = system
    device = get_device(u)

    # Get dimensions
    N = size(LY, 1)
    M = length(PQ_nodes)
    batch_size = size(u, 2)

    # Separate voltage components
    v_re = @view u[1:N, :]
    v_im = @view u[N+1:2*N, :]

    # Complex voltage
    v = v_re .+ im .* v_im

    # Current
    i = NNlib.batched_mul(LY, v)

    # Extract real and imaginary components
    i_real = real.(i)
    i_imag = imag.(i)
    LY_re = real.(LY)
    LY_im = imag.(LY)

    # Create identity matrix on the correct device
    I_eye = device(LinearAlgebra.I(N))

    # Compute diagonal operations using broadcasting instead of explicit diagonal matrices
    # For diag(a) * B, we can use a .* B where a is reshaped to (N, 1, batch_size)
    v_re_expanded = reshape(v_re, N, 1, batch_size)
    v_im_expanded = reshape(v_im, N, 1, batch_size)
    i_real_expanded = reshape(i_real, N, 1, batch_size)
    i_imag_expanded = reshape(i_imag, N, 1, batch_size)

    # Compute the blocks of the Jacobian
    # diag(i_real) + diag(v_re) * LY_re + diag(v_im) * LY_im
    upper_left = i_real_expanded .* I_eye .+ v_re_expanded .* LY_re .+ v_im_expanded .* LY_im
    
    # diag(i_imag) + diag(v_re) * (-LY_im) + diag(v_im) * LY_re  
    upper_right = i_imag_expanded .* I_eye .- v_re_expanded .* LY_im .+ v_im_expanded .* LY_re
    
    # -diag(i_imag) + diag(v_re) * (-LY_im) + diag(v_im) * LY_re
    lower_left = .-i_imag_expanded .* I_eye .- v_re_expanded .* LY_im .+ v_im_expanded .* LY_re
    
    # diag(i_real) - diag(v_re) * LY_re - diag(v_im) * LY_im
    lower_right = i_real_expanded .* I_eye .- v_re_expanded .* LY_re .- v_im_expanded .* LY_im

    # Extract rows for PQ nodes and concatenate blocks
    upper_left_pq = upper_left[PQ_nodes, :, :]
    upper_right_pq = upper_right[PQ_nodes, :, :]
    lower_left_pq = lower_left[PQ_nodes, :, :]
    lower_right_pq = lower_right[PQ_nodes, :, :]

    # Assemble final Jacobian
    upper_block = cat(upper_left_pq, upper_right_pq, dims=2)
    lower_block = cat(lower_left_pq, lower_right_pq, dims=2)
    J = cat(upper_block, lower_block, dims=1)

    return J
end

# EXPERIMENT 1: Neural ODE
function get_model(
    system::AbstractPowerGrid{T},
    ::Val{1},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    use_skip = false,
    kwargs...,
) where {T}
    N = size(system.LY, 1)
    slack_mask = _slack_derivative_mask(system, device)
    return FirstOrderAutonomousNeuralODE(
        2N,
        hidden_layers,
        hidden_width,
        activation,
        rng,
        T,
        device;
        preprocess_mlp_inputs = u -> normalize(u, normalization),
        postprocess_mlp_outputs = du -> du .* slack_mask,
        normalization,
        use_skip,
    )
end

function transform(u, ::AbstractPowerGrid, ::Val{1})
    return u
end

# EXPERIMENT 2: Stabilized neural ODE
function get_model(
    system::AbstractPowerGrid{T},
    experiment_version::Val{2},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    γ::T,
    backend = MAGMACholesky(),
    use_skip = false,
    kwargs...,
) where {T}
    N = size(system.LY, 1)
    slack_mask = _slack_derivative_mask(system, device)
    return FirstOrderAutonomousStabilizedNeuralODE(
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
        postprocess_mlp_outputs = du -> du .* slack_mask,
        normalization,
        backend,
        use_skip,
    )
end

function transform(u, ::AbstractPowerGrid, ::Val{2})
    return u
end

# EXPERIMENT 3: Projected neural ODE
function get_model(
    system::AbstractPowerGrid{T},
    experiment_version::Val{3},
    device,
    rng,
    activation,
    hidden_layers,
    hidden_width,
    normalization;
    backend = MAGMACholesky(),
    use_skip = false,
    kwargs...,
) where {T}
    N = size(system.LY, 1)
    slack_mask = _slack_derivative_mask(system, device)
    return FirstOrderAutonomousProjectedNeuralODE(
        2N,
        hidden_layers,
        hidden_width,
        activation,
        rng,
        T,
        device,
        system;
        preprocess_mlp_inputs = u -> normalize(u, normalization),
        postprocess_mlp_outputs = du -> du .* slack_mask,
        normalization,
        backend,
        use_skip,
    )
end

function transform(u, ::AbstractPowerGrid, ::Val{3})
    return u
end
